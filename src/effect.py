"""Cross-fitted augmented inverse propensity weighted ATT for recommendations.

Uses only public DATA_DIR inputs. Never reads truth/effect.json.
Causal assumptions: conditional exchangeability given measured pre-decision X,
positivity, consistency and limited interference. The default simulator assigns treatment using observable pre-decision
features, but causal identification still depends on consistency, positivity
and limited interference. Treat the estimate as observational.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

CATS = ['vehicle_id', 'model', 'planned_node', 'recommended_node']
SEED = 42


def _past_arrivals(arrival_times, query_times, window_s=3600):
    """Number of completed empty trips to a load site in the *past* hour."""
    left = np.searchsorted(arrival_times, query_times - window_s, side='left')
    right = np.searchsorted(arrival_times, query_times, side='left')
    return (right-left).astype(np.float64)


def load_data(data_dir):
    data_dir = Path(data_dir)
    rec = pd.read_csv(data_dir / 'recommendations.csv')
    vehicles = pd.read_csv(data_dir / 'vehicles.csv')
    weather = pd.read_csv(data_dir / 'weather.csv')
    trips = pd.read_csv(data_dir / 'trips.csv', usecols=['dest_node', 'loaded', 'arrive_ts'])
    rec = rec.merge(vehicles[['vehicle_id', 'model', 'capacity_t']], on='vehicle_id', how='left', validate='many_to_one')
    rec = rec.sort_values('ts', kind='stable').reset_index(drop=True)
    if rec.empty:
        raise ValueError('No recommendations')
    if rec[['accepted', 'outcome_s', 'ts']].isna().any().any():
        raise ValueError('Missing treatment/outcome/timestamp')
    if not rec['accepted'].isin([0,1]).all() or (rec['outcome_s'] < 0).any():
        raise ValueError('Invalid accepted or outcome_s')
    if rec['accepted'].nunique() != 2:
        raise ValueError('Need both accepted and rejected recommendations')

    # The data generator uses a fixed UTC+05 local clock for Asia/Yekaterinburg.
    local = pd.to_datetime(rec['ts'].to_numpy() + 5*3600, unit='s', utc=True)
    hour = local.hour.to_numpy() + local.minute.to_numpy()/60
    rec['hour_sin'] = np.sin(2*np.pi*hour/24)
    rec['hour_cos'] = np.cos(2*np.pi*hour/24)
    rec['day_index'] = ((rec['ts'] - rec['ts'].min()) // 86400).astype(int)
    rec['night'] = ((hour < 8) | (hour >= 20)).astype(int)
    rec['minutes_to_shift'] = np.minimum(np.abs(hour-8), np.minimum(np.abs(hour-20), np.abs(hour+4)))*60

    # Only available weather measurements at or before recommendation timestamp.
    weather = weather.sort_values('ts')
    past_weather = pd.merge_asof(rec[['ts']], weather, on='ts', direction='backward')
    for col in ['precip_mm_h', 'visibility_m']:
        rec[col] = past_weather[col].fillna(weather[col].iloc[0]).to_numpy()

    empty_arrivals = trips.loc[(trips['loaded']==0) & (trips['arrive_ts']>0), ['dest_node','arrive_ts']]
    arrival_map = {dest: np.sort(group['arrive_ts'].to_numpy()) for dest, group in empty_arrivals.groupby('dest_node')}
    ts = rec['ts'].to_numpy()
    rec['planned_arrivals_1h'] = [0.0]*len(rec)
    rec['suggested_arrivals_1h'] = [0.0]*len(rec)
    for node, indexes in rec.groupby('planned_node').groups.items():
        rec.loc[indexes, 'planned_arrivals_1h'] = _past_arrivals(arrival_map.get(node, np.array([], dtype=np.int64)), ts[np.asarray(indexes)])
    for node, indexes in rec.groupby('recommended_node').groups.items():
        rec.loc[indexes, 'suggested_arrivals_1h'] = _past_arrivals(arrival_map.get(node, np.array([], dtype=np.int64)), ts[np.asarray(indexes)])

    for c in CATS:
        rec[c] = rec[c].fillna('unknown').astype(str)
    rec['capacity_t'] = rec['capacity_t'].fillna(rec['capacity_t'].median())
    cols = CATS + ['capacity_t','est_gain_s','hour_sin','hour_cos','night',
                   'minutes_to_shift','day_index','precip_mm_h','visibility_m',
                   'planned_arrivals_1h','suggested_arrivals_1h']
    return rec, cols


def crossfit_att(rec, cols, iterations=220, seed=SEED, bootstrap=500):
    """Return ATT + percentile CI from day-cluster bootstrap of cross-fit scores."""
    t = rec['accepted'].to_numpy(dtype=np.int8)
    y = rec['outcome_s'].to_numpy(dtype=float)
    days = rec['day_index'].to_numpy(dtype=int)
    n = len(y)
    if n < 100 or t.sum() < 30 or (1-t).sum() < 30:
        raise ValueError('Insufficient treated/control data')
    unique_days = np.unique(days)
    if len(unique_days) < 4:
        raise ValueError('Need at least 4 distinct days for cross-fitting/cluster CI')
    rng = np.random.default_rng(seed)
    shuffled_days = rng.permutation(unique_days)
    fold_by_day = {int(day): int(k % 2) for k, day in enumerate(shuffled_days)}
    fold = np.array([fold_by_day[int(day)] for day in days])
    propensity = np.empty(n, dtype=float)
    control_mean = np.empty(n, dtype=float)
    cat_indices = [cols.index(c) for c in CATS]
    X = rec[cols]
    for heldout in (0, 1):
        train = fold != heldout
        test = ~train
        if np.sum(t[train]) < 30 or np.sum(1-t[train]) < 30:
            raise ValueError('One training fold contains too few treated or controls')
        ps = CatBoostClassifier(iterations=iterations, depth=5, learning_rate=0.06,
                                loss_function='Logloss', verbose=False,
                                thread_count=8, random_seed=seed+heldout,
                                l2_leaf_reg=5, allow_writing_files=False)
        ps.fit(X.loc[train], t[train], cat_features=cat_indices)
        propensity[test] = ps.predict_proba(X.loc[test])[:,1]
        m0 = CatBoostRegressor(iterations=iterations, depth=5, learning_rate=0.06,
                               loss_function='RMSE', verbose=False,
                               thread_count=8, random_seed=seed+7+heldout,
                               l2_leaf_reg=5, allow_writing_files=False)
        controls_train = train & (t == 0)
        m0.fit(X.loc[controls_train], y[controls_train], cat_features=cat_indices)
        control_mean[test] = m0.predict(X.loc[test])
    # Stabilize large inverse propensity weights; report trimming in diagnostics.
    e = np.clip(propensity, .05, .95)
    correction = y - control_mean
    score = t*correction - (1-t)*(e/(1-e))*correction
    att = float(score.sum() / t.sum())
    naive = float(y[t==1].mean()-y[t==0].mean())
    # Day cluster bootstrap of cross-fitted scores (nuisance models held fixed).
    by_day = [(score[days==d].sum(), int(t[days==d].sum())) for d in unique_days]
    day_numer = np.array([z[0] for z in by_day])
    day_denom = np.array([z[1] for z in by_day])
    samples = rng.integers(0, len(unique_days), size=(bootstrap, len(unique_days)))
    denominator = day_denom[samples].sum(axis=1)
    estimates = day_numer[samples].sum(axis=1)/np.maximum(denominator, 1)
    ci = np.percentile(estimates, [2.5,97.5]).tolist()
    result = {'att_s':round(att,3), 'ci95':[round(float(v),3) for v in ci],
              'naive_s':round(naive,3),
              'method':'2-fold cross-fitted AIPW ATT; propensity clipped [0.05,0.95]; day-cluster bootstrap CI (fixed nuisance)'}
    diagnostics = {'n':n,'n_accepted':int(t.sum()),
                   'mean_propensity_accepted':round(float(propensity[t==1].mean()),3),
                   'mean_propensity_rejected':round(float(propensity[t==0].mean()),3),
                   'propensity_clipped_pct':round(float(np.mean((propensity<.05)|(propensity>.95))*100),2),
                   'clusters':len(unique_days),
                   'mean_est_gain_s':round(float(rec['est_gain_s'].mean()),2),
                   'caveat':'Generator causal outcomes use a separate pre-treatment structural model, not the physical fleet loading timestamp. ATT and CI calibration are not guaranteed outside that structural model.'}
    return result, diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='effect.json')
    p.add_argument('--iterations', type=int, default=220)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    rec, cols = load_data(args.data)
    result, diagnostics = crossfit_att(rec, cols, args.iterations, args.seed, args.bootstrap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'effect':result,'diagnostics':diagnostics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
