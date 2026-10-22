"""Paired ETA model diagnostics; run after generating test predictions.

python -m src.diagnostics --data data --test build/test
Outputs: comparison_by_slice.csv, per_query_errors.csv, worst_catboost.csv,
         comparison.txt (all under --test).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_unique(path, keys):
    frame = pd.read_csv(path)
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing {sorted(missing)}')
    if frame.query_id.isna().any() or frame.query_id.duplicated().any():
        raise ValueError(f'{path}: missing or duplicate query_id')
    return frame


def build(data: Path, test: Path) -> pd.DataFrame:
    q = []
    with (test / 'queries.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                q.append({k: item[k] for k in ('query_id', 'trip_id', 'at_ts', 'depart_ts',
                                              'vehicle_id', 'driver_id', 'loaded', 'current_edge')})
    df = pd.DataFrame(q)
    if df.empty or df.query_id.duplicated().any():
        raise ValueError('Empty or duplicate queries')
    for file, name in [('truth.csv', 'truth'), ('baseline.csv', 'base'), ('catboost.csv', 'cb')]:
        required = ['query_id', 'remaining_s'] if name == 'truth' else ['query_id', 'p50_s', 'p90_s']
        x = read_unique(test / file, required)
        if name != 'truth':
            x = x.rename(columns={'p50_s': f'p50_{name}', 'p90_s': f'p90_{name}'})
        df = df.merge(x[['query_id'] + [c for c in x if c != 'query_id']], on='query_id', how='left', validate='one_to_one')
    if df[['remaining_s', 'p50_base', 'p90_base', 'p50_cb', 'p90_cb']].isna().any().any():
        raise ValueError('Missing truth or predictions')
    if not np.isfinite(df[['remaining_s', 'p50_base', 'p90_base', 'p50_cb', 'p90_cb']].to_numpy(float)).all():
        raise ValueError('Non-finite truth or predictions')
    edges = pd.read_csv(data / 'edges.csv', usecols=['edge_id', 'surface'])
    vehicles = pd.read_csv(data / 'vehicles.csv', usecols=['vehicle_id', 'model'])
    df = df.merge(edges.rename(columns={'edge_id': 'current_edge'}), on='current_edge', how='left', validate='many_to_one')
    df = df.merge(vehicles, on='vehicle_id', how='left', validate='many_to_one')
    weather = pd.read_csv(data / 'weather.csv', usecols=['ts', 'precip_mm_h']).sort_values('ts')
    df = pd.merge_asof(df.sort_values('at_ts'), weather, left_on='at_ts', right_on='ts', direction='backward')
    df['rain'] = np.where(df.precip_mm_h.fillna(0) > 0, 'rain', 'dry')
    df['remaining_bucket'] = pd.cut(df.remaining_s, bins=[-np.inf, 180, 600, np.inf], labels=['<=3m', '3-10m', '>10m'])
    df['trip_elapsed_bucket'] = pd.cut(df.at_ts - df.depart_ts, bins=[-np.inf, 0, 180, np.inf], labels=['departure', '0-3m', '>3m'])
    df['loaded_label'] = np.where(df.loaded.astype(int) == 1, 'loaded', 'empty')
    for name in ('base', 'cb'):
        df[f'ae_{name}'] = (df[f'p50_{name}'] - df.remaining_s).abs()
        e90 = df.remaining_s - df[f'p90_{name}']
        df[f'pb90_{name}'] = np.maximum(.9 * e90, -.1 * e90)
        df[f'covered_{name}'] = df.remaining_s <= df[f'p90_{name}']
    df['gain_mae_s'] = df.ae_base - df.ae_cb  # >0: CatBoost better
    df['gain_pb90_s'] = df.pb90_base - df.pb90_cb
    df['winner'] = np.select([df.gain_mae_s > 1e-6, df.gain_mae_s < -1e-6], ['CatBoost', 'Baseline'], default='tie')
    return df


def summarize(df: pd.DataFrame, by: str) -> list[dict]:
    results = []
    groups = [('ALL', df)] if by == 'ALL' else list(df.groupby(by, observed=True, dropna=False))
    for label, part in groups:
        if len(part) == 0:
            continue
        base = float(part.ae_base.mean())
        cb = float(part.ae_cb.mean())
        results.append({
            'slice': by, 'value': str(label), 'queries': len(part), 'trips': part.trip_id.nunique(),
            'mae_baseline_s': round(base, 3), 'mae_catboost_s': round(cb, 3),
            'mae_gain_s': round(base - cb, 3),
            'mae_gain_pct': round(100 * (base-cb)/base, 2) if base > 0 else np.nan,
            'cb_win_rate_pct': round(100 * (part.winner == 'CatBoost').mean(), 2),
            'baseline_win_rate_pct': round(100 * (part.winner == 'Baseline').mean(), 2),
            'p90_pinball_baseline_s': round(float(part.pb90_base.mean()), 3),
            'p90_pinball_catboost_s': round(float(part.pb90_cb.mean()), 3),
            'p90_coverage_baseline_pct': round(100 * part.covered_base.mean(), 2),
            'p90_coverage_catboost_pct': round(100 * part.covered_cb.mean(), 2),
        })
    return results


def run(data='data', test='build/test'):
    test = Path(test)
    df = build(Path(data), test)
    df.to_csv(test / 'per_query_errors.csv', index=False)
    slices = ['ALL', 'remaining_bucket', 'loaded_label', 'rain', 'surface', 'model', 'trip_elapsed_bucket']
    report = pd.DataFrame([row for s in slices for row in summarize(df, s)])
    report.to_csv(test / 'comparison_by_slice.csv', index=False)
    # Worst mistakes relative to baseline, de-duplicated to one representative row per trip
    worst = df.sort_values('gain_mae_s').drop_duplicates('trip_id').head(30)
    worst[['query_id', 'trip_id', 'vehicle_id', 'driver_id', 'surface', 'rain', 'loaded_label',
           'remaining_s', 'p50_base', 'p50_cb', 'ae_base', 'ae_cb', 'gain_mae_s']].to_csv(
        test / 'worst_catboost.csv', index=False)
    overall = report[report['slice'] == 'ALL'].iloc[0]
    # Each trip should have equal weight in this extra check.
    per_trip = df.groupby('trip_id')[['ae_base', 'ae_cb']].mean()
    lines = [f'Queries: {len(df):,}; trips: {df.trip_id.nunique():,}',
             'Positive MAE gain means CatBoost is better.',
             f'MAE: baseline={overall.mae_baseline_s:.2f}s, CatBoost={overall.mae_catboost_s:.2f}s; gain={overall.mae_gain_pct:+.2f}%',
             f'Per-query CatBoost wins: {overall.cb_win_rate_pct:.1f}%; baseline wins: {overall.baseline_win_rate_pct:.1f}%',
             f'Equal-weight-per-trip MAE: baseline={per_trip.ae_base.mean():.2f}s, CatBoost={per_trip.ae_cb.mean():.2f}s',
             'Slices (MAE gains in seconds):']
    for r in report.itertuples(index=False):
        if r.slice != 'ALL':
            lines.append(f'  {r.slice:20} {r.value:12} N={r.queries:6}  gain={r.mae_gain_s:+8.2f}s  win={r.cb_win_rate_pct:5.1f}%')
    lines += ['', 'Breakdowns cannot be identified reliably: edge_events.csv has no breakdown label.',
              'Remaining-time buckets use ground truth and are for evaluation ONLY, not model features.',
              'Query-level observations within the same trip are correlated.']
    text = '\n'.join(lines) + '\n'
    (test / 'comparison.txt').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--test', default='build/test')
    args = parser.parse_args()
    run(args.data, args.test)


if __name__ == '__main__':
    main()
