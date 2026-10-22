"""Compare baseline and ML ETA forecasts on a held-out query set.

Only evaluation reads ground-truth outcomes and full trip histories.
Usage: python -m src.eval --data data --test build/test --out build/test/eval.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _unique_csv(path: Path, required: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"{path}: missing columns {sorted(absent)}")
    if df['query_id'].isna().any() or df['query_id'].duplicated().any():
        raise ValueError(f"{path}: null or repeated query_id")
    return df


def load_queries(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            rows.append({k: q[k] for k in ('query_id', 'trip_id', 'driver_id', 'loaded', 'at_ts', 'depart_ts')})
    if not rows:
        raise ValueError('No queries')
    df = pd.DataFrame(rows)
    if df.query_id.duplicated().any():
        raise ValueError('queries.jsonl contains duplicate query_id')
    return df


def pinball(y: np.ndarray, p: np.ndarray, alpha: float) -> float:
    e = y - p
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def metrics(df: pd.DataFrame, suffix: str) -> dict[str, float]:
    y = df.remaining_s.to_numpy(float)
    p50 = df[f'p50_s_{suffix}'].to_numpy(float)
    p90 = df[f'p90_s_{suffix}'].to_numpy(float)
    return {
        'n': len(df),
        'mae_s': float(np.mean(np.abs(y - p50))),
        'pinball50_s': pinball(y, p50, .5),
        'pinball90_s': pinball(y, p90, .9),
        'coverage90': float(np.mean(y <= p90)),
        'crossing_pct': float(np.mean(p90 < p50) * 100),
        'bias50_s': float(np.mean(p50 - y)),
    }


def load_context(data: Path, q: pd.DataFrame) -> pd.DataFrame:
    trips = pd.read_csv(data / 'trips.csv', usecols=['trip_id', 'driver_id', 'depart_ts'])
    first_seen = trips.groupby('driver_id')['depart_ts'].min().to_dict()
    # New-driver definition: first ever observed trip starts in the final 3 days
    # of the dataset, rather than inferring from the driver name.
    end = int(trips.depart_ts.max())
    q = q.copy()
    q['new_driver'] = q.driver_id.map(first_seen).ge(end - 3 * 86400)
    q['duration_bucket'] = pd.cut(q.remaining_s, bins=[-1, 180, 600, np.inf],
                                  labels=['<3 min', '3-10 min', '>10 min'])
    weather_path = data / 'weather.csv'
    if weather_path.exists():
        w = pd.read_csv(weather_path, usecols=['ts', 'precip_mm_h']).sort_values('ts')
        w = w.rename(columns={'ts': 'weather_ts'})
        q = pd.merge_asof(q.sort_values('at_ts'), w, left_on='at_ts',
                          right_on='weather_ts', direction='backward')
        q['rain'] = q.precip_mm_h.fillna(0) > 0
    else:
        q['rain'] = False
    return q


def evaluate(data: str | Path, test: str | Path, out: str | Path | None = None) -> str:
    data, test = Path(data), Path(test)
    truth = _unique_csv(test / 'truth.csv', {'query_id', 'remaining_s'})
    queries = load_queries(test / 'queries.jsonl')
    df = queries.merge(truth, on='query_id', how='left', validate='one_to_one')
    if df.remaining_s.isna().any():
        raise ValueError('Some queries lack ground truth')
    for filename, suffix in [('baseline.csv', 'base'), ('catboost.csv', 'cb')]:
        raw = _unique_csv(test / filename, {'query_id', 'p50_s', 'p90_s'})
        raw = raw.rename(columns={'p50_s': f'p50_s_{suffix}', 'p90_s': f'p90_s_{suffix}'})
        df = df.merge(raw, on='query_id', how='left', validate='one_to_one')
        if df[f'p50_s_{suffix}'].isna().any() or df[f'p90_s_{suffix}'].isna().any():
            raise ValueError(f'Missing predictions in {filename}')
    for name in ('remaining_s', 'p50_s_base', 'p90_s_base', 'p50_s_cb', 'p90_s_cb'):
        if not np.isfinite(df[name].to_numpy(float)).all():
            raise ValueError(f'Non-finite values in {name}')
    df = load_context(data, df)
    lines = [f'Evaluation: {len(df):,} queries, {df.trip_id.nunique():,} trips',
             'Lower MAE/pinball is better; p90 coverage ideally ~90%.',
             'MAE is against p50. Model selection must not use this final holdout.', '']

    def section(label: str, part: pd.DataFrame) -> None:
        if part.empty:
            return
        base, cb = metrics(part, 'base'), metrics(part, 'cb')
        diff = (base['mae_s'] - cb['mae_s']) / max(base['mae_s'], 1e-12) * 100
        lines.append(f'[{label}] N={len(part):,}, trips={part.trip_id.nunique():,}; MAE improvement={diff:+.1f}%')
        lines.append('            MAE(s)   PB50(s)   PB90(s)   coverage90   crossing%   bias50(s)')
        for name, m in [('Baseline', base), ('CatBoost', cb)]:
            lines.append(f'{name:9} {m["mae_s"]:8.2f} {m["pinball50_s"]:9.2f} '
                         f'{m["pinball90_s"]:9.2f} {m["coverage90"]:11.1%} '
                         f'{m["crossing_pct"]:10.2f} {m["bias50_s"]:11.2f}')
        lines.append('')

    section('ALL', df)
    for name, mask in [('Loaded', df.loaded.astype(int) == 1),
                       ('Empty', df.loaded.astype(int) == 0),
                       ('Rain now', df.rain), ('Dry now', ~df.rain),
                       ('New driver', df.new_driver), ('Existing driver', ~df.new_driver)]:
        section(name, df[mask])
    for bucket in ('<3 min', '3-10 min', '>10 min'):
        section(f'Remaining {bucket}', df[df.duration_bucket.astype(str) == bucket])
    report = '\n'.join(lines).rstrip() + '\n'
    if out:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding='utf-8')
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--test', default='build/test')
    p.add_argument('--out', default='build/test/eval.txt')
    args = p.parse_args()
    print(evaluate(args.data, args.test, args.out))


if __name__ == '__main__':
    main()
