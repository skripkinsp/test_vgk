"""Measure end-to-end *single request* ETA inference, one CPU thread.

Warm = preloaded graph/model and warmed route cache. Cold = first prediction
with a fresh route cache, excluding model/CSV loading. Both are reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def benchmark(data, model, queries, n=300):
    # Force numeric backends to one thread before importing CatBoost/numpy-heavy ML.
    from .eta_runtime import EtaPredictor
    request = None
    with Path(queries).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                request = json.loads(line)
                break
    if request is None:
        raise ValueError('No queries to benchmark')
    pred = EtaPredictor(data, model)
    cold_start = time.perf_counter_ns()
    _, _, cold_fallback = pred.predict_one(request)
    cold_ms = (time.perf_counter_ns() - cold_start) / 1e6
    # Reuse a preloaded single-request object: this is the hot-path workload.
    latencies = []
    fallbacks = 0
    for _ in range(n):
        started = time.perf_counter_ns()
        _, _, fallback = pred.predict_one(request)
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        fallbacks += int(fallback)
    result = {
        'requests': n, 'cold_first_ms': round(cold_ms, 4),
        'p50_ms': round(float(np.percentile(latencies, 50)), 4),
        'p95_ms': round(float(np.percentile(latencies, 95)), 4),
        'max_ms': round(max(latencies), 4),
        'fallback_count': fallbacks, 'cold_fallback': cold_fallback,
        'ml_loaded': pred.models is not None,
        'meets_hot_p95_10ms': bool(np.percentile(latencies, 95) <= 10 and not fallbacks),
        'note': 'Repeats one query; excludes model loading; only hot single-request latency.',
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--model', default='model')
    parser.add_argument('--queries', default='build/test/queries.jsonl')
    parser.add_argument('--out', default='build/test/bench.json')
    parser.add_argument('--n', type=int, default=300)
    args = parser.parse_args()
    if args.n < 20:
        parser.error('--n must be at least 20')
    result = benchmark(args.data, args.model, args.queries, args.n)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['meets_hot_p95_10ms']:
        raise SystemExit('Benchmark target not met or model fell back to baseline')


if __name__ == '__main__':
    main()
