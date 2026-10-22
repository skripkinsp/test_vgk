"""Train/predict two CatBoost quantile models for remaining travel time.

Training requires UNTIL and strictly excludes trips not completed before it.
Prediction uses only graph, vehicles, site, weather and the supplied queries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import CAT_FEATURES, FeatureBuilder
from .queries import create_queries, parse_time


def _trip_completion(data_dir: Path) -> dict[str, int]:
    with (data_dir / "trips.csv").open(newline="", encoding="utf-8") as f:
        return {r["trip_id"]: int(r["arrive_ts"]) for r in csv.DictReader(f)}


def _sample_rate(queries_file: Path, completion: dict, until: int,
                 max_train: int, max_valid: int) -> tuple[int, tuple[float, float]]:
    # Split using trips actually represented in this query file. It also works
    # when a user supplies a limited historical interval for development.
    represented = set()
    with queries_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if completion.get(q["trip_id"], until) < until:
                    represented.add(q["trip_id"])
    arrivals = sorted(completion[t] for t in represented)
    if len(arrivals) < 10:
        raise ValueError("Need at least 10 eligible trips with queries")
    split = arrivals[int(len(arrivals) * .82)]
    counts = [0, 0]
    with queries_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            arrival = completion.get(q["trip_id"], until)
            if arrival >= until:
                continue
            counts[int(arrival >= split)] += 1
    return split, (min(1.0, max_train / max(1, counts[0])),
                   min(1.0, max_valid / max(1, counts[1])))


def _keep(query_id: str, fraction: float) -> bool:
    h = hashlib.blake2b(query_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64 < fraction


def train(data_dir: str, until: int, model_dir: str = "model", queries_file: str | None = None,
          max_train: int = 80000, max_valid: int = 15000, iterations: int = 350) -> None:
    root = Path(data_dir)
    completion = _trip_completion(root)
    eligible = sorted(t for t in completion.values() if t < until)
    if len(eligible) < 10:
        raise ValueError("Need at least 10 completed trips before UNTIL")

    # Default mode reuses the existing public-data-only query generator.
    with tempfile.TemporaryDirectory(prefix="eta_train_") as tmp:
        if queries_file is None:
            create_queries(root, None, until, tmp)
            source = Path(tmp) / "queries.jsonl"
        else:
            source = Path(queries_file)
        split, rates = _sample_rate(source, completion, until, max_train, max_valid)
        builder = FeatureBuilder(root)
        X_train, X_valid, y_train, y_valid = [], [], [], []
        seen_queries = set()
        with source.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                arrival = completion.get(q["trip_id"], until)
                if arrival >= until or int(q["at_ts"]) >= arrival:
                    continue
                # Departure and first-edge entrance may be duplicate triggers.
                key = (q["trip_id"], int(q["at_ts"]), q["current_edge"])
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                section = int(arrival >= split)
                if not _keep(q["query_id"], rates[section]):
                    continue
                target = arrival - int(q["at_ts"])
                row = builder.build(q)
                (X_valid if section else X_train).append(row)
                (y_valid if section else y_train).append(target)

    if not X_train or not X_valid:
        raise ValueError("Empty training or validation partition")
    columns = list(X_train[0])
    train_df = pd.DataFrame(X_train).reindex(columns=columns)
    valid_df = pd.DataFrame(X_valid).reindex(columns=columns)
    # Log relative time: keeps long trips from dominating the regression.
    # Monotone exp maps predicted conditional log quantiles back to time quantiles.
    baseline_train = np.maximum(train_df["baseline_eta_s"].to_numpy(float), 1.0)
    baseline_valid = np.maximum(valid_df["baseline_eta_s"].to_numpy(float), 1.0)
    yt = np.log(np.maximum(y_train, 1) / baseline_train)
    yv = np.log(np.maximum(y_valid, 1) / baseline_valid)
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    for quantile in (.5, .9):
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            iterations=iterations, depth=6, learning_rate=.06,
            l2_leaf_reg=5, thread_count=8, random_seed=42, verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_df, yt, cat_features=CAT_FEATURES,
                  eval_set=(valid_df, yv), early_stopping_rounds=40)
        model.save_model(str(out / f"q{int(quantile * 100)}.cbm"))
        prediction = baseline_valid * np.exp(np.clip(model.predict(valid_df), -5, 5))
        error = np.asarray(y_valid) - prediction
        pinball = np.mean(np.maximum(quantile * error, (quantile - 1) * error))
        print(f"q{int(quantile * 100)}: val pinball={pinball:.2f} s, trees={model.tree_count_}")
    (out / "meta.json").write_text(json.dumps({
        "features": columns, "categorical": CAT_FEATURES,
        "until": until, "validation_from": split,
        "n_train": len(X_train), "n_valid": len(X_valid),
        "target": "log(remaining_s / baseline_eta_s)"
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(X_train)} train and {len(X_valid)} validation rows to {out}")


def predict(data_dir: str, queries_file: str, out_file: str, model_dir: str = "model") -> int:
    """Predict one request at a time with baseline fallback; streaming memory."""
    from .eta_runtime import EtaPredictor
    predictor = EtaPredictor(data_dir, model_dir)
    output = Path(out_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = fallbacks = 0
    fallback_reasons = Counter()
    with Path(queries_file).open(encoding='utf-8') as source, \
         output.open('w', newline='', encoding='utf-8') as dest:
        writer = csv.writer(dest, lineterminator='\n')
        writer.writerow(('query_id', 'p50_s', 'p90_s'))
        for line in source:
            if not line.strip():
                continue
            query = json.loads(line)
            p50, p90, fallback = predictor.predict_one(query)
            writer.writerow((query['query_id'], f'{p50:.6f}', f'{p90:.6f}'))
            n += 1
            fallbacks += int(fallback)
            if fallback:
                fallback_reasons[predictor.last_fallback_reason or 'unspecified'] += 1
    sidecar = output.with_suffix('.fallbacks.json')
    sidecar.write_text(json.dumps({'requests': n, 'fallback_count': fallbacks,
                                   'reasons': dict(fallback_reasons)}, ensure_ascii=False, indent=2) + '\n',
                       encoding='utf-8')
    print(f'Baseline fallbacks: {fallbacks}/{n}')
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="model")
    parser.add_argument("--until")
    parser.add_argument("--queries")
    parser.add_argument("--out")
    parser.add_argument("--iterations", type=int, default=350)
    args = parser.parse_args()
    if args.command == "train":
        until = parse_time(args.until)
        if until is None:
            parser.error("train requires --until")
        train(args.data, until, args.model, args.queries, iterations=args.iterations)
    else:
        if not args.queries or not args.out:
            parser.error("predict requires --queries and --out")
        print(f"Predictions: {predict(args.data, args.queries, args.out, args.model)}")


if __name__ == "__main__":
    main()
