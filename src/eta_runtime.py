"""Reusable ETA inference with a mandatory per-request baseline fallback.

Model loading happens once. Each query is evaluated with the public graph and
observations only; no trips.csv, truth.csv or full edge telemetry is loaded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from .baseline import predict_one as baseline_one
from .road_context import RoadContext


class EtaPredictor:
    def __init__(self, data_dir='data', model_dir='model'):
        self.graph = RoadContext(data_dir)
        self.builder = None
        self.models = None
        self.columns = None
        self.load_error = None
        self.last_fallback_reason = None
        try:
            from .features import FeatureBuilder
            root = Path(model_dir)
            meta = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
            self.columns = meta['features']
            self.builder = FeatureBuilder(data_dir)
            self.models = []
            for name in ('q50.cbm', 'q90.cbm'):
                model = CatBoostRegressor()
                model.load_model(str(root / name))
                self.models.append(model)
        except (OSError, ValueError, ImportError, KeyError, RuntimeError) as exc:
            self.load_error = str(exc)
            self.builder = None
            self.models = None

    def predict_one(self, query: dict) -> tuple[float, float, bool]:
        """(p50, p90, used_fallback). Invalid graph queries are not recoverable."""
        base50, base90 = baseline_one(query, self.graph)
        self.last_fallback_reason = None
        if self.models is None:
            self.last_fallback_reason = "model_load: " + str(self.load_error)[:160]
            return base50, base90, True
        try:
            row = self.builder.build(query)
            frame = pd.DataFrame([row], columns=self.columns)
            values = []
            for model in self.models:
                correction = float(model.predict(frame, thread_count=1)[0])
                values.append(max(0.0, base50 * math.exp(max(-5., min(5., correction)))))
            p50, p90 = values
            if not (math.isfinite(p50) and math.isfinite(p90)):
                raise ValueError('Non-finite prediction')
            return p50, max(p50, p90), False
        except Exception as exc:
            # A per-query ML failure must never abort the batch.
            self.last_fallback_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            return base50, base90, True
