
import pandas as pd

truth = pd.read_csv("build/test/truth.csv")
base = pd.read_csv("build/test/baseline.csv")
cb = pd.read_csv("build/test/catboost.csv")

df = truth.merge(base, on="query_id").merge(
    cb, on="query_id", suffixes=("_base", "_cb")
)

y = df["remaining_s"]

for name, suffix in [("Baseline", "base"), ("CatBoost", "cb")]:
    p50 = df[f"p50_s_{suffix}"]
    p90 = df[f"p90_s_{suffix}"]

    mae = (y - p50).abs().mean()
    coverage = (y <= p90).mean()

    print(f"{name}:")
    print(f"  MAE: {mae:.2f} seconds")
    print(f"  p90 coverage: {coverage:.3f}")
