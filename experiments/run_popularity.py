"""First end-to-end run: train the Popularity baseline and report accuracy.

    python -m experiments.run_popularity

Writes per-model metrics to results/ and prints a summary. This is the template
every later model run follows (swap PopularityRecommender for CF/SASRec/LLM).
"""
from __future__ import annotations

import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.models.popularity import PopularityRecommender
from src.utils import paths
from src.utils.seeds import set_seed

K = 10


def main() -> None:
    paths.ensure_dirs()
    set_seed(0)  # popularity is deterministic; seed kept for a uniform run template

    splits = load_splits(write=False)

    model = PopularityRecommender().fit(splits)
    metrics, per_session = evaluate_model(model, splits, k=K)

    print("\n=== Popularity baseline ===")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    out = paths.RESULTS_DIR / "popularity_per_session.parquet"
    per_session.to_parquet(out, index=False)
    pd.DataFrame([{"model": model.name, **metrics}]).to_csv(
        paths.RESULTS_DIR / "popularity_metrics.csv", index=False
    )
    print(f"\nWrote {out}")
    print(f"Wrote {paths.RESULTS_DIR / 'popularity_metrics.csv'}")


if __name__ == "__main__":
    main()
