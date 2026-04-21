from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from evaluation.bas import biological_alignment_score
from evaluation.caet import cross_assay_explanation_transferability
from evaluation.ess import explanation_stability_score
from evaluation.metrics import evaluate_model
from evaluation.report import write_report


class ExplainableModel(Protocol):
    def predict_proba(self, x: Any) -> np.ndarray:
        ...

    def explain(self, x: Any) -> np.ndarray:
        ...


def _chunk_explanations(explanations: np.ndarray, chunks: int = 3) -> list[np.ndarray]:
    arr = np.asarray(explanations)
    if arr.ndim == 1:
        arr = arr[None, :]
    if len(arr) < chunks:
        return [arr]
    return [c for c in np.array_split(arr, chunks) if len(c) > 0]


def run_benchmark(
    models: Mapping[str, ExplainableModel],
    datasets: Mapping[str, tuple[Any, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evaluate all models across assays with unified performance/explanation metrics."""
    rows: list[dict[str, Any]] = []
    assay_explanations: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}

    for model_name, model in models.items():
        for assay_name, (x, y_true, bio_prior) in datasets.items():
            perf = evaluate_model(model, x, y_true)
            explanations = np.asarray(model.explain(x), dtype=float)
            assay_explanations[model_name][assay_name] = explanations

            ess = explanation_stability_score(_chunk_explanations(explanations))
            bas = biological_alignment_score(explanations, bio_prior)

            rows.append(
                {
                    "model": model_name,
                    "assay": assay_name,
                    **perf,
                    "ess": ess,
                    "bas": bas,
                    "caet": float("nan"),
                }
            )

    for row in rows:
        model_name = str(row["model"])
        assay_name = str(row["assay"])
        explanation_map = assay_explanations[model_name]
        if len(explanation_map) < 2:
            row["caet"] = 1.0
            continue

        reference_assay = next(a for a in explanation_map if a != assay_name)
        row["caet"] = cross_assay_explanation_transferability(
            explanation_map[reference_assay],
            explanation_map[assay_name],
        )

    return pd.DataFrame(rows)


def _collect_metrics_from_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_file in sorted(results_dir.glob("**/metrics.json")):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        exp_name = metrics_file.parent.name
        metrics_block = payload.get("metrics", payload)
        for model_name, metrics in metrics_block.items():
            row = {"experiment": exp_name, "model": model_name}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and report benchmark metrics.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_df = _collect_metrics_from_results(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    write_report(results_df, csv_path=args.output, latex_path=args.output.with_suffix(".tex"))
    print(f"Saved benchmark report to {args.output}")


if __name__ == "__main__":
    main()
