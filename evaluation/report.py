from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(results_df: pd.DataFrame, csv_path: str | Path, latex_path: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    if latex_path is not None:
        latex_path = Path(latex_path)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(results_df.to_latex(index=False), encoding="utf-8")
