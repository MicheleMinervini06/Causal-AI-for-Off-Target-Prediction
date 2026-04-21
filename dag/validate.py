import argparse
from pathlib import Path

import pandas as pd


def validate_dag_edges(features_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    expected = {
        "pam_score",
        "mismatch_count",
        "seed_mismatch_count",
        "mean_energy_penalty",
        "total_energy_penalty",
    }
    missing = expected - set(features_df.columns)
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")
        return issues

    bad_pam = ~features_df["pam_score"].between(0, 1)
    if bad_pam.any():
        issues.append("Found pam_score values outside [0, 1].")

    bad_seed = features_df["seed_mismatch_count"] > features_df["mismatch_count"]
    if bad_seed.any():
        issues.append("seed_mismatch_count is greater than mismatch_count in some rows.")

    bad_energy = features_df["mean_energy_penalty"] < 0
    if bad_energy.any():
        issues.append("Found negative mean_energy_penalty values.")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DAG relationships on processed data.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features/features.parquet"),
        help="Parquet file with engineered features",
    )
    args = parser.parse_args()

    if not args.features.exists():
        raise SystemExit(f"Features file not found: {args.features}")

    features_df = pd.read_parquet(args.features)
    errors = validate_dag_edges(features_df)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)

    print("DAG validation passed.")


if __name__ == "__main__":
    main()
