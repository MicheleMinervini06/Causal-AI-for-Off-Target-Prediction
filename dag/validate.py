import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAG_EDGES_TO_VALIDATE: list[tuple[str, str, str, str, str]] = [
    # ── Test interni: coerenza strutturale del grafo ──────────
    # Verificano che i nodi intermedi si comportino come atteso tra loro.
    ("node_A_pam", "pam_score", "positive", "internal", "[interno] nodo A correla con pam_score"),
    ("node_B_proximal", "mismatch_rate", "positive", "internal", "[interno] nodo B correla con mismatch_rate globale"),
    ("node_C_seed_extension", "mismatch_rate", "positive", "internal", "[interno] nodo C correla con mismatch_rate globale"),
    ("node_D_non_seed", "mismatch_rate", "positive", "internal", "[interno] nodo D correla con mismatch_rate globale"),
    ("mean_energy_penalty", "mismatch_rate", "positive", "internal", "[interno] energia media correla con mismatch_rate"),

    # ── Test esterni: rilevanza predittiva verso l'outcome ─────
    # Verificano che i nodi abbiano effetto associato all'attività osservata.
    ("node_A_pam", "label", "positive", "external", "[esterno] PAM più forte -> più off-target"),
    ("node_B_proximal", "label", "negative", "external", "[esterno] energia PAM-proximal -> meno off-target"),
    ("node_C_seed_extension", "label", "negative", "external", "[esterno] energia seed extension -> meno off-target"),
    ("node_D_non_seed", "label", "negative", "external", "[esterno] energia non-seed -> meno off-target"),
    ("mismatch_count", "label", "negative", "external", "[esterno] mismatch totali -> meno off-target"),
    ("mismatch_rate", "label", "negative", "external", "[esterno] mismatch rate -> meno off-target"),
]


def _direction_from_correlation(value: float) -> str:
    if np.isnan(value):
        return "undefined"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def validate_dag(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for source, target, expected_direction, category, description in DAG_EDGES_TO_VALIDATE:
        if source not in df.columns or target not in df.columns:
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "expected_direction": expected_direction,
                    "category": category,
                    "description": description,
                    "spearman": float("nan"),
                    "direction": "missing",
                    "status": "missing_columns",
                }
            )
            continue

        spearman = float(df[source].corr(df[target], method="spearman"))
        observed_direction = _direction_from_correlation(spearman)
        status = "pass" if observed_direction == expected_direction else "fail"

        rows.append(
            {
                "source": source,
                "target": target,
                "expected_direction": expected_direction,
                "category": category,
                "description": description,
                "spearman": spearman,
                "direction": observed_direction,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def empirical_sensitivity_profile(df: pd.DataFrame) -> np.ndarray:
    profile_columns = [f"profile_pos_{idx:02d}" for idx in range(1, 21)]
    if not set(profile_columns).issubset(df.columns):
        return analytic_prior()

    profile_df = df
    if "label" in df.columns and (df["label"] > 0).any():
        profile_df = df[df["label"] > 0]

    profile = profile_df[profile_columns].mean(axis=0).to_numpy(dtype=float)
    profile = np.clip(profile, a_min=0.0, a_max=None)
    total = float(profile.sum())
    if total <= 0:
        return analytic_prior()
    return profile / total


def analytic_prior(decay_rate: float = 0.20) -> np.ndarray:
    # Position 1 is PAM-proximal, so highest prior mass starts at index 0.
    positions = np.arange(20)
    prior = np.exp(-decay_rate * positions)
    return prior / prior.sum()


def validate_dag_edges(features_df: pd.DataFrame) -> list[str]:
    # Backward compatibility helper used by existing scripts.
    report = validate_dag(features_df)
    issues = []
    for _, row in report.iterrows():
        if row["status"] != "pass":
            issues.append(f"{row['source']} -> {row['target']}: {row['status']}")
    return issues


def _print_report_section(report_df: pd.DataFrame, category: str, title: str) -> None:
    section = report_df[report_df["category"] == category]
    print(f"\n{title}")
    if section.empty:
        print("(nessun edge)")
        return
    print(section.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DAG edges on engineered features")
    parser.add_argument("--features", type=Path, required=True, help="Parquet file with engineered features")
    parser.add_argument("--report-out", type=Path, default=None, help="Optional output CSV for validation report")
    args = parser.parse_args()

    if not args.features.exists():
        raise SystemExit(f"Features file not found: {args.features}")

    features_df = pd.read_parquet(args.features)
    report_df = validate_dag(features_df)

    _print_report_section(report_df, "internal", "Test interni")
    _print_report_section(report_df, "external", "Test esterni")

    print("\nReport completo")
    print(report_df.to_string(index=False))

    empirical_prior = empirical_sensitivity_profile(features_df)
    print("Empirical prior (first 5):", empirical_prior[:5])

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(args.report_out, index=False)


if __name__ == "__main__":
    main()
