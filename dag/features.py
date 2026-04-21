import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from dag.energy import positional_energy_profile
from dag.mismatch import mismatch_type
from dag.nodes import CRISPRPairFeatures
from dag.pam import pam_score


def _gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    seq = sequence.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def _pair_to_row(pair: CRISPRPairFeatures) -> dict[str, float | str | int]:
    length = min(len(pair.guide_seq), len(pair.target_seq))
    mismatch_flags = []
    for i in range(length):
        mismatch_flags.append(mismatch_type(pair.guide_seq[i], pair.target_seq[i]) != "match")

    energy = positional_energy_profile(pair.guide_seq, pair.target_seq)
    mismatch_count = int(sum(mismatch_flags))
    seed_mismatch_count = int(sum(mismatch_flags[:10]))

    return {
        "guide_seq": pair.guide_seq,
        "target_seq": pair.target_seq,
        "pam": pair.pam,
        "assay": pair.assay,
        "enzyme": pair.enzyme,
        "guide_length": pair.guide_length,
        "pam_score": pam_score(pair.pam, pair.enzyme),
        "mismatch_count": mismatch_count,
        "seed_mismatch_count": seed_mismatch_count,
        "gc_guide": _gc_content(pair.guide_seq),
        "gc_target": _gc_content(pair.target_seq),
        "mean_energy_penalty": float(energy.mean()) if len(energy) else 0.0,
        "total_energy_penalty": float(energy.sum()),
    }


def build_feature_dataframe(pairs: Iterable[CRISPRPairFeatures]) -> pd.DataFrame:
    rows = [_pair_to_row(pair) for pair in pairs]
    return pd.DataFrame(rows)


def _load_raw_pairs(input_dir: Path) -> list[CRISPRPairFeatures]:
    pairs: list[CRISPRPairFeatures] = []
    for csv_path in sorted(input_dir.rglob("*.csv")):
        raw = pd.read_csv(csv_path)
        required = {"guide_seq", "target_seq", "pam"}
        if not required.issubset(raw.columns):
            continue

        for _, row in raw.iterrows():
            pairs.append(
                CRISPRPairFeatures(
                    guide_seq=str(row["guide_seq"]),
                    target_seq=str(row["target_seq"]),
                    pam=str(row["pam"]),
                    assay=str(row.get("assay", csv_path.parent.name)),
                    enzyme=str(row.get("enzyme", "SpCas9")),
                )
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DAG features from raw assay files.")
    parser.add_argument("--input", type=Path, required=True, help="Input raw data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    args = parser.parse_args()

    pairs = _load_raw_pairs(args.input)
    features_df = build_feature_dataframe(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(args.output, index=False)
    print(f"Saved {len(features_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
