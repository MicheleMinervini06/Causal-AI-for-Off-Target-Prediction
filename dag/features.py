import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Aggiunto per la stratificazione rigorosa
from tqdm import tqdm

from dag.nodes import CRISPRPairFeatures
from dag.mismatch import TYPE_TO_INDEX

STANDARD_COLUMNS = ["sgRNA_seq", "off_seq", "label", "guide_name", "reads", "assay"]

COLUMN_CANDIDATES: dict[str, list[str]] = {
    # Mappa "target" come sequenza della guida
    "sgRNA_seq": ["sgrna_seq", "sgrna", "guide_seq", "guide", "grna", "on_seq", "target"],
    
    # Mappa "offtarget_sequence"
    "off_seq": ["off_seq", "offtarget_seq", "offtarget_sequence", "offtarget", "target_seq"],
    
    "label": ["label", "y", "class", "is_offtarget", "active"],
    
    # Mappa "name" come ID della guida
    "guide_name": ["guide_name", "guide_id", "sgrna_id", "guide", "name"],
    
    # Aggiunti sia "changeseq_reads" che "guideseq_reads"
    "reads": ["reads", "read_count", "counts", "n_reads", "count", "changeseq_reads", "guideseq_reads"],
    
    "assay": ["assay", "dataset", "source", "assay_type"],
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _standardize_columns(raw: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    lower_to_original = {column.lower(): column for column in raw.columns}
    out = pd.DataFrame(index=raw.index)

    for standard_name, candidates in COLUMN_CANDIDATES.items():
        selected = None
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                selected = original
                break
        if selected is not None:
            out[standard_name] = raw[selected]

    if "sgRNA_seq" not in out.columns or "off_seq" not in out.columns:
        missing = [name for name in ["sgRNA_seq", "off_seq"] if name not in out.columns]
        raise ValueError(f"Missing mandatory raw columns: {missing}")

    if "label" not in out.columns:
        out["label"] = 0
    if "guide_name" not in out.columns:
        out["guide_name"] = out["sgRNA_seq"]
    if "reads" not in out.columns:
        out["reads"] = np.nan
    if "assay" not in out.columns:
        out["assay"] = dataset_type

    out["sgRNA_seq"] = out["sgRNA_seq"].astype(str).str.upper().str.replace("U", "T", regex=False)
    out["off_seq"] = out["off_seq"].astype(str).str.upper().str.replace("U", "T", regex=False)
    out["guide_name"] = out["guide_name"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0.0)
    out["label"] = (out["label"] > 0).astype(int)
    out["reads"] = pd.to_numeric(out["reads"], errors="coerce")
    out["assay"] = out["assay"].astype(str)
    return out[STANDARD_COLUMNS]


def load_raw(path: str | Path, dataset_type: str = "auto") -> pd.DataFrame:
    input_path = Path(path)

    if input_path.is_file():
        frames = [_standardize_columns(_read_table(input_path), dataset_type)]
    elif input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet"}
        )
        frames = []
        for file_path in files:
            assay_name = dataset_type if dataset_type != "auto" else file_path.parent.name
            frames.append(_standardize_columns(_read_table(file_path), assay_name))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame, vectors_out: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    mm_vectors: list[np.ndarray] = []
    type_vectors: list[np.ndarray] = []
    energy_vectors: list[np.ndarray] = []
    profiles: list[np.ndarray] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
        pair = CRISPRPairFeatures(
            sgRNA_seq=str(row["sgRNA_seq"]),
            off_seq=str(row["off_seq"]),
        )

        features = pair.to_feature_dict()
        features["assay"] = str(row.get("assay", "unknown"))
        features["guide_name"] = str(row.get("guide_name", row.get("sgRNA_seq", "unknown")))
        features["label"] = int(row.get("label", 0))
        reads_value = row.get("reads", np.nan)
        features["reads"] = float(reads_value) if pd.notna(reads_value) else np.nan

        concepts = pair.to_concept_dict()
        features.update(concepts)

        rows.append(features)
        mm_vectors.append(pair.mm_vector)
        type_vectors.append(pair.type_vector)
        energy_vectors.append(pair.energy_vector)
        profiles.append(pair.to_position_profile())

    feature_df = pd.DataFrame(rows)

    if vectors_out is not None:
        vector_path = Path(vectors_out)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_type_vectors = np.asarray(type_vectors, dtype=np.int8)
        with h5py.File(vector_path, mode="w") as h5:
            h5.create_dataset("mm_vector", data=np.asarray(mm_vectors, dtype=np.int8))
            h5.create_dataset("type_vector", data=encoded_type_vectors)
            h5.create_dataset("energy_vector", data=np.asarray(energy_vectors, dtype=float))
            h5.create_dataset("position_profile", data=np.asarray(profiles, dtype=float))
            h5.attrs["type_encoding"] = str(TYPE_TO_INDEX)
    return feature_df


def create_guide_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Esegue uno split guidato stratificando le guide in base alla 
    loro promiscuità (numero di off-target positivi) per evitare target leakage
    e mantenere un class imbalance omogeneo tra i fold.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    split_df = df.copy()
    if "guide_name" not in split_df.columns:
        split_df["guide_name"] = split_df["sgRNA_seq"].astype(str)

    # 1. Calcolo livello di attività per guida
    guide_stats = split_df.groupby("guide_name")["label"].sum().reset_index()
    guide_stats.rename(columns={"label": "n_positives"}, inplace=True)

    if len(guide_stats) < 3:
        raise ValueError("At least 3 unique guides are required to build train/val/test splits")

    # 2. Stratificazione per quartili
    try:
        guide_stats["activity_bin"] = pd.qcut(
            guide_stats["n_positives"], 
            q=4, 
            labels=["specific", "low_promiscuity", "med_promiscuity", "high_promiscuity"]
        )
    except ValueError:
        # Fallback robusto se le frequenze si accavallano troppo
        guide_stats["activity_bin"] = pd.cut(
            guide_stats["n_positives"], 
            bins=4, 
            labels=["specific", "low_promiscuity", "med_promiscuity", "high_promiscuity"]
        )

    # 3. Split Train+Val vs Test
    train_val_guides, test_guides = train_test_split(
        guide_stats["guide_name"], 
        test_size=test_size, 
        random_state=seed, 
        stratify=guide_stats["activity_bin"]
    )
    
    # 4. Split Train vs Val
    val_relative_size = val_size / (train_size + val_size)
    train_val_stats = guide_stats[guide_stats["guide_name"].isin(train_val_guides)]
    
    train_guides, val_guides = train_test_split(
        train_val_stats["guide_name"], 
        test_size=val_relative_size, 
        random_state=seed, 
        stratify=train_val_stats["activity_bin"]
    )

    # 5. Estrazione DataFrame
    train_df = split_df[split_df["guide_name"].isin(train_guides)].reset_index(drop=True)
    val_df = split_df[split_df["guide_name"].isin(val_guides)].reset_index(drop=True)
    test_df = split_df[split_df["guide_name"].isin(test_guides)].reset_index(drop=True)

    # 6. Logging dei risultati dello split per verifica dell'Imbalance
    print("\n--- Stratified Split Diagnostics ---")
    for name, sub_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        n_pos = sub_df["label"].sum()
        n_guides = sub_df["guide_name"].nunique()
        ratio = n_pos / n_guides if n_guides > 0 else 0
        imbalance = (len(sub_df) - n_pos) / n_pos if n_pos > 0 else 0
        print(f"Split {name.ljust(5)} | Guide: {n_guides:3d} | Positivi: {int(n_pos):5d} | Pos/Guida: {ratio:6.1f} | Imbalance: {imbalance:5.1f}x")
    print("------------------------------------\n")

    return train_df, val_df, test_df


def build_feature_dataframe(pairs: Iterable[CRISPRPairFeatures]) -> pd.DataFrame:
    # Backward compatibility helper for existing tests and notebooks.
    rows: list[dict[str, float | str | int]] = []
    for pair in pairs:
        row = pair.to_feature_dict()
        row["gc_guide"] = float(row["gc_sgRNA"])
        row["gc_target"] = float(row["gc_offtarget"])
        row["seed_mismatch_count"] = int(np.sum(pair.mm_vector[:10]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAG feature pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Raw input file or directory")
    parser.add_argument("--dataset-type", type=str, default="auto", help="changeseq | guideseq | cclmoff | auto")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file")
    parser.add_argument("--vectors-out", type=Path, default=None, help="Optional HDF5 path for vector outputs")
    parser.add_argument("--split-out-dir", type=Path, default=None, help="Optional directory for train/val/test parquet")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_df = load_raw(args.input, dataset_type=args.dataset_type)
    feature_df = build_features(raw_df, vectors_out=args.vectors_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(args.output, index=False)

    if args.split_out_dir is not None:
        args.split_out_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = create_guide_split(
            feature_df,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        train_df.to_parquet(args.split_out_dir / "train.parquet", index=False)
        val_df.to_parquet(args.split_out_dir / "val.parquet", index=False)
        test_df.to_parquet(args.split_out_dir / "test.parquet", index=False)

    print(f"Loaded {len(raw_df)} rows; generated {len(feature_df)} feature rows")


if __name__ == "__main__":
    main()