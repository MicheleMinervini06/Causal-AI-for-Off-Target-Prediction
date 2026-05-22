"""Shared utilities for explainability scripts.

Helpers used by:
  - simulate_intervention_batch.py     (batch counterfactual analysis)
  - simulate_intervention.py           (single-pair demo)
  - calibrate_assay_shift.py           (P1 calibration / SMS validation)

The module exposes:
  - Constants:  EPS
  - Numeric helpers:  sigmoid, reads_to_prob, gc_fraction, compute_gc_context_batch
  - SCM helpers:  abduct_U, counterfactual_prob_pct (mode-aware: additive / multiplicative)
  - Model helpers:  load_neural_scm, model_forward_batched
  - Data helpers:  load_positive_dataset, build_offtarget_dataframe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


# ---------- costanti ----------
EPS = 1e-7

DATASET_CSV_PATHS = {
    "changeseq": "data/raw/changeseq/CHANGEseq_positive.csv",
    "guideseq":  "data/raw/guideseq/GUIDEseq_positive.csv",
}
DATASET_READS_COLS = {
    "changeseq": "CHANGEseq_reads",
    "guideseq":  "GUIDEseq_reads",
}


# ---------- utility numeriche vettorizzate ----------

def reads_to_prob(reads: np.ndarray, max_reads: np.ndarray, method: str = "log") -> np.ndarray:
    """Converte read counts in probabilità (%), mitigando il bias di amplificazione PCR.
    Capped at 99% per evitare logit infiniti."""
    reads = np.maximum(0, reads).astype(np.float64)
    max_reads = np.maximum(reads, max_reads).astype(np.float64)
    if method == "log":
        p = np.log1p(reads) / np.log1p(max_reads)
    elif method == "linear":
        p = reads / max_reads
    else:
        raise ValueError(f"Metodo {method} non supportato.")
    return np.minimum(p * 100.0, 99.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_from_prob_pct(prob_pct: np.ndarray) -> np.ndarray:
    """logit di y in % con clipping a (EPS, 1-EPS)."""
    p_unit = np.clip(np.asarray(prob_pct, dtype=np.float64) / 100.0, EPS, 1.0 - EPS)
    return np.log(p_unit / (1.0 - p_unit))


def gc_fraction(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / max(len(seq), 1)


def compute_gc_context_batch(guides: list[str], targets: list[str], device: torch.device) -> torch.Tensor:
    """Restituisce un tensore [B, 3] con (gc_sgRNA, gc_offtarget, gc_delta) per ogni coppia."""
    gc_sg = np.array([gc_fraction(g) for g in guides], dtype=np.float32)
    gc_tg = np.array([gc_fraction(t) for t in targets], dtype=np.float32)
    delta = gc_sg - gc_tg
    arr = np.stack([gc_sg, gc_tg, delta], axis=1)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ---------- abduzione e controfattuale (mode-aware con shift opzionale) ----------

def abduct_U(
    y_obs_prob_pct: np.ndarray,
    struct_logit: np.ndarray,
    pam_gate: np.ndarray,
    pam_mode: str = "additive",
    assay_shift: float = 0.0,
) -> np.ndarray:
    """
    Abduzione mode-aware con supporto per assay shift `b̂` (P1 calibration).

    Modello:
      additive:        y = σ(struct_logit + b_assay + U)
      multiplicative:  y = pam_gate * σ(struct_logit + b_assay + U)

    L'output dell'encoder (out["logit"]) NON include `b_assay`. Quando si vuole
    correggere per un assay diverso da quello di training (es. modello trainato
    su CHANGE-seq applicato a GUIDE-seq), si passa `assay_shift = b̂_(target) − b̂_(source)`
    (o equivalentemente `assay_shift = median(L_true − L_pred)` sul calibration set).

    additive:        U = logit(y_obs) − (struct_logit + assay_shift)
    multiplicative:  U = logit(y_obs / pam_gate) − (struct_logit + assay_shift)
    """
    if pam_mode == "additive":
        p_unit = np.clip(y_obs_prob_pct / 100.0, EPS, 1.0 - EPS)
    elif pam_mode == "multiplicative":
        p_unit = np.clip(y_obs_prob_pct / 100.0 / pam_gate, EPS, 1.0 - EPS)
    else:
        raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")
    return np.log(p_unit / (1.0 - p_unit)) - struct_logit - assay_shift


def counterfactual_prob_pct(
    struct_logit_cf: np.ndarray,
    pam_gate_cf: np.ndarray,
    U: np.ndarray,
    pam_mode: str = "additive",
    assay_shift: float = 0.0,
) -> np.ndarray:
    """y_cf (%) con supporto per assay shift.

    additive:        y_cf = σ(struct_logit_cf + assay_shift + U) * 100
    multiplicative:  y_cf = pam_gate_cf * σ(struct_logit_cf + assay_shift + U) * 100
    """
    if pam_mode == "additive":
        return sigmoid(struct_logit_cf + assay_shift + U) * 100.0
    elif pam_mode == "multiplicative":
        return pam_gate_cf * sigmoid(struct_logit_cf + assay_shift + U) * 100.0
    else:
        raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")


def model_pred_pct(
    struct_logit: np.ndarray,
    pam_gate: np.ndarray,
    pam_mode: str = "additive",
    assay_shift: float = 0.0,
) -> np.ndarray:
    """Predizione del modello in % (mode-aware), senza U (inferenza standard)."""
    if pam_mode == "additive":
        return sigmoid(struct_logit + assay_shift) * 100.0
    return pam_gate * sigmoid(struct_logit + assay_shift) * 100.0


# ---------- forward batched ----------

def model_forward_batched(
    model: NeuralSCM,
    guides: list[str],
    targets: list[str],
    ctx: torch.Tensor,
    batch_size: int,
    intervention: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward batched. Restituisce (struct_logit, pam_gate) come np.ndarray.

    Per positional_mlp, `model.do(intervention)` supporta:
      - intervention["pam_gate"]: valore pre-sigmoid (mult) o pam_logit_contrib (additive)
      - intervention["pos_<i>"]:  penalty value, i in 0..19
    """
    n = len(guides)
    logits = np.empty(n, dtype=np.float32)
    pam_gates = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            if intervention is None:
                out = model(guides[i:j], targets[i:j], context_features=ctx[i:j])
            else:
                out = model.do(
                    guides[i:j], targets[i:j], intervention, context_features=ctx[i:j]
                )
            logits[i:j] = out["logit"].squeeze(-1).cpu().numpy()
            pam_gates[i:j] = out["pam_gate"].squeeze(-1).cpu().numpy()
    return logits, pam_gates


# ---------- model loading ----------

def load_neural_scm(
    model_path: Path | str,
    pam_mode: str = "additive",
    device: torch.device | None = None,
    architecture: str = "positional_mlp",
    hidden_dim: int = 8,
) -> tuple[NeuralSCM, torch.device, int]:
    """Carica un checkpoint di NeuralSCM e ricava il context_dim dal dict.

    Restituisce (model_in_eval_mode, device, context_dim).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(str(model_path), map_location=device)
    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        encoder=encoder,
        architecture=architecture,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        pam_mode=pam_mode,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device, context_dim


# ---------- data loading ----------

def load_positive_dataset(dataset: str) -> tuple[pd.DataFrame, str]:
    """Carica il CSV positivo del dataset richiesto. Restituisce (df, reads_col)."""
    if dataset not in DATASET_CSV_PATHS:
        raise ValueError(f"Dataset non riconosciuto: {dataset}")
    csv_path = DATASET_CSV_PATHS[dataset]
    reads_col = DATASET_READS_COLS[dataset]
    df = pd.read_csv(csv_path)
    return df, reads_col


def build_offtarget_dataframe(
    df: pd.DataFrame,
    reads_col: str,
    on_target_mode: str = "drop",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Costruisce il dataframe di coppie (guida, off-target) valide e arricchito con
    y_obs_off_prob e y_obs_on_prob. Logica identica a quella usata in
    simulate_intervention_batch.py — centralizzata qui per riuso.

    on_target_mode:
      - "drop":       y_obs_on_prob = NaN  (denominatore non disponibile)
      - "per_run":    y_obs_on_prob = reads_to_prob(on_reads, max_reads_in_run)
      - "global_max": y_obs_on_prob = reads_to_prob(on_reads, max_reads_globally)
    """
    # Lookup on-target per name
    on_rows = df[df["distance"] == 0].copy()
    on_lookup = (
        on_rows.sort_values(reads_col, ascending=False)
        .drop_duplicates("name")
        .set_index("name")[["offtarget_sequence", reads_col]]
        .rename(columns={"offtarget_sequence": "on_target_seq", reads_col: "on_reads"})
    )
    if verbose:
        print(f"Guide con on-target di riferimento: {len(on_lookup)}")

    # Off-target rows
    off_df = df[df["distance"] > 0].copy().join(on_lookup, on="name", how="inner")
    off_df["sgRNA"] = off_df["target"].str[:20]
    off_df["off_target"] = off_df["offtarget_sequence"]
    off_df["off_reads"] = off_df[reads_col]

    valid = (
        (off_df["sgRNA"].str.len() == 20)
        & (off_df["off_target"].str.len() == 23)
        & (off_df["on_target_seq"].str.len() == 23)
    )
    dropped = int((~valid).sum())
    if dropped and verbose:
        print(f"[WARN] Scartate {dropped} righe per lunghezze incompatibili")
    off_df = off_df[valid].reset_index(drop=True)
    if verbose:
        print(f"Coppie analizzabili: {len(off_df)}")

    # Probabilità osservate
    off_df["y_obs_off_prob"] = reads_to_prob(off_df["off_reads"].values, off_df["on_reads"].values)

    if on_target_mode == "drop":
        off_df["y_obs_on_prob"] = np.nan
    elif on_target_mode == "per_run":
        if "run" not in off_df.columns:
            raise ValueError(
                f"Dataset non ha colonna 'run', usa on_target_mode='drop' o 'global_max'"
            )
        run_max = df.groupby("run")[reads_col].max().to_dict()
        off_df["run_max_reads"] = off_df["run"].map(run_max).astype(np.float64)
        off_df["y_obs_on_prob"] = reads_to_prob(
            off_df["on_reads"].values, off_df["run_max_reads"].values
        )
        if verbose:
            print(f"Run-level max reads: {run_max}")
    elif on_target_mode == "global_max":
        global_max = float(df[reads_col].max())
        off_df["y_obs_on_prob"] = reads_to_prob(
            off_df["on_reads"].values, np.full(len(off_df), global_max)
        )
        if verbose:
            print(f"Global max reads: {global_max:.0f}")
    else:
        raise ValueError(f"on_target_mode non riconosciuto: {on_target_mode}")

    return off_df


def resolve_on_target_mode(arg_mode: str | None, dataset: str) -> str:
    """Risolve il default di on_target_mode: per_run per guideseq, drop per changeseq."""
    if arg_mode is not None:
        return arg_mode
    return "per_run" if dataset == "guideseq" else "drop"


def filter_saturated_pairs(off_df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, int]:
    """
    Rimuove le coppie saturate: quelle con `off_reads >= on_reads`.

    Una coppia saturata ha `y_obs_off_prob = 99%` per costruzione del cap di
    `reads_to_prob`. Sono coppie con perdita di informazione misurativa
    (cell-free assay hyper-permissivity / cap del sequencing). Vedi F22.1 in
    `doc/findings.md` per la motivazione metodologica.

    Restituisce (filtered_df, n_dropped). Il dataframe atteso deve contenere
    le colonne `off_reads` e `on_reads` (presenti dopo `build_offtarget_dataframe`).
    """
    if "off_reads" not in off_df.columns or "on_reads" not in off_df.columns:
        raise ValueError(
            "filter_saturated_pairs richiede colonne 'off_reads' e 'on_reads' nel df. "
            "Chiama build_offtarget_dataframe prima."
        )
    is_saturated = off_df["off_reads"] >= off_df["on_reads"]
    n_dropped = int(is_saturated.sum())
    n_before = len(off_df)
    filtered = off_df[~is_saturated].reset_index(drop=True)
    if verbose:
        print(
            f"[filter-saturated] Removed {n_dropped} saturated pairs "
            f"({100*n_dropped/max(n_before, 1):.1f}% of {n_before}); kept {len(filtered)}"
        )
    return filtered, n_dropped
