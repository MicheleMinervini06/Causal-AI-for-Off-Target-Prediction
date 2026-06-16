"""
evaluation/metrics.py
---------------------
Funzione di valutazione unificata per tutti i modelli.

Convenzione input:
    y_pred_proba può essere:
        - array [n] di probabilità per classe 1
        - array [n, 2] di probabilità per entrambe le classi
    In entrambi i casi viene normalizzato a [n] internamente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)

THRESHOLD_DEFAULT = 0.5


def _to_proba_1d(y_pred_proba: np.ndarray) -> np.ndarray:
    """Normalizza output modello a array 1D di probabilità per classe 1."""
    arr = np.asarray(y_pred_proba, dtype=float)
    if arr.ndim == 2:
        return arr[:, 1]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"y_pred_proba deve essere 1D o 2D, trovato shape {arr.shape}")


@dataclass
class EvalResult:
    model_name:   str
    split:        str

    auroc:        float = 0.0
    auprc:        float = 0.0
    f1:           float = 0.0
    balanced_acc: float = 0.0
    threshold:    float = THRESHOLD_DEFAULT

    n_pos_true: int = 0
    n_pos_pred: int = 0
    n_total:    int = 0

    pr_precision: np.ndarray = field(default_factory=lambda: np.array([]))
    pr_recall:    np.ndarray = field(default_factory=lambda: np.array([]))
    roc_fpr:      np.ndarray = field(default_factory=lambda: np.array([]))
    roc_tpr:      np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {
            "model":        self.model_name,
            "split":        self.split,
            "AUPRC":        round(self.auprc, 4),
            "AUROC":        round(self.auroc, 4),
            "F1":           round(self.f1, 4),
            "balanced_acc": round(self.balanced_acc, 4),
            "threshold":    round(self.threshold, 4),
            "n_pos_true":   self.n_pos_true,
            "n_pos_pred":   self.n_pos_pred,
            "n_total":      self.n_total,
        }

    def print(self) -> None:
        print(f"\n{'='*55}")
        print(f"  {self.model_name} - {self.split}")
        print(f"{'='*55}")
        print(f"  AUPRC:         {self.auprc:.4f}  <- main metric")
        print(f"  AUROC:         {self.auroc:.4f}")
        print(f"  F1:            {self.f1:.4f}  (threshold={self.threshold:.3f})")
        print(f"  Balanced Acc:  {self.balanced_acc:.4f}")
        print(f"  Pos predicted: {self.n_pos_pred} / {self.n_pos_true} true")
        print(f"  Total:         {self.n_total}")
        print(f"{'='*55}")


def evaluate_model(
    model_name:    str,
    y_true:        np.ndarray,
    y_pred_proba:  np.ndarray,
    split:         str = "test",
    threshold:     float | None = None,
    store_curves:  bool = True,
) -> EvalResult:
    """
    Valuta un modello su un set di predizioni.

    Args:
        model_name:   Nome del modello per il report.
        y_true:       Label vere (0/1).
        y_pred_proba: Probabilità predette. Accetta [n] o [n, 2].
        split:        Identificatore dello split.
        threshold:    Soglia di classificazione.
                      Se None usa THRESHOLD_DEFAULT (0.5).
                      Passa sempre il threshold ottimizzato sul val set,
                      mai cercare il threshold ottimale sul test set.
        store_curves: Se True salva curve PR e ROC nel risultato.

    Returns:
        EvalResult con tutte le metriche.
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_proba = _to_proba_1d(y_pred_proba)

    if threshold is None:
        threshold = THRESHOLD_DEFAULT

    y_pred = (y_proba >= threshold).astype(int)

    result = EvalResult(
        model_name=model_name,
        split=split,
        threshold=threshold,
        n_pos_true=int(y_true.sum()),
        n_pos_pred=int(y_pred.sum()),
        n_total=len(y_true),
    )

    try:
        result.auroc = float(roc_auc_score(y_true, y_proba))
    except ValueError as e:
        log.warning("AUROC non calcolabile: %s", e)
        result.auroc = float("nan")

    try:
        result.auprc = float(average_precision_score(y_true, y_proba))
    except ValueError as e:
        log.warning("AUPRC non calcolabile: %s", e)
        result.auprc = float("nan")

    result.f1           = float(f1_score(y_true, y_pred, zero_division=0))
    result.balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    if store_curves:
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            result.pr_precision = prec
            result.pr_recall    = rec
        except Exception:
            pass

        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            result.roc_fpr = fpr
            result.roc_tpr = tpr
        except Exception:
            pass

    result.print()
    return result


def find_optimal_threshold(
    y_true:       np.ndarray,
    y_pred_proba: np.ndarray,
    metric:       str = "f1",
) -> float:
    """
    Trova il threshold ottimale sul validation set.

    IMPORTANTE: chiamare SOLO sul val set, mai sul test set.

    Args:
        y_true:       Label vere.
        y_pred_proba: Probabilità predette. Accetta [n] o [n, 2].
        metric:       "f1" | "balanced_acc"

    Returns:
        Threshold ottimale in [0, 1].
    """
    y_proba = _to_proba_1d(y_pred_proba)
    y_true  = np.asarray(y_true, dtype=float)

    thresholds  = np.linspace(0.01, 0.99, 100)
    best_score  = -1.0
    best_thr    = THRESHOLD_DEFAULT

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"metric non supportata: {metric}. Usa 'f1' o 'balanced_acc'.")

        if score > best_score:
            best_score = score
            best_thr   = thr

    log.info(
        "Threshold ottimale (val set, %s=%.4f): %.3f",
        metric, best_score, best_thr,
    )
    return best_thr


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Converte una lista di EvalResult in DataFrame per il report finale."""
    return pd.DataFrame([r.to_dict() for r in results])