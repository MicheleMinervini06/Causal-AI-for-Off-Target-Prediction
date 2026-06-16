"""
Diagnostica del Variational SCM.

Misure quantitative per caratterizzare lo stato della latente U:
  1) Statistiche di mu_U sul dataset (collapse encoder?)
  2) Statistiche di sigma_U
  3) KL per esempio (distribuzione, non solo mean)
  4) Correlazione mu_U vs (label, residuo strutturale)
  5) Sensitivity del decoder a U (l'output dipende davvero da U?)
  6) Active units (per latent multi-dim — qui ridotto al caso 1D)

Risultato: dict JSON-serializzabile con summary statistics e una manciata di
percentili (no array grezzi per evitare file enormi).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# Soglia per dichiarare una dimensione di U "attiva" (KL > soglia in mean)
ACTIVE_KL_THRESHOLD = 0.01


def _percentiles(arr: np.ndarray, qs: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _summary_stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation robusto a varianza nulla."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def compute_variational_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Calcola le metriche diagnostiche per il path variational.

    Richiede `model.variational == True`. Itera su `loader` raccogliendo
    mu_U, log_sigma_U, struct_pred, pred_with_U, pred_with_zero_U, labels.
    """
    if not getattr(model, "variational", False):
        return {"status": "skipped", "reason": "model is not variational"}

    model.eval()

    mu_chunks: list[np.ndarray] = []
    log_sigma_chunks: list[np.ndarray] = []
    kl_per_example_chunks: list[np.ndarray] = []
    struct_pred_chunks: list[np.ndarray] = []
    pred_with_U_chunks: list[np.ndarray] = []
    pred_with_zero_U_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    batches_done = 0
    for batch in loader:
        sgrnas = batch["sgrnas"]
        off_targets = batch["off_targets"]
        labels = batch["labels"].to(device)

        context_features = batch.get("context_features")
        if context_features is not None:
            context_features = context_features.to(device)

        # 1) Forward strutturale (no U)
        out_struct = model(sgrnas, off_targets, context_features=context_features, U=None)
        structural_logit = out_struct["logit"].squeeze(-1)
        struct_pred = out_struct["activity_probability"].squeeze(-1)

        # 2) Posterior q(U | structural_logit, y)
        mu_U, log_sigma_U = model.encode_U(structural_logit, labels)
        # Note: encode_U restituisce shape [B, 1]; appiattiamo a [B]
        mu_flat = mu_U.squeeze(-1)
        log_sigma_flat = log_sigma_U.squeeze(-1)

        # 3) KL per esempio (forma chiusa N(mu,sigma) vs N(0,1))
        sigma_sq = (2.0 * log_sigma_flat).exp()
        kl_pe = 0.5 * (mu_flat ** 2 + sigma_sq - 1.0 - 2.0 * log_sigma_flat)

        # 4) Forward con U = mu_U (best estimate, deterministico) e con U = 0
        # Usiamo mu_U (non un sample) per misurare il segnale medio della latente.
        out_with_U = model(sgrnas, off_targets, context_features=context_features, U=mu_flat)
        out_with_zero = model(
            sgrnas, off_targets,
            context_features=context_features,
            U=torch.zeros_like(mu_flat),
        )

        mu_chunks.append(mu_flat.detach().cpu().numpy())
        log_sigma_chunks.append(log_sigma_flat.detach().cpu().numpy())
        kl_per_example_chunks.append(kl_pe.detach().cpu().numpy())
        struct_pred_chunks.append(struct_pred.detach().cpu().numpy())
        pred_with_U_chunks.append(out_with_U["activity_probability"].squeeze(-1).detach().cpu().numpy())
        pred_with_zero_U_chunks.append(out_with_zero["activity_probability"].squeeze(-1).detach().cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())

        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break

    mu_arr = np.concatenate(mu_chunks)
    log_sigma_arr = np.concatenate(log_sigma_chunks)
    sigma_arr = np.exp(log_sigma_arr)
    kl_arr = np.concatenate(kl_per_example_chunks)
    struct_arr = np.concatenate(struct_pred_chunks)
    pred_U_arr = np.concatenate(pred_with_U_chunks)
    pred_zero_arr = np.concatenate(pred_with_zero_U_chunks)
    labels_arr = np.concatenate(label_chunks)

    residual_arr = labels_arr - struct_arr  # gap fattuale-strutturale (in [-1, 1])
    abs_delta_arr = np.abs(pred_U_arr - pred_zero_arr)  # sensitivity decoder a U

    diagnostics: dict[str, Any] = {
        "status": "ok",
        "n_samples": int(mu_arr.size),
        "n_batches": int(batches_done),

        # 1) Encoder: mu_U
        "mu_U": {
            **_summary_stats(mu_arr),
            **_percentiles(mu_arr),
        },

        # 2) Encoder: sigma_U
        "sigma_U": {
            **_summary_stats(sigma_arr),
            **_percentiles(sigma_arr),
        },
        "log_sigma_U": {
            **_summary_stats(log_sigma_arr),
            **_percentiles(log_sigma_arr),
        },

        # 3) KL per esempio
        "kl_per_example": {
            **_summary_stats(kl_arr),
            **_percentiles(kl_arr),
        },

        # 4) Correlazioni — abduzione funzionante?
        "correlations": {
            "pearson_mu_label": _safe_pearson(mu_arr, labels_arr),
            "pearson_mu_residual": _safe_pearson(mu_arr, residual_arr),
            "pearson_mu_struct_pred": _safe_pearson(mu_arr, struct_arr),
        },

        # 5) Sensitivity decoder: |sigma(struct + mu_U) - sigma(struct)|
        "decoder_sensitivity_to_U": {
            **_summary_stats(abs_delta_arr),
            **_percentiles(abs_delta_arr),
        },

        # 6) Active units (semplice indicatore per U scalare)
        "active_units": {
            "kl_threshold": ACTIVE_KL_THRESHOLD,
            "is_active": bool(float(np.mean(kl_arr)) > ACTIVE_KL_THRESHOLD),
            "fraction_examples_above_threshold": float(np.mean(kl_arr > ACTIVE_KL_THRESHOLD)),
        },

        # Verdetto sintetico (interpretazione automatica)
        "verdict": _build_verdict(mu_arr, sigma_arr, kl_arr, abs_delta_arr),
    }

    return diagnostics


def _build_verdict(
    mu_arr: np.ndarray,
    sigma_arr: np.ndarray,
    kl_arr: np.ndarray,
    abs_delta_arr: np.ndarray,
) -> dict[str, Any]:
    """
    Heuristica per etichettare lo stato del modello:
      - 'full_collapse': encoder = prior, decoder ignora U
      - 'encoder_collapse': encoder = prior anche se decoder reattivo
      - 'decoder_ignores_U': encoder informativo ma decoder non lo usa
      - 'healthy': entrambi attivi
    """
    mean_kl = float(np.mean(kl_arr))
    mean_abs_mu = float(np.mean(np.abs(mu_arr)))
    mean_sensitivity = float(np.mean(abs_delta_arr))

    encoder_collapsed = mean_kl < ACTIVE_KL_THRESHOLD
    decoder_inert = mean_sensitivity < 1e-3  # delta < 0.1% in probability

    if encoder_collapsed and decoder_inert:
        label = "full_collapse"
    elif encoder_collapsed:
        label = "encoder_collapse"
    elif decoder_inert:
        label = "decoder_ignores_U"
    else:
        label = "healthy"

    return {
        "label": label,
        "mean_kl": mean_kl,
        "mean_abs_mu": mean_abs_mu,
        "mean_decoder_sensitivity": mean_sensitivity,
    }
