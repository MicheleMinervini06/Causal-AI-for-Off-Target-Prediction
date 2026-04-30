from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .losses import NeuralSCMLoss

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NeuralSCMLoss,
    device: torch.device,
) -> dict[str, float]:
    """
    Esegue una singola epoca di addestramento.
    Gestisce i batch come dizionari per massima flessibilità e robustezza.
    """
    model.train()
    
    total_loss = 0.0
    total_pred = 0.0
    total_causal = 0.0
    total_consist = 0.0
    steps = 0

    for batch in loader:
        optimizer.zero_grad()
        
        # Unpacking esplicito via dizionario (Opzione B)
        sgrnas = batch["sgrnas"]
        off_targets = batch["off_targets"]
        labels = batch["labels"].to(device)

        # FORWARD 1: Batch Osservazionale
        out_base = model(sgrnas, off_targets)
        y_pred = out_base["activity_probability"].squeeze(-1)

        out_mut = None
        unaltered_masks = None
        expected_direction = None

        # Estrazione sicura dei dati causali generati on-the-fly
        sgrnas_mut = batch.get("sgrnas_mut")
        off_targets_mut = batch.get("off_targets_mut")

        # FORWARD 2: Batch Intervenuto (se presente nel batch corrente)
        if sgrnas_mut is not None and off_targets_mut is not None:
            out_mut = model(sgrnas_mut, off_targets_mut)
            
            masks = batch.get("unaltered_masks")
            if masks is not None:
                unaltered_masks = {k: v.to(device) for k, v in masks.items()}
                
            exp_dir = batch.get("expected_direction")
            if exp_dir is not None:
                expected_direction = exp_dir.to(device)

        # Calcolo unificato della Loss
        loss_dict = loss_fn(
            y_pred=y_pred,
            y_true=labels,
            out_base=out_base,
            out_mut=out_mut,
            unaltered_masks=unaltered_masks,
            expected_direction=expected_direction
        )

        loss = loss_dict["loss"]
        loss.backward()
        
        # Gradient Clipping per stabilizzare i Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred += loss_dict["loss_pred"].item()
        total_consist += loss_dict["loss_consist"].item()
        total_causal += loss_dict["loss_causal"].item()
        steps += 1

    return {
        "train_loss": total_loss / steps,
        "train_pred_loss": total_pred / steps,
        "train_consist_loss": total_consist / steps,
        "train_causal_loss": total_causal / steps,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device
) -> dict[str, float]:
    """
    Valuta il modello sul dataset corrente (Validation o Test).
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in loader:
        sgrnas = batch["sgrnas"]
        off_targets = batch["off_targets"]
        labels = batch["labels"]
        
        preds = model.predict_proba_batch(sgrnas, off_targets)
        
        all_preds.extend(preds.cpu().numpy())
        
        # FIX: Trasferimento in CPU prima del cast a numpy per evitare crash CUDA
        all_labels.extend(labels.cpu().numpy())
        
    y_true = np.array(all_labels)
    y_scores = np.array(all_preds)
    y_pred_binary = (y_scores >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {"auprc": 0.0, "auroc": 0.0, "f1": 0.0}

    return {
        "auprc": float(average_precision_score(y_true, y_scores)),
        "auroc": float(roc_auc_score(y_true, y_scores)),
        "f1": float(f1_score(y_true, y_pred_binary))
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    tracker: Any | None = None,
) -> nn.Module:
    """
    Main training loop con Early Stopping, salvataggio dei pesi migliori e logging.
    """
    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)
    
    lr = config.get("learning_rate", 1e-3)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    
    loss_fn = NeuralSCMLoss(
        pos_weight=config.get("pos_weight", 1.0),
        lambda_causal=config.get("lambda_causal", 0.5),
        lambda_consist=config.get("lambda_consist", 0.5)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auprc = -1.0
    best_model_state = None
    epochs_without_improvement = 0

    logger.info(f"Inizio addestramento. Device: {device}, Epoche: {epochs}")

    if tracker is not None:
        tracker.watch_model(model)
        logger.info("Integrazione con tracker attiva.")

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        
        val_metrics = evaluate(model, val_loader, device)
        
        current_auprc = val_metrics["auprc"]
        
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {train_metrics['train_loss']:.4f} "
            f"(P:{train_metrics['train_pred_loss']:.4f} C:{train_metrics['train_causal_loss']:.4f}) | "
            f"Val AUPRC: {current_auprc:.4f} | Val AUROC: {val_metrics['auroc']:.4f}"
        )

        # Logging su tracker (se presente)
        if tracker is not None:
            tracker.log_metrics({
                "epoch": epoch + 1,
                "train/loss_total": train_metrics.get("train_loss", 0.0),
                "train/loss_pred": train_metrics.get("train_pred_loss", 0.0),
                "train/loss_causal": train_metrics.get("train_causal_loss", 0.0),
                "train/loss_consist": train_metrics.get("train_consist_loss", 0.0),
                "val/auprc": val_metrics.get("auprc", 0.0),
                "val/auroc": val_metrics.get("auroc", 0.0),
                "val/f1_score": val_metrics.get("f1", 0.0),
                "system/learning_rate": optimizer.param_groups[0]["lr"]
            })

        # Calcolo dei pesi effettivi per il logging (applichiamo softplus per garantire negatività se usato)
        w_prox_eff = -F.softplus(getattr(model, "w_proximal")).detach().cpu().item()
        w_seed_eff = -F.softplus(getattr(model, "w_seed")).detach().cpu().item()
        w_nonseed_eff = -F.softplus(getattr(model, "w_nonseed")).detach().cpu().item()
        bias_eff = float(getattr(model, "bias").detach().cpu().item())

        logger.info(
            f"  Combiner (Effettivi): w_prox={w_prox_eff:.4f} "
            f"w_seed={w_seed_eff:.4f} w_nonseed={w_nonseed_eff:.4f} bias={bias_eff:.4f}"
        )

        if current_auprc > best_auprc:
            best_auprc = current_auprc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggerato all'epoca {epoch+1}. Miglior Val AUPRC: {best_auprc:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model