from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .losses import NeuralSCMLoss, FocalNeuralSCMLoss, compute_irm_penalty

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: Any | None = None,
    current_lambda_irm: float = 0.0,  # Typo corretto
    pos_weight: float = 1.0           # Default a 1.0 come richiesto
) -> dict[str, float]:
    """
    Esegue una singola epoca di addestramento.
    Gestisce i batch come dizionari per massima flessibilità e robustezza.
    Include la regolarizzazione causale IRM.
    """
    model.train()
    
    total_loss = 0.0
    total_pred = 0.0
    total_causal = 0.0
    total_consist = 0.0
    total_irm = 0.0  # Nuovo contatore per tracciare il peso dell'IRM
    steps = 0
    all_train_scores = []
    all_train_labels = []

    for batch in loader:
        optimizer.zero_grad()
        
        # Unpacking esplicito via dizionario
        sgrnas = batch["sgrnas"]
        off_targets = batch["off_targets"]
        labels = batch["labels"].to(device)

        # Estrazione del contesto esogeno (U)
        context_features = batch.get("context_features")
        if context_features is not None:
            context_features = context_features.to(device)

        # FORWARD 1: Batch Osservazionale
        out_base = model(sgrnas, off_targets, context_features=context_features)
        y_pred = out_base["activity_probability"].squeeze(-1)
        all_train_scores.extend(y_pred.detach().cpu().numpy())
        all_train_labels.extend(labels.detach().cpu().numpy())

        out_mut = None
        unaltered_masks = None
        expected_direction = None

        # Estrazione sicura dei dati causali generati on-the-fly
        sgrnas_mut = batch.get("sgrnas_mut")
        off_targets_mut = batch.get("off_targets_mut")

        # FORWARD 2: Batch Intervenuto (se presente nel batch corrente)
        if sgrnas_mut is not None and off_targets_mut is not None:
            out_mut = model(sgrnas_mut, off_targets_mut, context_features=context_features)
            
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

        # --- INTEGRAZIONE IRM (Invariant Risk Minimization) ---
        irm_val = 0.0
        if current_lambda_irm > 0.0:
            # Requisito critico: out_base DEVE contenere la chiave "logit" restituita dal forward del modello
            irm_pen = compute_irm_penalty(
                logits=out_base["logit"].squeeze(-1),
                targets=labels,
                environments=sgrnas,  # Trattiamo l'sgRNA come variabile ambientale
                pos_weight=pos_weight
            )
            loss = loss + (current_lambda_irm * irm_pen)
            irm_val = irm_pen.item()

        loss.backward()
        
        # Gradient Clipping per stabilizzare i Transformer/MLP
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            # OneCycleLR.step() should be called after each optimizer.step()
            try:
                scheduler.step()
            except Exception:
                # Non-fatal: se lo scheduler non supporta step(), ignoriamo
                pass

        # Aggiornamento contatori
        total_loss += loss.item()
        total_pred += loss_dict["loss_pred"].item()
        total_consist += loss_dict.get("loss_consist", torch.tensor(0.0)).item()
        total_causal += loss_dict.get("loss_causal", torch.tensor(0.0)).item()
        total_irm += irm_val
        steps += 1

    # Calcolo metriche
    y_true = np.array(all_train_labels)
    y_scores = np.array(all_train_scores)
    y_pred_binary = (y_scores >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        train_auprc = 0.0
        train_auroc = 0.0
        train_f1 = 0.0
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
        train_auprc = float(average_precision_score(y_true, y_scores))
        train_auroc = float(roc_auc_score(y_true, y_scores))
        train_f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    return {
        "train_loss": total_loss / steps,
        "train_pred_loss": total_pred / steps,
        "train_consist_loss": total_consist / steps,
        "train_causal_loss": total_causal / steps,
        "train_irm_loss": total_irm / steps,  # Utile per monitoraggio W&B
        "train_auprc": train_auprc,
        "train_auroc": train_auroc,
        "train_f1": train_f1,
    }


@torch.no_grad()
def evaluate(
    model: Any, 
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
        
        # Estrazione del contesto esogeno (U) per l'inferenza
        context_features = batch.get("context_features")
        if context_features is not None:
            context_features = context_features.to(device)

        preds = model.predict_proba_batch(sgrnas, off_targets, context_features=context_features)
        
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
    Main training loop con Early Stopping, salvataggio dei pesi migliori, IRM e logging.
    """
    # Si aspetta il `config` completo che contiene la sezione `training`.
    train_cfg = config.get("training", {})
    device = torch.device(train_cfg.get("device", config.get("device", "cpu")))
    model = model.to(device)
    lr = train_cfg.get("learning_rate", 1e-3)
    epochs = train_cfg.get("epochs", 50)
    patience = train_cfg.get("patience", 10)
    
    # Parametri IRM dal Config
    irm_enabled = train_cfg.get("irm_enabled", False)
    lambda_irm_max = train_cfg.get("lambda_irm_max", 1.0)
    irm_warmup = train_cfg.get("irm_warmup_epochs", 5)
    pos_weight_val = train_cfg.get("pos_weight", 1.0)  # Default a 1.0 se non specificato
    
    loss_type = train_cfg.get("loss_type", "bce") 

    if loss_type == "bce":
        logger.info("Inizializzazione NeuralSCMLoss classica (BCE + pos_weight)")
        loss_fn = NeuralSCMLoss(
            pos_weight=pos_weight_val,
            lambda_causal=train_cfg.get("lambda_causal", 0.5),
            lambda_consist=train_cfg.get("lambda_consist", 0.5)
        ).to(device)
    elif loss_type == "focal":
        logger.info("Inizializzazione FocalNeuralSCMLoss")
        loss_fn = FocalNeuralSCMLoss(
            alpha=train_cfg.get("focal_alpha", 0.25),
            gamma=train_cfg.get("focal_gamma", 2.0),
            lambda_causal=train_cfg.get("lambda_causal", 0.01),
            lambda_consist=train_cfg.get("lambda_consist", 0.01)
        ).to(device)
    else:
        raise ValueError(f"Tipo di loss sconosciuto: {loss_type}")    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    try:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=train_cfg.get("pct_start", 0.3),
            cycle_momentum=False,
        )
        logger.info("OneCycleLR scheduler attivato: max_lr=%.6f pct_start=%.2f", lr, train_cfg.get("pct_start", 0.3))

        # # --- NUOVO SCHEDULER: ReduceLROnPlateau ---
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='max',           # Monitoriamo una metrica da massimizzare (AUPRC)
        #     factor=0.5,           # Dimezza il LR quando entra in plateau
        #     patience=3,           # Aspetta 3 epoche senza miglioramenti prima di tagliare
        #     min_lr=1e-6          # Non scendere sotto questo limite
        #)
        #logger.info("ReduceLROnPlateau scheduler attivato (monitoraggio Val AUPRC)")

    except Exception as e:
        scheduler = None
        logger.warning("Impossibile istanziare Scheduler: %s. Continuo senza scheduler.", e)

    best_auprc = -1.0
    best_model_state = None
    epochs_without_improvement = 0

    logger.info(f"Inizio addestramento. Device: {device}, Epoche: {epochs}, IRM Enabled: {irm_enabled}")

    if tracker is not None:
        logger.info("Tracker inizializzato - metriche verranno loggiate.")

    for epoch in range(epochs):
        
        # --- 1. WARMUP DINAMICO IRM ---
        if irm_enabled:
            if epoch < irm_warmup:
                current_lambda_irm = 0.0
            else:
                progress = (epoch - irm_warmup) / max(1, (epochs - irm_warmup))
                current_lambda_irm = lambda_irm_max * progress
        else:
            current_lambda_irm = 0.0

        # --- 2. TRAIN EPOCH ---
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
            current_lambda_irm=current_lambda_irm, # Passiamo il lambda IRM
            pos_weight=pos_weight_val
        )
        
        val_metrics = evaluate(model, val_loader, device)
        current_auprc = val_metrics["auprc"]
        
        # Calcolo della norma L2 totale della rete
        l2_norm = sum(p.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        current_lr = optimizer.param_groups[0]["lr"]

        # # Step dello scheduler ReduceLROnPlateau basato sulla AUPRC di validazione
        # if scheduler is not None:
        #     scheduler.step(current_auprc)

        logger.info(
            f"Epoch {epoch+1:03d} | IRM Lambda: {current_lambda_irm:.2f} | "
            f"Loss: {train_metrics['train_loss']:.4f} "
            f"(P:{train_metrics['train_pred_loss']:.4f} C:{train_metrics['train_causal_loss']:.4f}) | "
            f"Train AUPRC: {train_metrics['train_auprc']:.4f} | "
            f"Train AUROC: {train_metrics['train_auroc']:.4f} | "
            f"Val AUPRC: {current_auprc:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | "
            f"LR: {current_lr:.6f} | L2: {l2_norm:.4f}"
        )

        if tracker is not None:
            tracker_metrics = {
                "epoch": epoch + 1,
                "train/loss_total": train_metrics.get("train_loss", 0.0),
                "train/loss_pred": train_metrics.get("train_pred_loss", 0.0),
                "train/loss_causal": train_metrics.get("train_causal_loss", 0.0),
                "train/auprc": train_metrics.get("train_auprc", 0.0),
                "train/auroc": train_metrics.get("train_auroc", 0.0),
                "train/f1_score": train_metrics.get("train_f1", 0.0),
                "val/auprc": val_metrics.get("auprc", 0.0),
                "val/auroc": val_metrics.get("auroc", 0.0),
                "val/f1_score": val_metrics.get("f1", 0.0),
                "train/learning_rate": current_lr,
                "train/l2_norm": l2_norm
            }
            if irm_enabled:
                tracker_metrics["train/lambda_irm"] = current_lambda_irm
                
            tracker.log_metrics(tracker_metrics, step=epoch + 1)

        # Calcolo del bias effettivo (Comune a tutti)
        try:
            bias_eff = float(torch.clamp(getattr(model, "bias"), min=-4.0, max=3.0).detach().cpu().item())
        except Exception:
            bias_eff = float(getattr(model, "bias").detach().cpu().item())

        # Logging specifico per architettura
        if getattr(model, "architecture", "") == "positional_mlp":
            # Estraiamo i 20 pesi e formattiamoli
            w_pos_eff = -F.softplus(getattr(model, "w_pos")).detach().cpu().numpy()
            w_pos_str = " ".join([f"{w:.2f}" for w in w_pos_eff])
            logger.info(f"   Positional Weights: [{w_pos_str}] | bias={bias_eff:.4f}")
        else:
            # Vecchio logging per i 3 pesi
            w_prox_eff = -F.softplus(getattr(model, "w_proximal")).detach().cpu().item()
            w_nonseed_base = F.softplus(getattr(model, "w_nonseed")).detach().cpu().item()
            w_seed_extra = F.softplus(getattr(model, "w_seed")).detach().cpu().item()
            
            w_nonseed_eff = -w_nonseed_base
            w_seed_eff = -(w_nonseed_base + w_seed_extra)
            
            logger.info(
                f"   Combiner (Effettivi): w_prox={w_prox_eff:.4f} "
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

    model_save_path = config.get("output", {}).get("model_pt", "neural_scm.pt")

    if tracker is not None:
        tracker.log_model_artifact(model_save_path)

    return model