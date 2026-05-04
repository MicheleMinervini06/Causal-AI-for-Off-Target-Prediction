import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class NeuralSCMLoss(nn.Module):
    """
    Funzione di costo multi-obiettivo per l'addestramento del Neural SCM.
    Combina accuratezza predittiva, rigore topologico e direzionalità causale.
    """

    def __init__(self, pos_weight: float = 1.0, lambda_causal: float = 0.5, lambda_consist: float = 0.5):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.lambda_causal = lambda_causal
        self.lambda_consist = lambda_consist

    def predictive_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross Entropy con class weighting.
        y_pred è atteso essere la probabilità post-sigmoide.
        """
        weight = torch.where(
            y_true == 1.0, 
            self.pos_weight.expand_as(y_true), 
            torch.ones_like(y_true)
        )
        return F.binary_cross_entropy(y_pred, y_true, weight=weight)

    def consistency_loss(self, out_base: dict, out_mut: dict, unaltered_masks: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Penalizza la rete se moduli topologicamente non correlati alla mutazione
        cambiano il loro output scalare.
        """
        loss = torch.tensor(0.0, device=out_base["activity_probability"].device)
        
        for region in ["proximal", "seed", "nonseed"]:
            if region in unaltered_masks:
                mask = unaltered_masks[region].float() # (B,)
                
                base_scalar = out_base[f"{region}_scalar"].squeeze(-1) # (B,)
                mut_scalar = out_mut[f"{region}_scalar"].squeeze(-1)   # (B,)
                
                diff_sq = (base_scalar - mut_scalar) ** 2
                loss += (diff_sq * mask).mean()
                
        return loss

    def causal_loss(self, prob_base: torch.Tensor, prob_intervened: torch.Tensor, expected_direction: torch.Tensor) -> torch.Tensor:
        """
        Directional Margin Loss (Monotonicità Causale).
        """
        margin = 0.05
        
        # Se prob_intervened > prob_base, delta_pred è positivo
        delta_pred = prob_intervened - prob_base 
        
        # Penalizziamo se il modello va nella direzione opposta a expected_direction
        violation = delta_pred * (-expected_direction) + margin
        loss = F.relu(violation)
        
        # FIX: Azzera la loss costante per gli interventi ininfluenti (expected_direction == 0)
        active_mask = (expected_direction != 0.0).float()
        
        # Evita divisioni per zero se il batch intero è neutro
        if active_mask.sum() == 0:
            return torch.tensor(0.0, device=loss.device)
            
        return (loss * active_mask).sum() / active_mask.sum()

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor,
        out_base: dict | None = None,
        out_mut: dict | None = None,
        unaltered_masks: dict[str, torch.Tensor] | None = None,
        expected_direction: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Aggrega le loss. 
        
        CONTRATTO PER IL TRAINING LOOP:
        - Batch Standard: Passare solo y_pred e y_true. (Calcola solo pred_loss).
        - Batch Contrastivo (Topologia): Passare out_base, out_mut e unaltered_masks.
        - Batch Causale (Direzionalità): Passare out_base, out_mut e expected_direction.
        - Batch Misto: Passare tutti i parametri.
        """
        loss_pred = self.predictive_loss(y_pred, y_true)
        
        loss_consist = torch.tensor(0.0, device=loss_pred.device)
        if out_base is not None and out_mut is not None and unaltered_masks is not None:
            loss_consist = self.consistency_loss(out_base, out_mut, unaltered_masks)
            
        loss_causal = torch.tensor(0.0, device=loss_pred.device)
        if out_base is not None and out_mut is not None and expected_direction is not None:
            prob_base = out_base["activity_probability"].squeeze(-1)
            prob_intervened = out_mut["activity_probability"].squeeze(-1)
            loss_causal = self.causal_loss(prob_base, prob_intervened, expected_direction)

        total_loss = loss_pred + (self.lambda_consist * loss_consist) + (self.lambda_causal * loss_causal)

        return {
            "loss": total_loss,
            "loss_pred": loss_pred,
            "loss_consist": loss_consist,
            "loss_causal": loss_causal
        }
    
class FocalNeuralSCMLoss(NeuralSCMLoss):
    """
    Estensione di NeuralSCMLoss che sostituisce la BCE con la Focal Loss.
    Eredita la logica causale e di consistenza dalla classe base.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_causal: float = 0.01,
        lambda_consist: float = 0.01,
    ):
        # Non registriamo `pos_weight` per la Focal Loss: inizializzazione leggera
        nn.Module.__init__(self)

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_causal = lambda_causal
        self.lambda_consist = lambda_consist

    def predictive_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calcola la Focal Loss a partire dalle probabilità `y_pred`.
        Non utilizza `pos_weight` (contratto senza pesatura delle classi).
        """
        eps = 1e-6
        p = y_pred.clamp(min=eps, max=1.0 - eps)

        # elementwise binary cross entropy
        ce = F.binary_cross_entropy(p, y_true.float(), reduction="none")

        # probabilità corrette per y_true
        p_t = p * y_true + (1.0 - p) * (1.0 - y_true)
        alpha_t = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)

        focal = alpha_t * ((1.0 - p_t) ** self.gamma) * ce

        return focal.mean()
    

def compute_irm_penalty(logits: torch.Tensor, targets: torch.Tensor, environments: list[str], pos_weight: float) -> torch.Tensor:
    """
    Calcola l'IRMv1 penalty raggruppando i campioni del batch per guida.
    """
    device = logits.device
    penalty = torch.tensor(0.0, device=device)
    
    # Trova gli ID univoci delle guide in questo batch
    unique_envs = list(set(environments))
    
    # Se c'è solo 1 guida nel batch, l'IRM non ha termini di paragone
    if len(unique_envs) <= 1:
        return penalty

    # Il classificatore fittizio (w=1.0) di cui calcoleremo il gradiente
    scale = torch.tensor(1.0, device=device, requires_grad=True)
    valid_envs = 0

    for env in unique_envs:
        # Crea una maschera booleana per i campioni di questa specifica guida
        mask = torch.tensor([e == env for e in environments], device=device, dtype=torch.bool)
        
        if mask.sum() < 2:
            continue  # Salta ambienti con 1 solo campione (gradiente instabile)
            
        env_logits = logits[mask]
        env_targets = targets[mask]
        
        # BCE Loss classica applicata SOLO a questo ambiente, scalata dal classificatore fittizio
        env_loss = F.binary_cross_entropy_with_logits(
            env_logits * scale, 
            env_targets, 
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        
        # Calcola il gradiente della loss dell'ambiente rispetto a 'scale'
        grad = autograd.grad(env_loss, [scale], create_graph=True)[0]
        
        # IRMv1: Somma la norma al quadrato del gradiente
        penalty += torch.sum(grad ** 2)
        valid_envs += 1

    return penalty / valid_envs if valid_envs > 0 else penalty