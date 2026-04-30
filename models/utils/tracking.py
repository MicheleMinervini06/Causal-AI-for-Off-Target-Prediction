import logging
from typing import Any

log = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Adapter agnostico per il tracciamento degli esperimenti.
    Isola la logica di MLOps dal core matematico del modello.
    """
    def __init__(self, config: dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            try:
                import wandb
                # Estraiamo il nome dell'esperimento dal config per organizzare la dashboard
                exp_name = config.get("experiment", {}).get("name", "neural_scm_run")
                
                wandb.init(
                    project="crispr-causal-scm",
                    name=exp_name,
                    config=config
                )
                log.info("W&B Tracker inizializzato. Dashboard attiva.")
            except ImportError:
                log.warning("Libreria 'wandb' non trovata. Esegui 'uv add wandb'. Tracking disabilitato.")
                self.enabled = False

    def watch_model(self, model: Any):
        """Traccia i gradienti e la topologia della rete per diagnosticare vanishing/exploding gradients."""
        if self.enabled:
            import wandb
            wandb.watch(model, log="all", log_freq=10)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Invia un dizionario di metriche alla dashboard."""
        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    def close(self):
        """Chiude formalmente la run sul cloud liberando risorse."""
        if self.enabled:
            import wandb
            wandb.finish()