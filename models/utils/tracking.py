import logging
import datetime
import os
from typing import Any

log = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Adapter agnostico per il tracciamento degli esperimenti.
    Isola la logica di MLOps dal core matematico del modello.
    Gestisce automaticamente timestamp, tags, note e salvataggio pesi.
    """
    def __init__(self, config: dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            try:
                import wandb
                
                # Estrazione metadati dal config
                exp_config = config.get("experiment", {})
                base_name = exp_config.get("name", "neural_scm_run")
                tags = exp_config.get("tags", [])
                notes = exp_config.get("notes", "")
                
                # Generazione timestamp per nome run univoco (MeseGiorno_OraMinuto)
                timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                run_name = f"{base_name}_{timestamp}"

                wandb.init(
                    project="crispr-causal-scm",
                    name=run_name,
                    config=config,
                    tags=tags,
                    notes=notes
                )
                log.info(f"W&B Tracker inizializzato. Run Name: {run_name}")
            
            except ImportError:
                log.warning("Libreria 'wandb' non trovata. Esegui 'uv add wandb'. Tracking disabilitato.")
                self.enabled = False
            except OSError as exc:
                log.warning("W&B non si è avviato su questo sistema (%s). Tracking disabilitato.", exc)
                self.enabled = False
            except Exception as exc:
                log.warning("W&B init fallito (%s). Tracking disabilitato.", exc)
                self.enabled = False

    def watch_model(self, model: Any):
        """Traccia i gradienti e la topologia della rete per diagnosticare vanishing/exploding gradients."""
        if self.enabled:
            import wandb
            # Commenta o rimuovi log_freq se rallenta troppo il training
            wandb.watch(model, log="all", log_freq=10)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Invia un dizionario di metriche alla dashboard."""
        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    def log_model_artifact(self, model_path: str, artifact_name: str = "neural_scm_weights"):
        """
        Carica il file dei pesi (.pt) sul cloud di W&B come Artifact.
        Questo garantisce di non perdere mai il modello se sovrascritto in locale.
        """
        if self.enabled and os.path.exists(model_path):
            import wandb
            try:
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                log.info(f"Modello {model_path} salvato in cloud come Artifact '{artifact_name}'.")
            except Exception as exc:
                log.warning("Fallimento nel salvataggio dell'artifact W&B (%s).", exc)

    def close(self):
        """Chiude formalmente la run sul cloud liberando risorse."""
        if self.enabled:
            import wandb
            wandb.finish()