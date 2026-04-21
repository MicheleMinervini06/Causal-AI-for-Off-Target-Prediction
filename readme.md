# crispr-explainability

Repository per predizione off-target CRISPR con focus su explainability biologicamente informata.

## Setup rapido (UV)

Prerequisito: installa `uv`.

```powershell
uv --version
```

Se il comando non esiste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sincronizza ambiente e dipendenze:

```powershell
uv sync --extra dev
```

## Variabili ambiente

```powershell
copy .env.example .env
```

Compila i valori in `.env` (`DATA_DIR`, `RESULTS_DIR`, `WANDB_KEY`).

## Comandi principali

Con `make`:

```powershell
make data
make train
make eval
```

Se `make` non e disponibile su Windows:

```powershell
uv run python -m dag.features --input data/raw --output data/processed/features/features.parquet
uv run crispr-exp01
uv run crispr-exp02
uv run crispr-exp03
uv run crispr-benchmark --results-dir experiments/results --output experiments/results/benchmark_metrics.csv
```

## Principi architetturali

1. `dag/` e indipendente da `models/` e `explainability/`.
2. `evaluation/` usa una interfaccia unificata (`predict_proba`, `explain`) valida per tutti i modelli.
3. `experiments/*/run.py` contiene solo orchestrazione: carica config, invoca moduli, salva output.

## Test

```powershell
uv run pytest -q
```