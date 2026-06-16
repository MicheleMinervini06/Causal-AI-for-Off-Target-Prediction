# crispr-explainability

A **Neural Structural Causal Model (SCM)** for CRISPR/Cas9 off-target prediction.

The goal is not raw predictive performance but **identification**: the architecture is
built to *isolate the invariant cleavage mechanism* `f(X)` (PAM recognition + positional
mismatch penalties along the spacer) from the *exogenous, assay-specific noise* `U`, so
that the model

- **transports across assays** — train *in vitro* on CHANGE-seq, predict *in vivo* on
  GUIDE-seq, with the distribution shift absorbed by a single per-environment offset
  `b(E)` (Sparse Mechanism Shift);
- supports **counterfactual reasoning** — `do(·)` interventions on the guide, the
  off-target, or individual DAG nodes, via algebraic abduction → action → prediction.

This repository accompanies the MSc thesis (`doc/thesis/`) and its defense slides
(`phd-interview/slides/laurea/`). Research is logged finding-by-finding (F1–F27) in
[`doc/findings.md`](doc/findings.md).

---

## Headline results

Adopted model: **`Exp30`** — a `positional_mlp` Neural SCM (≈700 parameters; hard
monotonicity prior `w_pos ≤ 0`, additive PAM gate, 12-dim biological-mismatch encoding,
GC context head, causal-margin weight `λ_causal = 0.1`). Trained on CHANGE-seq and
evaluated cross-assay on GUIDE-seq under **per-sgRNA-disjoint** splits.

| Model | CHANGE-seq AUPRC | GUIDE-seq AUPRC | Counterfactual? |
|---|---:|---:|:--:|
| XGBoost (DAG features) | 0.290 | 0.265 | no |
| CatBoost (DAG features) | 0.394 | 0.329 | no |
| CCLMoff (99 M-param foundation model) | 0.014 | 0.159 | no |
| **Neural SCM (this work, ≈700 params)** | **0.401** | **0.377** | **yes** |

- Statistically tied with the strongest baseline (CatBoost) in-distribution (p = 0.16);
  **+0.048 AUPRC cross-assay** (p < 0.001), DeLong / paired-bootstrap on the matched rows.
- **Causal Consistency Score (CCS)** = 1.00 on the adopted model, and graded 0.40–1.00
  across 11 falsification variants — an evaluation axis the discriminative baselines
  cannot access (no `do(·)` interface).

Full tables and analysis in `doc/thesis/4_results.tex`; the Phase-9 synthesis is at the
end of `doc/findings.md`.

---

## Repository structure

```
dag/                         DAG feature engineering (mismatch, PAM, energetics,
                             independence tests, parametric SCM)
models/
  baseline/                  XGBoost + CatBoost on DAG features
  deep/
    encoding.py              BiologicalMismatch / Pairwise / ContextAware encoders
    modules.py               PAM, spacer-region, mismatch-vector, typed modules
    neural_scm.py            NeuralSCM (positional_mlp adopted; + typed/linear/…)
    train.py                 Focal loss, OneCycleLR, causal-margin regulariser
  utils/
    variational_diagnostics.py   β-VAE abduction diagnostics (the failed alternative)
evaluation/
  ccs.py                     Causal Consistency Score (five do(·) rules)
  metrics.py                 AUROC / AUPRC + bootstrap CIs
  benchmark.py               cross-experiment comparison
explainability/
  explain_thermodynamics.py        positional penalty profile |w_i · φ(type)|
  simulate_intervention.py         single-pair counterfactual
  simulate_intervention_batch.py   population counterfactual + Pareto + U-dist
  run_counterfactual_thesis.py     thesis counterfactual battery → thesis_results/
  characterize_saturated_pairs.py  saturation diagnostic (U bimodality)
  make_*_figure*.py, thesis_plots.py   thesis / defense figures
experiments/
  exp_01_baseline/           XGBoost + CatBoost
  exp_02_scm/                parametric SCM + independence tests
  exp_03_neural_scm/         Neural SCM (run.py + config_exp18…30.yaml)
  results/                   per-run output (Exp24 merged split … Exp30 adopted,
                             cclmoff/ baseline, ccs_falsifiability.json)
doc/
  findings.md                research journal (F1–F27, Phases 1–9)
  thesis/                    LaTeX thesis (0_abstract … 5_conclusions, refs.bib)
  *_presentation.md, metrics.md, plan.md, project_report.md
phd-interview/               open-slide React deck — defense presentation (slides/laurea)
data/                        raw / processed / logs   (local, git-ignored)
tests/                       pytest suite + CCS / statistical-comparison scripts
```

---

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```powershell
uv --version  # if missing:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync --extra dev
copy .env.example .env   # then fill in DATA_DIR, RESULTS_DIR, WANDB_KEY
```

Torch is installed from the CUDA 13.0 index (see `pyproject.toml`); for CPU-only builds
edit the `[tool.uv.sources]` section.

> **Data** lives under `data/` (raw / processed / logs) and is **not versioned** — point
> `DATA_DIR` at your local copy.

---

## Workflow

### Train & benchmark

```powershell
make data    # DAG features → data/processed/features/
make train   # crispr-exp01 (baseline) + exp02 (SCM) + exp03 (Neural SCM)
make eval    # cross-experiment benchmark in experiments/results/
```

Without `make`:

```powershell
uv run python -m dag.features --input data/raw --output data/processed/features/features.parquet
uv run crispr-exp01
uv run crispr-exp02
uv run crispr-exp03
uv run crispr-benchmark --results-dir experiments/results --output experiments/results/benchmark_metrics.csv
```

To train a specific Neural SCM variant, select a config under
`experiments/exp_03_neural_scm/` (`config.yaml` is the default; the ablation grid is
`config_exp18…30_*.yaml`, with **Exp30** the adopted model — each config inherits from a
parent via `_base:`). Checkpoints land in `experiments/results/<run_name>/`.

### Counterfactual analysis

```powershell
# Thesis battery: regional healing hierarchy, node-level do(P_k = 0), PAM canonization
#   → explainability/thesis_results/
uv run python explainability/run_counterfactual_thesis.py

# Population batch: (Δon, Δoff) Pareto + inferred-noise distributions
#   → explainability/batch_results/
uv run python explainability/simulate_intervention_batch.py --dataset guideseq
uv run python explainability/simulate_intervention_batch.py --dataset changeseq

# Single-pair sanity check
uv run python explainability/simulate_intervention.py
```

See F8–F27 in `findings.md` for the correct reading of these outputs.

### Explainability & figures

```powershell
uv run python explainability/explain_thermodynamics.py        # positional penalty profile
uv run python explainability/characterize_saturated_pairs.py  # U bimodality / saturation
uv run python explainability/make_dag_figure.py               # thesis DAG (also make_dag_figure_tufte.py)
```

Figures land in `explainability/plots/` and `explainability/thesis_results/`.

### Causal Consistency Score & statistical comparison

```powershell
uv run pytest tests/test_ccs_adopted.py tests/test_ccs_falsifiability.py
uv run python tests/statistical_comparison.py    # DeLong / paired-bootstrap vs baselines
```

---

## Tests

```powershell
uv run pytest -q
```

---

## Thesis & defense slides

- **Thesis** — `doc/thesis/` (LaTeX: `0_abstract` … `5_conclusions`, `refs.bib`). Chapter 4
  holds the headline results, ablations, the abducted-noise `U` analysis, the cross-assay
  calibration `b(E)`, and the Causal Consistency Score.
- **Defense slides** — `phd-interview/` is a self-contained open-slide React deck; the
  defense lives in `slides/laurea/`:

  ```powershell
  cd phd-interview
  npm install   # first time
  npm run dev   # http://localhost:5173  → open the "laurea" deck
  ```

---

## Architectural principles

1. `dag/` is independent of `models/` and `explainability/`. DAG features are an *input*
   for the baselines and a *structural guide* for the Neural SCM.
2. `evaluation/` exposes a uniform `predict_proba(...)` / `explain(...)` API across all
   models, plus the `do(·)`-based Causal Consistency Score (`evaluation/ccs.py`).
3. `experiments/<exp>/run.py` is orchestration only: load config, invoke modules, save
   output. No model logic inside.
4. The Neural SCM is assembled from **independent modules** in an explicit DAG (PAM gate +
   positional penalties + optional GC context), with two biophysical priors enforced
   *parametrically* rather than as soft regularisation: non-negative per-position penalties
   (`P_i ≥ 0`) and non-positive combiner weights (`w_pos ≤ 0`, monotonicity). Each
   architecture is a different combination of modules on the same interface
   (`models/deep/neural_scm.py`).

---

## Status

**Phase 9 (final).** The split audit (F24), the external CCLMoff baseline (F25), the
`λ_causal` sweet spot (F26), and the merged-split final model (F27) close the empirical
program; the adopted **Exp30** model is the result. The one identified-but-unremoved term
is the evaluation component `b_eval(E)` of the cross-assay calibration (binary-vs-continuous
target mismatch) — discussed in `doc/thesis/5_conclusions.tex` and the end of
`doc/findings.md`.
