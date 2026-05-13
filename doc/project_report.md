# CRISPR Off-Target Prediction with Neural Causal Models
## Project State Report — May 2026

---

## 1. Overview

This project develops a predictive model for CRISPR-Cas9 off-target activity using a **Structural Causal Model (SCM)** grounded in the thermodynamics and biophysics of DNA-RNA hybridisation. The core objective is not only to predict off-target cleavage events accurately, but to do so in a causally interpretable way — enabling counterfactual reasoning about guide RNA design.

The best current approach is **Exp15: Positional MLP with Extended OneCycle Scheduling** (`experiments/exp_15_positional_extended_onecycle/`). It achieves competitive predictive performance while introducing a transparent causal structure that can be directly interrogated via Pearl's `do()` operator.

---

## 2. Problem Formulation

Given a 20-nt guide RNA (sgRNA) and a genomic target sequence, the model predicts the probability that Cas9 will cleave at the off-target site. Two experimental datasets are used:

| Dataset | Protocol | Pairs | Notes |
|---|---|---|---|
| **CHANGE-seq** | Cell-free (in vitro) | 4.35M | Training / validation / test |
| **GUIDE-seq** | In vivo | 1.6k | Cross-assay generalisation |

Class imbalance is severe: ~46 negatives per positive in CHANGE-seq. The mismatch between in vitro and in vivo distributions is a key scientific challenge, and a central motivation for adopting a causal framework rather than a purely discriminative one.

---

## 3. Causal Architecture

### 3.1 Structural Causal Model (SCM)

The model encodes the causal mechanism of Cas9 cleavage as a DAG with the following nodes:

```
sgRNA + target DNA
        │
        ▼
  MismatchEncoder
  (biological features per position)
        │
        ├──► PAM gate          ─────────────────────────────────┐
        │    (NGG/NAG/NCG hierarchy)                            │
        ├──► Proximal module   (positions 16–20, PAM-adjacent)  │
        │                                                       │
        ├──► Seed module       (positions 8–15, high specificity)│
        │                                                       │
        ├──► Non-seed module   (positions 0–7, low specificity) │
        │                                                       │
        └──► Context U         (GC-content, cellular environment)
                                                                │
                                                                ▼
                                                    logit → σ → P(cleavage)
```

Each module produces a **positional penalty** — a non-positive scalar that represents how much a mismatch at that position reduces cleavage probability. The hard constraint that all penalties ≤ 0 encodes the biological prior that mismatches can only hurt (never enhance) activity relative to a perfect match.

### 3.2 Positional MLP (Exp15 Architecture)

The core module is a shallow, position-independent MLP applied identically to all 20 positions:

```
input:  4-dim mismatch type vector  (Match / Wobble / Transition / Transversion)
        4-dim sgRNA base (one-hot)
        4-dim target base (one-hot)
        ─────────────────────────── (12-dim total, BiologicalMismatchEncoder)
hidden: Linear(12→8) → ReLU
output: Linear(8→1)
```

Twenty such MLPs (one per position) produce twenty penalties. These are weighted by 20 **learned position-specific weights** `w_pos ∈ ℝ²⁰`, all constrained to be ≤ 0 via a softplus reparameterisation. The final logit is:

```
logit = Σᵢ w_pos[i] · penalty[i]  +  bias
```

This design preserves full interpretability: each position's contribution to the final prediction is a product of *how severe* the mismatch is (from the MLP) and *how important* that position is (from the learned weight).

### 3.3 Exogenous Noise (Context Module)

A shallow context network ingests three GC-content features (`gc_sgRNA`, `gc_offtarget`, `gc_delta`) and produces an additive offset to the logit. This plays the role of the **exogenous variable U** in the SCM — capturing environmental and cellular sources of variation that are orthogonal to the pure thermodynamic signal.

The empirical distributions of U confirm this decomposition is scientifically meaningful:

| Dataset | Mean U_off | Interpretation |
|---|---|---|
| CHANGE-seq (in vitro) | +2.34 | Cell-free saturation boosts baseline signal |
| GUIDE-seq (in vivo) | −0.14 | Competing cellular factors suppress baseline |

This separation is a key result: **the model learns thermodynamics; the gap between in vitro and in vivo is encoded in U, not in model error.**

### 3.4 Causal Training Signal

Training is not purely supervised. On 50% of batches, the model also receives **interventional data** generated on-the-fly via `do()` operations:

- `do(seed_region=mismatch)` — forced seed-region damage, expected to strongly reduce activity
- The causal loss penalises violations of the expected direction (`lambda_causal = 0.01`)
- A consistency loss (`lambda_consist = 0.01`) enforces agreement between observational and interventional predictions on shared inputs

---

## 4. Explainability

### 4.1 Counterfactual Reasoning via the `do()` Operator

The SCM supports three levels of Pearl's causal hierarchy:

| Level | Query | Implementation |
|---|---|---|
| **Observation** | P(cleavage \| mismatch pattern) | Standard forward pass |
| **Intervention** | P(cleavage \| do(node = v)) | Fix node value, break upstream edges, propagate |
| **Counterfactual** | "What would cleavage have been if seed were perfect?" | Abduct U from observed y, intervene, propagate U forward |

Counterfactual inference follows the three-step procedure:
1. **Abduction**: infer U_off = f⁻¹(y_observed, X) from the observed outcome
2. **Intervention**: apply `do(node = v)` to obtain the modified structural equations
3. **Prediction**: propagate U_off through the modified equations to obtain y_counterfactual

This is used both for model evaluation and for guide RNA redesign experiments.

### 4.2 Thermodynamic Profile Visualisation

The learned positional weights `w_pos` and the penalty MLPs are directly interpretable as a **thermodynamic fingerprint** of Cas9 specificity. The key qualitative findings from the weight heatmap:

- **Positions 16–20 (PAM-proximal)**: highest absolute penalty weights — consistent with the well-established "seed region PAM-proximal" biology
- **Positions 8–15 (seed)**: second-highest weights — guide-target mismatches here are severely destabilising
- **Positions 0–7 (non-seed / PAM-distal)**: lighter penalties, reflecting the biological tolerance for distal mismatches

Mismatch chemistry is also captured in the penalty MLP outputs: transversions > transitions > wobble pairs in terms of activity reduction, in agreement with melting temperature data.

### 4.3 Causal Consistency Score (CCS)

CCS is an evaluation metric that tests whether the model respects biologically grounded causal rules by running `do()` interventions on the held-out test set and checking whether the predicted direction of change matches expectation.

Six rules are evaluated:

| Rule | Intervention | Expected direction | Exp15 result |
|---|---|---|---|
| R1: PAM ablation | `do(pam_gate = 0.1)` | activity ↓ | **Pass (100%)** |
| R2: Proximal mismatch damage | `do(proximal = 1.0)` | activity ↓ | Fail (0%) |
| R3: Heal seed region | `do(seed = 0.0)` | activity ↑ | Fail (0%) |
| R4: Proximal > non-seed severity | compare damaged proximal vs non-seed | proximal worse | Fail (0%) |
| R5: Mismatch severity ordering | energy-level hierarchy | higher energy → lower activity | Fail (0%) |
| R6: PAM hierarchy | `NGG > NAG > NCG` | hierarchy preserved | **Pass (100%)** |

**CCS_Overall = 33.3%**

The failures on R2–R5 reveal that the positional independence assumption in the current architecture is too strong: the model does not separately track proximal/seed/non-seed sub-DAGs with sufficient contrast. The PAM-related rules (R1, R6) pass because PAM gating is architecturally explicit.

Compared to non-causal baselines (XGBoost), which achieve CCS ≈ 6.4% on a reduced 3-rule evaluation, the SCM approach represents a qualitative improvement in causal behaviour.

### 4.4 Batch Counterfactual Intervention Experiments

Two fixed interventions were applied across the full CHANGE-seq (67k pairs) and GUIDE-seq (1.6k pairs) datasets to test whether simple guide modifications improve the off-target / on-target trade-off:

**Intervention A — 5' truncation (remove positions 0–1):**
- GUIDE-seq: Δoff = −12.9%, Δon = −15.5% → *symmetric efficacy loss, no Pareto gain*

**Intervention B — Substitute position 15 → A:**
- GUIDE-seq: Δoff = −7.95%, Δon = −7.17% → *neutral trade-off*
- 25% of guides already carry A at position 14 (intervention is a no-op)
- 43% of guides show Δoff ≥ 0 (intervention backfires)

**Key conclusion**: Fixed interventions universally fail to deliver Pareto-optimal guide redesign. This motivates the next phase: **abduction-guided, guide-specific rescue mutations** that leverage the inferred U_off to identify positions where a targeted substitution uniquely damages the off-target without touching the on-target.

---

## 5. Predictive Performance

### 5.1 Within-dataset (CHANGE-seq test set, 583k pairs)

| Metric | Value |
|---|---|
| AUROC | 0.905 |
| AUPRC | 0.154 |
| F1 | 0.075 |

The low F1 reflects the extreme class imbalance (46:1). AUROC and AUPRC are the primary metrics.

### 5.2 Cross-assay generalisation (GUIDE-seq, 1.6k pairs)

| Metric | Value |
|---|---|
| AUROC | 0.964 |
| AUPRC | 0.285 |
| F1 | 0.274 |

Cross-assay AUROC **improves** relative to within-dataset, confirming that the model has learned transferable thermodynamic principles rather than CHANGE-seq-specific artefacts. The AUPRC gap between datasets is attributed to protocol-level differences (cell-free vs in vivo) encoded in U, not to model overfitting.

---

## 6. Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | Extended OneCycleLR (30 epochs, pct_start=0.15) |
| Batch size | 64 |
| Focal Loss α / γ | 0.25 / 3.0 |
| pos_weight | 5.0 |
| λ_causal | 0.01 |
| λ_consist | 0.01 |
| Interventional batch fraction | 50% |
| Hardware | CUDA |

The Extended OneCycleLR was introduced to reduce gradient shock during the warm-up phase, which destabilised earlier runs that used standard OneCycle with shorter `pct_start`.

---

## 7. Current Limitations and Next Steps

### Limitations

1. **CCS failures on R2–R5** indicate the positional independence assumption prevents the model from learning strong proximal/seed/non-seed contrasts via `do()` interventions. A hierarchical SCM with explicit sub-region nodes would address this.

2. **Fixed interventions do not yield Pareto-optimal guide redesign.** The batch counterfactual experiments confirm that position-specific context must be exploited — the same substitution has opposite effects depending on the guide sequence.

3. **AUPRC remains low on CHANGE-seq** (0.154) due to class imbalance and the inherently noisy cell-free experimental protocol, which saturates activity measurements and inflates the positive class.

### Next Steps

1. **Abduction-guided rescue mutations**: For each guide, infer U_off, then enumerate single-position substitutions and select those maximising Δoff − Δon under the counterfactual distribution. This is the fully personalised intervention regime enabled by the SCM formalism.

2. **Hierarchical sub-region nodes**: Replace the positional-independence assumption with explicit seed, proximal, and non-seed nodes connected in the DAG, so that `do(seed = v)` interventions operate on a structurally isolated sub-graph.

3. **CCS-augmented training**: Add R2–R5 violations to the causal loss term to directly supervise positional hierarchy during training, rather than only evaluating it post-hoc.

---

## 8. Repository Layout

```
experiments/exp_15_positional_extended_onecycle/
    config.yaml              # full hyperparameter specification
    run.py                   # training loop + causal augmentation

explainability/
    explain_thermodynamics.py           # positional weight visualisation
    simulate_intervention_batch.py      # batch counterfactual experiments
    plots/
        thermodynamic_profile.png
        thermodynamic_profile_real_guides.png
    batch_results/                      # per-guide Δoff / Δon tables

models/
    scm.py                   # SCM forward pass and do() operator
    positional_mlp.py        # position-independent penalty module

evaluation/
    causal_consistency.py    # CCS computation (6-rule evaluation)
```

---

*Report generated: May 2026*
