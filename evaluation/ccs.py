import numpy as np
from typing import Callable, Literal

def _to_proba_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError(f"predict_fn returned 2D array with invalid shape {arr.shape}")
        arr = arr[:, 1]
    elif arr.ndim != 1:
        raise ValueError(f"predict_fn must return a 1D or 2D array, got shape {arr.shape}")
    return arr


def _predict_checked(
    predict_fn: Callable[[list[str], list[str]], np.ndarray],
    guides: list[str],
    off_targets: list[str],
) -> np.ndarray:
    probs = _to_proba_1d(predict_fn(guides, off_targets))
    if probs.shape[0] != len(guides):
        raise ValueError(
            "predict_fn returned a different number of predictions "
            f"({probs.shape[0]}) than input pairs ({len(guides)})"
        )
    if not np.isfinite(probs).all():
        raise ValueError("predict_fn returned non-finite probabilities")
    return probs


def mutate_base(base: str, mutation_type: Literal["transition", "transversion", "wobble"] = "transversion") -> str:
    """Ritorna una base mutata secondo il tipo richiesto."""
    base = base.upper()
    if mutation_type == "wobble":
        # Preferisci mutazioni wobble G<->T; fallback su transizioni per A/C.
        wobble = {"G": "T", "T": "G", "A": "G", "C": "T"}
        return wobble.get(base, "N")
    
    transitions = {"A": "G", "G": "A", "C": "T", "T": "C"}
    transversions = {"A": "C", "G": "T", "C": "A", "T": "G"} # Una trasversione rappresentativa
    
    if mutation_type == "transition":
        return transitions.get(base, "N")
    return transversions.get(base, "N")

def calculate_ccs(
    unique_guides: list[str], 
    predict_fn: Callable[[list[str], list[str]], np.ndarray],
    mode: Literal["3_rules", "6_rules"] = "3_rules"
) -> dict:
    """
    Calcola il Causal Consistency Score (CCS) generando interventi sintetici.
    
    Args:
        unique_guides: Lista di sequenze spacer da 20nt (es. dal test set).
        predict_fn: Funzione che accetta (sgRNA_seqs, off_seqs) e ritorna array di probabilità.
        mode: "3_rules" (Base) o "6_rules" (Estesa).
        
    Returns:
        Dizionario con le percentuali di superamento per singola regola e il CCS totale.
    """
    if mode not in {"3_rules", "6_rules"}:
        raise ValueError("mode must be '3_rules' or '6_rules'")

    if len(unique_guides) == 0:
        raise ValueError("unique_guides cannot be empty")

    if any(len(g) < 20 for g in unique_guides):
        raise ValueError("Each guide sequence must be at least 20 nt long")

    results = {}
    
    # --- Generazione Baseline (Match Perfetto) ---
    guides = [g[:20].upper() for g in unique_guides]
    targets_perfect = [g + "AGG" for g in guides] # Assumiamo NGG canonico come baseline
    
    # Calcolo probabilità baseline
    p_baseline = _predict_checked(predict_fn, guides, targets_perfect)
    
    # ==========================================
    # REGOLA 1: PAM NGG -> NAA (P scende)
    # ==========================================
    targets_naa = [g + "AAA" for g in guides]
    p_naa = _predict_checked(predict_fn, guides, targets_naa)
    r1_pass = (p_naa < p_baseline).astype(int)
    results["R1_PAM_Ablation"] = np.mean(r1_pass)

    # ==========================================
    # REGOLA 2: Mismatch in Posizione 1 (PAM-proximal, indice 19) (P scende)
    # ==========================================
    targets_mm1 = [g[:19] + mutate_base(g[19]) + "AGG" for g in guides]
    p_mm1 = _predict_checked(predict_fn, guides, targets_mm1)
    r2_pass = (p_mm1 < p_baseline).astype(int)
    results["R2_Pos1_Mismatch"] = np.mean(r2_pass)

    # ==========================================
    # REGOLA 3: Heal Seed (P_sporco < P_guarito)
    # Creiamo un target con 3 mismatch nel seed (pos 17, 18, 19), poi verifichiamo che 
    # la probabilità della baseline (guarita) sia maggiore.
    # ==========================================
    targets_dirty_seed = [
        g[:17] + mutate_base(g[17]) + mutate_base(g[18]) + mutate_base(g[19]) + "AGG" 
        for g in guides
    ]
    p_dirty_seed = _predict_checked(predict_fn, guides, targets_dirty_seed)
    r3_pass = (p_dirty_seed < p_baseline).astype(int)
    results["R3_Heal_Seed"] = np.mean(r3_pass)

    # --- CCS finale: media dei punteggi per regola ---
    results["CCS_Overall"] = float(np.mean([results["R1_PAM_Ablation"], results["R2_Pos1_Mismatch"], results["R3_Heal_Seed"]]))

    if mode == "6_rules":
        # ==========================================
        # REGOLA 4: Seed (pos 19) vs Non-Seed (pos 0) Mismatch
        # P_seed_mm < P_nonseed_mm
        # ==========================================
        targets_mm_nonseed = [mutate_base(g[0]) + g[1:] + "AGG" for g in guides]
        p_mm_nonseed = _predict_checked(predict_fn, guides, targets_mm_nonseed)
        r4_pass = (p_mm1 < p_mm_nonseed).astype(int) # p_mm1 calcolato nella Regola 2
        results["R4_Seed_vs_NonSeed"] = np.mean(r4_pass)

        # ==========================================
        # REGOLA 5: Wobble vs Trasversione (pos 15)
        # P_trasversione < P_wobble
        # ==========================================
        targets_wobble = [g[:15] + mutate_base(g[15], "wobble") + g[16:] + "AGG" for g in guides]
        targets_transv = [g[:15] + mutate_base(g[15], "transversion") + g[16:] + "AGG" for g in guides]
        p_wobble = _predict_checked(predict_fn, guides, targets_wobble)
        p_transv = _predict_checked(predict_fn, guides, targets_transv)
        r5_pass = (p_transv < p_wobble).astype(int)
        results["R5_Wobble_vs_Transv"] = np.mean(r5_pass)

        # ==========================================
        # REGOLA 6: PAM Affinity (NGG > NAG > NCG)
        # ==========================================
        targets_nag = [g + "AAG" for g in guides]
        targets_ncg = [g + "ACG" for g in guides]
        p_nag = _predict_checked(predict_fn, guides, targets_nag)
        p_ncg = _predict_checked(predict_fn, guides, targets_ncg)
        r6_pass = ((p_baseline > p_nag) & (p_nag > p_ncg)).astype(int)
        results["R6_PAM_Hierarchy"] = np.mean(r6_pass)

        # --- CCS finale: media di tutte le regole disponibili ---
        results["CCS_Overall"] = float(np.mean([
            results["R1_PAM_Ablation"],
            results["R2_Pos1_Mismatch"],
            results["R3_Heal_Seed"],
            results["R4_Seed_vs_NonSeed"],
            results["R5_Wobble_vs_Transv"],
            results["R6_PAM_Hierarchy"],
        ]))

    return results

def calculate_ccs_neural(
    model,
    unique_guides: list[str],
    mode: Literal["3_rules", "6_rules"] = "3_rules",
) -> dict:
    """
    CCS per Neural SCM usando do() nativo sui nodi del DAG.

    Differenza rispetto a calculate_ccs():
        - Gli interventi operano direttamente sui nodi del DAG
          dopo l'encoding, non sulle sequenze grezze.
        - Elimina il rumore introdotto dalla re-codifica
          di sequenze mutate sinteticamente.

    Regole:
        R1: do(pam_gate=0.1)    → activity scende     [nodo A]
        R2: do(proximal=1.0)    → activity scende     [nodo B]
        R3: do(seed=0.0)        → activity sale       [nodo C]
        R4: do(proximal=1.0) < do(nonseed=1.0)        [nodo B vs D]
        R5: do(seed=0.75) < do(seed=0.4)              [monotonicità nodo C]
        R6: do(pam_gate=1.0) > do(pam_gate=0.2) > do(pam_gate=0.1)  [gerarchia nodo A]
    """

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for calculate_ccs_neural. Please install it to use this function.")
    
    if not unique_guides:
        raise ValueError("unique_guides cannot be empty")

    guides  = [g[:20].upper() for g in unique_guides]
    targets = [g + "AGG" for g in guides]

    def _do(intervention: dict) -> np.ndarray:
        with torch.no_grad():
            out = model.do(guides, targets, intervention)
        return out["activity_probability"].squeeze(-1).cpu().numpy()

    # Baseline osservazionale
    with torch.no_grad():
        out_base = model.forward(guides, targets)
    p_base = out_base["activity_probability"].squeeze(-1).cpu().numpy()

    # ── R1: PAM ablation ─────────────────────────────────────────────
    r1_pass = (_do({"pam_gate": 0.1}) < p_base).astype(int)

    # ── R2: massima penalità PAM-proximal ────────────────────────────
    r2_pass = (_do({"proximal": 1.0}) < p_base).astype(int)

    # ── R3: seed perfetta → activity sale ────────────────────────────
    r3_pass = (_do({"seed": 0.0}) > p_base).astype(int)

    results = {
        "R1_PAM_Ablation":  float(np.mean(r1_pass)),
        "R2_Pos1_Mismatch": float(np.mean(r2_pass)),
        "R3_Heal_Seed":     float(np.mean(r3_pass)),
        "CCS_Overall":      float(np.mean([
            np.mean(r1_pass),
            np.mean(r2_pass),
            np.mean(r3_pass),
        ])),
        "method":           "neural_do",
    }

    if mode == "6_rules":
        # ── R4: seed penalizza più di non-seed ───────────────────────
        p_prox_stress   = _do({"proximal": 1.0})
        p_nonseed_stress = _do({"non_seed": 1.0})
        r4_pass = (p_prox_stress < p_nonseed_stress).astype(int)

        # ── R5: monotonicità nodo C (alta energia > bassa energia) ───
        p_seed_high = _do({"seed": 0.75})
        p_seed_low  = _do({"seed": 0.40})
        r5_pass = (p_seed_high < p_seed_low).astype(int)

        # ── R6: gerarchia PAM NGG > NAG > NCG ────────────────────────
        p_ngg = _do({"pam_gate": 1.00})
        p_nag = _do({"pam_gate": 0.20})
        p_ncg = _do({"pam_gate": 0.10})
        r6_pass = ((p_ngg > p_nag) & (p_nag > p_ncg)).astype(int)

        results.update({
            "R4_Seed_vs_NonSeed":  float(np.mean(r4_pass)),
            "R5_Wobble_vs_Transv": float(np.mean(r5_pass)),
            "R6_PAM_Hierarchy":    float(np.mean(r6_pass)),
            "CCS_Overall":         float(np.mean([
                np.mean(r1_pass),
                np.mean(r2_pass),
                np.mean(r3_pass),
                np.mean(r4_pass),
                np.mean(r5_pass),
                np.mean(r6_pass),
            ])),
        })

    return results


def calculate_ccs_neural_v2(
    model,
    unique_guides: list[str],
    h_penalty: float = 5.0,
    pam_levels: tuple[float, float, float] = (-1.0, 0.0, 2.0),
    pam_ablation: float = -3.0,
) -> dict:
    """
    CCS v2 — designed for the positional_mlp + additive PAM architecture (Exp20+).

    The original calculate_ccs_neural() probes regional interventions
    (do(seed=...), do(proximal=...), do(non_seed=...)) which positional_mlp
    ignores by construction — the regional aggregations are derived sums of
    20 positional nodes, not separate causal nodes.

    This v2 probes positional-level interventions do(pos_i=v) and PAM
    interventions do(pam_gate=v), which the additive architecture supports
    natively. Five rules, each with a clear biological interpretation:

        R1: PAM ablation                     do(pam_gate=-3) < baseline
        R2: PAM hierarchy (monotone)         do(pam=-1) < do(pam=0) < do(pam=2)
        R3: Seed > Distal sensitivity        do(pos_18=h) < do(pos_2=h)
        R4: Position gradient monotonic      drop_2 < drop_8 < drop_14 < drop_18
        R5: Heal mismatch via intervention   on a synthetically-stressed target,
                                             do(pos_18=0) > no-intervention
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for calculate_ccs_neural_v2.")

    if not unique_guides:
        raise ValueError("unique_guides cannot be empty")
    if any(len(g) < 20 for g in unique_guides):
        raise ValueError("Each guide must be at least 20 nt long")

    guides = [g[:20].upper() for g in unique_guides]
    perfect_targets = [g + "AGG" for g in guides]

    def _forward(targets):
        with torch.no_grad():
            out = model.forward(guides, targets)
        return out["activity_probability"].squeeze(-1).cpu().numpy()

    def _do(targets, intervention):
        with torch.no_grad():
            out = model.do(guides, targets, intervention)
        return out["activity_probability"].squeeze(-1).cpu().numpy()

    # Baseline on perfect-match targets
    p_base = _forward(perfect_targets)

    # ── R1: PAM ablation ────────────────────────────────────────────────
    p_pam_ablated = _do(perfect_targets, {"pam_gate": pam_ablation})
    r1_pass = (p_pam_ablated < p_base).astype(int)

    # ── R2: PAM hierarchy (3-level monotonicity) ────────────────────────
    p_pam_a = _do(perfect_targets, {"pam_gate": pam_levels[0]})
    p_pam_b = _do(perfect_targets, {"pam_gate": pam_levels[1]})
    p_pam_c = _do(perfect_targets, {"pam_gate": pam_levels[2]})
    r2_pass = ((p_pam_a < p_pam_b) & (p_pam_b < p_pam_c)).astype(int)

    # ── R3: Seed (pos 18) > PAM-distal (pos 2) sensitivity ─────────────
    p_pos18 = _do(perfect_targets, {"pos_18": h_penalty})
    p_pos2  = _do(perfect_targets, {"pos_2":  h_penalty})
    r3_pass = (p_pos18 < p_pos2).astype(int)

    # ── R4: Position gradient monotonic across {2, 8, 14, 18} ──────────
    positions = [2, 8, 14, 18]
    drops = []
    for pos in positions:
        p_pos = _do(perfect_targets, {f"pos_{pos}": h_penalty})
        drops.append(p_base - p_pos)
    monotonic = (drops[0] < drops[1]) & (drops[1] < drops[2]) & (drops[2] < drops[3])
    r4_pass = monotonic.astype(int)

    # ── R5: Heal mismatch via do(pos_18=0) on a stressed target ────────
    stressed_targets = [
        g[:18] + mutate_base(g[18]) + g[19] + "AGG" for g in guides
    ]
    p_dirty  = _forward(stressed_targets)
    p_healed = _do(stressed_targets, {"pos_18": 0.0})
    r5_pass = (p_healed > p_dirty).astype(int)

    return {
        "R1_PAM_Ablation":      float(np.mean(r1_pass)),
        "R2_PAM_Hierarchy":     float(np.mean(r2_pass)),
        "R3_Seed_vs_Distal":    float(np.mean(r3_pass)),
        "R4_Position_Gradient": float(np.mean(r4_pass)),
        "R5_Heal_Mismatch":     float(np.mean(r5_pass)),
        "CCS_Overall": float(np.mean([
            np.mean(r1_pass),
            np.mean(r2_pass),
            np.mean(r3_pass),
            np.mean(r4_pass),
            np.mean(r5_pass),
        ])),
        "method": "neural_do_v2_positional_mlp_additive_pam",
        "config": {
            "h_penalty":    h_penalty,
            "pam_levels":   list(pam_levels),
            "pam_ablation": pam_ablation,
            "n_guides":     len(guides),
        },
    }