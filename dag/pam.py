from typing import Dict

IUPAC_BASES: Dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "N": {"A", "C", "G", "T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
}

PAM_COMPATIBILITY: Dict[str, Dict[str, float]] = {
    "SpCas9": {"NGG": 1.00, "NAG": 0.65},
    "SaCas9": {"NNGRRT": 1.00, "NNGRRN": 0.75},
    "Cas12a": {"TTTV": 1.00, "TTV": 0.70},
}


def _motif_match_fraction(motif: str, pam: str) -> float:
    if len(motif) != len(pam):
        return 0.0

    matches = 0
    for motif_base, pam_base in zip(motif, pam):
        motif_set = IUPAC_BASES.get(motif_base, {motif_base})
        if pam_base in motif_set:
            matches += 1
    return matches / len(motif)


def pam_score(pam: str, enzyme: str = "SpCas9") -> float:
    """Return a soft compatibility score in [0, 1] for a PAM sequence."""
    rules = PAM_COMPATIBILITY.get(enzyme, {})
    if not rules:
        return 0.0

    pam = pam.upper()
    best = 0.0
    for motif, weight in rules.items():
        best = max(best, _motif_match_fraction(motif, pam) * weight)

    return round(max(0.0, min(1.0, best)), 4)
