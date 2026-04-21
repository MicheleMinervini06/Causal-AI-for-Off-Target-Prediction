TRANSITIONS = {
    ("A", "G"),
    ("G", "A"),
    ("C", "T"),
    ("T", "C"),
}

WOBBLE = {
    ("G", "T"),
    ("T", "G"),
}

BASE_MISMATCH_PENALTY = {
    "match": 0.0,
    "transition": 0.5,
    "transversion": 1.0,
    "wobble": 0.3,
}


def mismatch_type(guide_nt: str, target_nt: str) -> str:
    pair = (guide_nt.upper(), target_nt.upper())
    if pair[0] == pair[1]:
        return "match"
    if pair in WOBBLE:
        return "wobble"
    if pair in TRANSITIONS:
        return "transition"
    return "transversion"


def mismatch_energy_penalty(kind: str, position: int) -> float:
    """Position-aware mismatch penalty with stronger seed region impact."""
    base = BASE_MISMATCH_PENALTY[kind]
    seed_multiplier = 1.4 if position < 10 else 1.0
    return base * seed_multiplier
