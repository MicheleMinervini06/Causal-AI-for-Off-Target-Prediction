from dag.mismatch import mismatch_energy_penalty, mismatch_type


def test_mismatch_type_transition() -> None:
    assert mismatch_type("A", "G") == "transition"


def test_mismatch_type_match() -> None:
    assert mismatch_type("C", "C") == "match"


def test_seed_penalty_is_higher() -> None:
    seed = mismatch_energy_penalty("transversion", position=2)
    non_seed = mismatch_energy_penalty("transversion", position=15)
    assert seed > non_seed
