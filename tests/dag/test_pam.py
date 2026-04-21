from dag.pam import pam_score


def test_spcas9_prefers_ngg() -> None:
    assert pam_score("AGG", enzyme="SpCas9") > pam_score("AAA", enzyme="SpCas9")


def test_unknown_enzyme_returns_zero() -> None:
    assert pam_score("AGG", enzyme="Unknown") == 0.0
