import numpy as np

from evaluation.ess import explanation_stability_score


def test_ess_identical_explanations_is_high() -> None:
    base = np.array([0.1, 0.2, 0.3, 0.4])
    score = explanation_stability_score([base, base.copy(), base.copy()])
    assert score > 0.99
