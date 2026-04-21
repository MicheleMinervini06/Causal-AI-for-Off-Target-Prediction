import numpy as np

from evaluation.bas import biological_alignment_score


def test_bas_increases_for_aligned_profiles() -> None:
    bio_prior = np.array([0.6, 0.2, 0.1, 0.1])
    explanations = np.array(
        [
            [0.7, 0.2, 0.05, 0.05],
            [0.65, 0.2, 0.1, 0.05],
        ]
    )
    assert biological_alignment_score(explanations, bio_prior) > 0.7
