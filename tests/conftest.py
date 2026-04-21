import pandas as pd
import pytest

from dag.nodes import CRISPRPairFeatures


@pytest.fixture
def sample_pairs() -> list[CRISPRPairFeatures]:
    return [
        CRISPRPairFeatures(
            guide_seq="GATTACAGATTACAGATTACA",
            target_seq="GATTACAGACTACAGATTACA",
            pam="AGG",
            assay="changeseq",
            enzyme="SpCas9",
        ),
        CRISPRPairFeatures(
            guide_seq="CCCCAAAATTTTGGGGCCCCA",
            target_seq="CCCCAAAATTTTGGGGCCCCT",
            pam="TGG",
            assay="guideseq",
            enzyme="SpCas9",
        ),
    ]


@pytest.fixture
def mock_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pam_score": [1.0, 0.8],
            "mismatch_count": [1, 2],
            "seed_mismatch_count": [1, 1],
            "mean_energy_penalty": [0.2, 0.4],
            "total_energy_penalty": [4.0, 8.0],
        }
    )
