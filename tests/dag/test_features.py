from dag.features import build_feature_dataframe


def test_feature_dataframe_has_expected_columns(sample_pairs) -> None:
    df = build_feature_dataframe(sample_pairs)
    expected = {
        "pam_score",
        "mismatch_count",
        "seed_mismatch_count",
        "gc_guide",
        "gc_target",
        "mean_energy_penalty",
        "total_energy_penalty",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 2
