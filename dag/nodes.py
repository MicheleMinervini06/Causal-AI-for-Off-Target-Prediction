from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CRISPRPairFeatures:
    """Container for one guide-target pair before feature extraction."""

    guide_seq: str
    target_seq: str
    pam: str
    assay: str = "unknown"
    enzyme: str = "SpCas9"

    def __post_init__(self) -> None:
        object.__setattr__(self, "guide_seq", self.guide_seq.upper())
        object.__setattr__(self, "target_seq", self.target_seq.upper())
        object.__setattr__(self, "pam", self.pam.upper())

    @property
    def guide_length(self) -> int:
        return len(self.guide_seq)
