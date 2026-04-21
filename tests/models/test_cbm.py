import torch

from models.deep.cbm import ConceptBottleneckModel


def test_cbm_forward_shapes() -> None:
    model = ConceptBottleneckModel(input_dim=6, concept_dim=4, hidden_dim=12)
    x = torch.randn(5, 6)
    logits, concepts = model(x)

    assert logits.shape == (5,)
    assert concepts.shape == (5, 4)
