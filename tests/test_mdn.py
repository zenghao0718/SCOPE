import torch
from models.mdn import MDN


def test_shapes_and_count():
    model = MDN()
    assert sum(p.numel() for p in model.parameters()) == 7923
    pi, mu, sigma = model(torch.zeros(5, 6))
    assert pi.shape == (5, 3) and mu.shape == sigma.shape == (5, 3, 8)
    assert torch.all(sigma >= 0.05)
    assert MDN(num_components=4).net[-1].out_features == 68
