import math
import torch
from losses.mdn_nll import patch_nll


def test_gaussian_constant_and_negative_nll():
    x = torch.zeros(1, 8)
    pi = torch.zeros(1, 1)
    mu = torch.zeros(1, 1, 8)
    sigma = torch.ones(1, 1, 8)
    assert torch.allclose(patch_nll(x, pi, mu, sigma), torch.tensor([4*math.log(2*math.pi)]))
    assert patch_nll(x, pi, mu, sigma * 0.01).item() < 0
    far = patch_nll(torch.full((1, 8), 1000.), pi, mu, sigma)
    assert torch.isfinite(far).all()
