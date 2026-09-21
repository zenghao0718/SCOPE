"""Full diagonal Gaussian mixture negative log likelihood."""
import math
import torch


def patch_nll(r_z, log_pi, mu, sigma):
    gaussian_log_density = -0.5 * (((r_z[:, None, :] - mu) / sigma).square()
                                   + 2 * sigma.log() + math.log(2 * math.pi)).sum(dim=-1)
    return -torch.logsumexp(log_pi + gaussian_log_density, dim=-1)
