"""Conditional three-component diagonal Gaussian MDN."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MDN(nn.Module):
    def __init__(self, c_dim=6, r_dim=8, num_components=3, sigma_floor=0.05, hidden_dims=(64, 64)):
        super().__init__()
        if num_components < 1 or sigma_floor <= 0 or len(hidden_dims) != 2:
            raise ValueError("invalid MDN configuration")
        self.r_dim, self.num_components, self.sigma_floor = r_dim, num_components, sigma_floor
        self.net = nn.Sequential(nn.Linear(c_dim, hidden_dims[0]), nn.ReLU(),
                                 nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
                                 nn.Linear(hidden_dims[1], num_components * (1 + 2 * r_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.net[0], self.net[2]):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        output = self.net[4]
        nn.init.normal_(output.weight, 0, 1e-3)
        k, d = self.num_components, self.r_dim
        with torch.no_grad():
            output.bias[:k].zero_()
            output.bias[k:k+k*d].normal_(0, 0.1)
            output.bias[k+k*d:].fill_(math.log(math.expm1(0.95)))

    def forward(self, c):
        out = self.net(c)
        k, d = self.num_components, self.r_dim
        log_pi = F.log_softmax(out[:, :k], dim=-1)
        mu = out[:, k:k+k*d].reshape(-1, k, d)
        sigma = self.sigma_floor + F.softplus(out[:, k+k*d:].reshape(-1, k, d), beta=1, threshold=20)
        return log_pi, mu, sigma
