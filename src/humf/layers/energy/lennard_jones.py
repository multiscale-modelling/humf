import torch
from torch import nn


class LennardJones(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances, params_1, params_2):
        epsilon = torch.sqrt(torch.abs(params_1[:, 0] * params_2[:, 0]))
        sigma = torch.sqrt(torch.abs(params_1[:, 1] * params_2[:, 1]))
        r = sigma / distances
        r6 = r**6
        r12 = r6**2
        return 4 * epsilon * (r12 - r6)
