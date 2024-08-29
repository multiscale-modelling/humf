from torch import nn


class Coulomb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances, charges_1, charges_2):
        return charges_1 * charges_2 / distances
