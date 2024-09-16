from typing import Iterable, Union

import torch
from torch import nn


class InverseDistancePolynomial(nn.Module):
    def __init__(self, orders: Union[int, Iterable[int]]):
        super().__init__()
        if isinstance(orders, int):
            orders = range(1, orders + 1)
        self.register_buffer("orders", torch.tensor(orders))

    def forward(
        self,
        distances,  # [num_pairs]
        parameters,  # [num_pairs, 2, num_params_per_site]
    ):
        inverse_distances = 1 / distances  # [num_pairs]
        inverse_distance_powers = (
            inverse_distances.unsqueeze(1) ** self.orders
        )  # [num_pairs, num_params_per_site]

        mixed_parameters = parameters[:, 0] * parameters[:, 1]

        energies = torch.sum(
            inverse_distance_powers * mixed_parameters, dim=1
        )  # [num_pairs]
        return energies
