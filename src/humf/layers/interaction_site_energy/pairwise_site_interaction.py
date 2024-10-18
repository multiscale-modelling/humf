import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter


class PairwiseSiteInteraction(Module):
    def __init__(self, pair_energy: Module) -> None:
        super().__init__()
        self.pair_energy = pair_energy

    def forward(
        self,
        batch,
        interaction_site_positions: Tensor,  # [num_sites, 3]
        interaction_site_parameters: Tensor,  # [num_sites, 3] (epsilon, sigma, charge)
        interaction_site_edge_index: Tensor,  # [2, num_edges]
        interaction_site_batch: Tensor,  # [num_sites]
    ) -> Tensor:
        # TODO: PBC
        pair_positions = interaction_site_positions[
            interaction_site_edge_index
        ]  # [2, num_edges, 3]
        pair_distances = torch.norm(
            pair_positions[0] - pair_positions[1], dim=-1
        )  # [num_edges]
        pair_parameters = interaction_site_parameters[
            interaction_site_edge_index
        ]  # [2, num_edges, num_parameters]
        contribs = self.pair_energy(pair_distances, pair_parameters)  # [num_edges]
        energy = scatter(
            contribs,
            interaction_site_batch,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )  # [batch_size]
        return energy
