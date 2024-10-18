import torch
from torch import Tensor
from torch.nn import Module


class AtomCenteredStaticSites(Module):
    def __init__(
        self,
        initial_type_parameters,
    ):
        super().__init__()
        initial_type_parameters = torch.tensor(
            initial_type_parameters, dtype=torch.float32
        )
        self.type_parameters = torch.nn.Parameter(initial_type_parameters)

    @classmethod
    def with_random_initial_parameters(
        cls,
        num_unique_molecule_atom_types: int,
        num_parameters_per_site: int,
    ):
        initial_parameters = torch.randn(
            num_unique_molecule_atom_types, num_parameters_per_site
        )
        return cls(initial_parameters)

    def forward(
        self,
        batch,
        interaction_types: Tensor,
        molecule_types: Tensor,
        molecule_atom_types: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del molecule_types
        sites_positions = batch.positions
        sites_batch = (
            batch.batch
            if batch.batch is not None
            else torch.zeros(
                sites_positions.shape[0],
                dtype=torch.int64,
                device=sites_positions.device,
            )
        )
        return (
            sites_positions,
            self.type_parameters[molecule_atom_types],
            batch.edge_index[:, interaction_types == 1],
            sites_batch,
        )
