import torch
from torch import nn

# TODO: We make the following assumptions:
# 1. All molecules in the batch have the same number and types of atoms.
# 2. All atoms of the same molecule have contiguous indices in the batch.
# 3. The order of the atoms in each molecule determines their type.
# Can the classifier guarantee these assumptions?

# TODO: Should we allow for constraints?
# For example, when we use this layer for coulomb charges,
# the charges should sum to some integer for each molecule.


class AtomCenteredStatic(nn.Module):
    def __init__(
        self,
        initial_type_params,
        type_index,
    ):
        super().__init__()
        initial_type_params = torch.tensor(initial_type_params, dtype=torch.float32)
        self.type_params = nn.Parameter(initial_type_params)
        self.type_index = self.register_buffer(
            "type_index", torch.tensor(type_index, dtype=torch.int64)
        )
        self.num_atoms_per_mol = len(type_index)

    def forward(self, batch):
        sites_pos = batch.pos
        num_molecules = batch.pos.shape[0] // self.num_atoms_per_mol
        sites_params = self.type_params[self.type_index].repeat(num_molecules, 1)
        sites_batch = (
            batch.batch
            if batch.batch is not None
            else torch.zeros(
                sites_pos.shape[0], dtype=torch.int64, device=sites_pos.device
            )
        )
        sites_mol = torch.arange(
            num_molecules, device=sites_pos.device
        ).repeat_interleave(self.num_atoms_per_mol)
        return sites_pos, sites_params, sites_batch, sites_mol
