import torch
from torch import nn

# TODO: We make the following assumptions:
# 1. All molecules in the batch have the same number and types of atoms.
# 2. All atoms of the same molecule have contiguous indices in the batch.
# 3. The order of the atoms in each molecule determines their type.
# Can the classifier guarantee these assumptions?

# TODO: Should we allow for constraints?
# For example, when we use this layer for OHH molecules,
# `mol_params[1]` should be identical to `mol_params[2]`.


class AtomCenteredStatic(nn.Module):
    def __init__(
        self,
        num_atoms_per_mol: int,
        num_params_per_atom: int,
        initial_params,
    ):
        super().__init__()
        initial_params = torch.tensor(initial_params, dtype=torch.float32)
        assert initial_params.shape == (num_atoms_per_mol, num_params_per_atom)
        self.mol_params = nn.Parameter(initial_params)
        self.num_atoms_per_mol = num_atoms_per_mol

    def forward(self, batch):
        sites_pos = batch.pos
        num_molecules = batch.pos.shape[0] // self.num_atoms_per_mol
        sites_params = self.mol_params.repeat(num_molecules, 1)
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
