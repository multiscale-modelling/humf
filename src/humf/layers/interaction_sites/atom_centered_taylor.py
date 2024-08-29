import torch
from torch import nn


class AtomCenteredTaylor(nn.Module):
    def __init__(self, num_atoms_per_mol):
        super().__init__()
        self.num_atoms_per_mol = num_atoms_per_mol

    def forward(self, batch):
        sites_pos = batch.pos
        sites_batch = batch.batch
        num_molecules = batch.pos.shape[0] // self.num_atoms_per_mol
        sites_mol = torch.arange(num_molecules).repeat_interleave(
            self.num_atoms_per_mol
        )
        return sites_pos, sites_batch, sites_mol
