import torch
from torch import nn
from torch_geometric.utils import scatter

from humf.layers.energy.coulomb import Coulomb
from humf.layers.energy.lennard_jones import LennardJones
from humf.layers.interaction_sites.atom_centered_static import AtomCenteredStatic


class Tip3pLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.lennard_jones_sites = AtomCenteredStatic(
            num_atoms_per_mol=3, num_params_per_atom=2
        )
        self.coulomb_sites = AtomCenteredStatic(
            num_atoms_per_mol=3, num_params_per_atom=1
        )
        self.lennard_jones = LennardJones()
        self.coulomb = Coulomb()

    def forward(self, batch):
        distances, charges, frames = self.get_pairs(*self.coulomb_sites(batch))
        coulomb_contribs = self.coulomb(distances, charges[:, 0, 0], charges[:, 1, 0])
        coulomb_energy = scatter(
            coulomb_contribs, frames, dim_size=batch.batch_size
        ).unsqueeze(1)

        distances, params, frames = self.get_pairs(*self.lennard_jones_sites(batch))
        lj_contribs = self.lj(distances, params[:, 0], params[:, 1])
        lj_energy = scatter(lj_contribs, frames, dim_size=batch.batch_size).unsqueeze(1)

        return coulomb_energy + lj_energy

    def get_pairs(self, sites_pos, sites_params, sites_batch, sites_mol):
        same_frame = sites_batch.unsqueeze(0) == sites_batch.unsqueeze(1)
        different_mol = sites_mol.unsqueeze(0) != sites_mol.unsqueeze(1)
        interacting_pairs = torch.nonzero(
            same_frame & different_mol
        )  # [num_interactions, 2]
        pair_sites = sites_pos[interacting_pairs]  # [num_interactions, 2, 3]
        pair_distances = torch.norm(
            pair_sites[:, 0] - pair_sites[:, 1], dim=1
        )  # [num_interactions]
        pair_params = sites_params[
            interacting_pairs
        ]  # [num_interactions, 2, num_params_per_site]
        pair_frame = sites_batch[interacting_pairs[:, 0]]  # [num_interactions]
        return pair_distances, pair_params, pair_frame
