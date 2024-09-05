import torch
from torch.nn import Module
from torch_geometric.utils import scatter

from humf.layers.energy.coulomb import Coulomb
from humf.layers.energy.lennard_jones import LennardJones


class LennardJonesCoulomb(Module):
    def __init__(self, lennard_jones_sites: Module, coulomb_sites: Module):
        super().__init__()
        self.lennard_jones_sites = lennard_jones_sites
        self.coulomb_sites = coulomb_sites
        self.lennard_jones = LennardJones()
        self.coulomb = Coulomb()

    def forward(self, batch):
        distances, charges, frames = self.get_pairs(*self.coulomb_sites(batch))
        coulomb_contribs = self.coulomb(distances, charges[:, 0, 0], charges[:, 1, 0])
        coulomb_energy = scatter(
            coulomb_contribs,
            frames,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )

        distances, params, frames = self.get_pairs(*self.lennard_jones_sites(batch))
        lj_contribs = self.lennard_jones(distances, params[:, 0], params[:, 1])
        lj_energy = scatter(
            lj_contribs,
            frames,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )

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
