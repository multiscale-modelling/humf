from torch.nn import Module
from torch_geometric.utils import scatter

from humf.layers.energy.utils import get_pairs
from humf.layers.pair_energy.coulomb import Coulomb
from humf.layers.pair_energy.lennard_jones import LennardJones


class LennardJonesCoulomb(Module):
    def __init__(self, lennard_jones_sites: Module, coulomb_sites: Module):
        super().__init__()
        self.lennard_jones_sites = lennard_jones_sites
        self.coulomb_sites = coulomb_sites
        self.lennard_jones = LennardJones()
        self.coulomb = Coulomb()

    def forward(self, batch):
        distances, charges, frames = get_pairs(*self.coulomb_sites(batch))
        coulomb_contribs = self.coulomb(distances, charges[:, 0, 0], charges[:, 1, 0])
        coulomb_energy = scatter(
            coulomb_contribs,
            frames,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )

        distances, params, frames = get_pairs(*self.lennard_jones_sites(batch))
        lj_contribs = self.lennard_jones(distances, params[:, 0], params[:, 1])
        lj_energy = scatter(
            lj_contribs,
            frames,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )

        return coulomb_energy + lj_energy
