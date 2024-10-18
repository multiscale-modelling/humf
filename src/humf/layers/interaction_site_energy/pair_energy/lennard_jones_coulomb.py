from torch.nn import Module

from humf.layers.interaction_site_energy.pair_energy.coulomb import Coulomb
from humf.layers.interaction_site_energy.pair_energy.lennard_jones import LennardJones


class LennardJonesCoulomb(Module):
    def __init__(self) -> None:
        super().__init__()
        self.lennard_jones = LennardJones()
        self.coulomb = Coulomb()

    def forward(
        self,
        distances,  # [num_pairs]
        parameters,  # [2, num_pairs, 3] (epsilon, sigma, charge)
    ):
        lj_contribs = self.lennard_jones(
            distances, parameters[0, :, :2], parameters[1, :, :2]
        )
        coulomb_contribs = self.coulomb(
            distances, parameters[0, :, 2], parameters[1, :, 2]
        )
        return lj_contribs + coulomb_contribs
