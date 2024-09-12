from torch.nn import Module
from torch_geometric.utils import scatter

from humf.layers.energy.utils import get_pairs


class InteractingSites(Module):
    def __init__(self, sites: Module, pair_energy: Module):
        super().__init__()
        self.sites = sites
        self.pair_energy = pair_energy

    def forward(self, batch):
        distances, parameters, frames = get_pairs(*self.sites(batch))
        pair_energies = self.pair_energy(distances, parameters)
        energy = scatter(
            pair_energies,
            frames,
            dim_size=batch.batch_size if batch.batch is not None else None,
        )
        return energy
