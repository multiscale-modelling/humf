import numpy as np
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RadiusGraphPBC(BaseTransform):
    def __init__(self, cutoff, cell=None, pbc=None):
        self.cutoff = cutoff
        self.cell = cell
        self.pbc = pbc

    def forward(self, data: Data) -> Data:
        data = data.clone()
        assert data.pos is not None
        cell = self.cell if self.cell is not None else data.cell
        pbc = self.pbc if self.pbc is not None else data.pbc
        atoms = Atoms(positions=data.pos.numpy(), cell=cell, pbc=pbc)
        i, j, d, D = neighbour_list("ijdD", atoms, self.cutoff)
        data.edge_index = torch.tensor(np.array([i, j]), dtype=torch.int64)
        data.edge_distances = torch.tensor(d, dtype=torch.float32).unsqueeze(1)
        data.edge_vectors = torch.tensor(D, dtype=torch.float32)
        return data
