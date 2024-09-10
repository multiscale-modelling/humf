import torch
from ase import io
from torch_geometric.data import Data, InMemoryDataset

from humf.data.utils import has_nans


class ASEDataset(InMemoryDataset):
    """Read dataset from an XYZ file written by ASE."""

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=True,
        raise_on_nan=True,
    ):
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])
        # for data in self:
        #     if raise_on_nan and has_nans(data):
        #         raise ValueError("Data contains NaNs.")

    @property
    def raw_file_names(self):
        return ["dataset.xyz"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        atoms_list = io.read(self.raw_paths[0], ":")
        assert type(atoms_list) is list
        data_list = []
        for atoms in atoms_list:
            positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
            types = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
            energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
            info = {
                k: torch.tensor(v, dtype=torch.float32) for k, v in atoms.info.items()
            }
            data = Data(
                pos=positions,
                types=types,
                energy=energy,
                **info,
            )
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])
