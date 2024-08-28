import torch
from ase import Atoms, io
from torch_geometric.data import Data, InMemoryDataset


class ASEDataset(InMemoryDataset):
    """Read dataset from an XYZ file written by ASE."""

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.xyz"]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        atoms_list = io.read(self.raw_paths[0], ":")
        assert type(atoms_list) == list[Atoms]
        data_list = []
        for atoms in atoms_list:
            positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
            types = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
            info = {
                k: torch.tensor(v, dtype=torch.float32) for k, v in atoms.info.items()
            }
            data = Data(
                pos=positions,
                types=types,
                **info,
            )
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])
