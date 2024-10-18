from torch import Tensor
from torch.nn import Module


class ManualClassifier(Module):
    def __init__(self, molecule_types: Tensor, molecule_atom_types: Tensor):
        super().__init__()
        self.register_buffer("molecule_types", molecule_types)
        self.register_buffer("molecule_atom_types", molecule_atom_types)

    def forward(self, batch, interaction_types: Tensor):
        del interaction_types
        assert self.molecule_types.size(0) == batch.num_nodes
        assert self.molecule_atom_types.size(0) == batch.num_nodes
        return self.molecule_types, self.molecule_atom_types
