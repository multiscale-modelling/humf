import pytest

from humf.data.ase_dataset import ASEDataset

atom_count = 9


@pytest.fixture
def ase_dataset():
    root = "tests/inputs/data/"
    return ASEDataset(root, force_reload=True)


class TestASEDataset:
    def test_shapes(self, ase_dataset):
        for sample in ase_dataset:
            assert sample.pos.shape == (atom_count, 3)
            assert sample.types.shape == (atom_count,)
            assert sample.energy.shape == (1,)
            assert sample.forces.shape == (atom_count, 3)
            assert sample.forces_total.shape == (atom_count, 3)
            assert sample.energy_total.shape == (1,)
