from humf.data.ase_dataset import ASEDataset


class TestASEDataset:
    def test_init(self):
        ASEDataset("tests/inputs/data/", force_reload=True)
