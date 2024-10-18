from unittest.mock import MagicMock

import torch

from humf.layers.molecule_classifier.manual_classifier import ManualClassifier


class TestManualClassifier:
    def test_forward(self):
        molecule_types = torch.tensor([0, 1, 1, 2])
        molecule_atom_types = torch.tensor([0, 1, 2, 3])
        manual_classifier = ManualClassifier(molecule_types, molecule_atom_types)
        mock_batch = MagicMock()
        mock_batch.num_nodes = 4
        interaction_types = torch.tensor([0, 1, 0])
        assert all(
            torch.equal(x, y)
            for x, y in zip(
                manual_classifier(mock_batch, interaction_types),
                (molecule_types, molecule_atom_types),
            )
        )
