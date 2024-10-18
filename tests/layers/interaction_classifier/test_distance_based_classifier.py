from unittest.mock import MagicMock

import torch

from humf.layers.interaction_classifier.distance_based_classifier import (
    DistanceBasedClassifier,
)


class TestDistanceBasedClassifier:
    def test_forward(self):
        distance_based_classifier = DistanceBasedClassifier(cutoff=3.0)
        mock_batch = MagicMock()
        mock_batch.edge_distances = torch.tensor([2.0, 4.0, 1.0])
        assert torch.equal(
            distance_based_classifier(mock_batch), torch.tensor([0, 1, 0])
        )
