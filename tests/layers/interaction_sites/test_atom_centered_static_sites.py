from unittest.mock import MagicMock

import torch

from humf.layers.interaction_sites.atom_centered_static_sites import (
    AtomCenteredStaticSites,
)


class TestAtomCenteredStaticSites:
    def test_forward(self):
        initial_type_parameters = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        atom_centered_static_sites = AtomCenteredStaticSites(initial_type_parameters)

        mock_batch = MagicMock()
        mock_batch.positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        )
        mock_batch.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        mock_batch.batch = torch.tensor([0, 0, 1, 1])

        interaction_types = torch.tensor([0, 0, 1, 1])
        molecule_types = torch.tensor([0, 0, 1, 1])
        molecule_atom_types = torch.tensor([0, 1, 2, 2])

        sites_positions, sites_parameters, sites_edge_index, sites_batch = (
            atom_centered_static_sites(
                mock_batch, interaction_types, molecule_types, molecule_atom_types
            )
        )

        expected_sites_parameters = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]]
        )
        expected_sites_edge_index = torch.tensor([[1, 2], [2, 1]])
        assert torch.equal(sites_positions, mock_batch.positions)
        assert torch.equal(sites_parameters, expected_sites_parameters)
        assert torch.equal(sites_edge_index, expected_sites_edge_index)
        assert torch.equal(sites_batch, mock_batch.batch)
