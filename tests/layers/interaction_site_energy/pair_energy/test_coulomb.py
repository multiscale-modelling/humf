import torch

from humf.layers.interaction_site_energy.pair_energy.coulomb import Coulomb


class TestCoulomb:
    def test_forward(self):
        coulomb = Coulomb()
        distances = torch.tensor([2.0, 2.0])
        prameters = torch.ones(2, 2)
        contribs = coulomb(distances, prameters)
        sum = torch.sum(contribs)
        expected = torch.tensor(332.0637)
        assert torch.allclose(sum, expected, atol=1e-4)

        double_distances = distances * 2
        double_params = prameters * 2
        contribs_2 = coulomb(double_distances, double_params)
        sum_2 = torch.sum(contribs_2)
        expected_2 = expected * 2
        assert torch.allclose(sum_2, expected_2, atol=1e-4)
