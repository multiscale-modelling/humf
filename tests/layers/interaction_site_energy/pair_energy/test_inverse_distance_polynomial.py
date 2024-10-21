import torch

from humf.layers.interaction_site_energy.pair_energy.inverse_distance_polynomial import (
    InverseDistancePolynomial,
)


class TestInverseDistancePolynomial:
    def test_forward(self):
        polynomial = InverseDistancePolynomial(orders=3)
        distances = torch.tensor([2.0, 2.0])
        prameters = torch.ones(2, 2, 3)
        contribs = polynomial(distances, prameters)
        sum = torch.sum(contribs)
        expected = torch.tensor(1.75)
        assert torch.allclose(sum, expected, atol=1e-4)

        double_params = prameters * 2
        contribs_2 = polynomial(distances, double_params)
        sum_2 = torch.sum(contribs_2)
        expected_2 = expected * 2
        assert torch.allclose(sum_2, expected_2, atol=1e-4)
