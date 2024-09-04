import torch
from torch import nn


class LennardJones(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances, params_1, params_2):
        """Lennard-Jones energies between pairs of sites.

        Parameters
        ----------
        distances : torch.Tensor
            Tensor of shape (num_pairs,) with the distances between pairs of sites,
            in units of Angstrom.
        params_1 : torch.Tensor
            Tensor of shape (num_pairs, 2) with the parameters of the first site in each pair.
            The first column contains the well depth epsilon, in units of kcal/mol.
            The second column contains the zero crossing sigma, in units of Angstrom.
        params_2 : torch.Tensor
            As params_1, but for the second site in each pair.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_pairs,) with the Lennard-Jones energies between pairs of sites,
            in units of kcal/mol.
        """
        epsilon = torch.sqrt(torch.abs(params_1[:, 0] * params_2[:, 0]))
        sigma = torch.sqrt(torch.abs(params_1[:, 1] * params_2[:, 1]))
        r = sigma / distances
        r6 = r**6
        r12 = r6**2
        energy = 4 * epsilon * (r12 - r6)

        # # DEBUG: Assert that all intermediate values are not NaN.
        # assert not torch.isnan(epsilon).any()
        # assert not torch.isnan(sigma).any()
        # assert not torch.isnan(r).any()
        # assert not torch.isnan(r6).any()
        # assert not torch.isnan(r12).any()
        # assert not torch.isnan(energy).any()
        #
        # # DEBUG: Assert that all intermediate values are finite.
        # assert torch.isfinite(epsilon).all()
        # assert torch.isfinite(sigma).all()
        # assert torch.isfinite(r).all()
        # assert torch.isfinite(r6).all()
        # assert torch.isfinite(r12).all()
        # assert torch.isfinite(energy).all()
        #
        # # DEBUG: Print max and min of all intermediate values.
        # print("distances", distances.min(), distances.max())
        # print("epsilon", epsilon.min(), epsilon.max())
        # print("sigma", sigma.min(), sigma.max())
        # print("r", r.min(), r.max())
        # print("r6", r6.min(), r6.max())
        # print("r12", r12.min(), r12.max())
        # print("energy", energy.min(), energy.max())

        return energy
