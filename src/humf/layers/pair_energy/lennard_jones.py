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
        # Geometric mixing
        # delta = 1e-8  # Avoids division by zero in the derivative of sqrt.
        # epsilon = torch.sqrt(torch.abs(params_1[:, 0] * params_2[:, 0]) + delta)
        # sigma = torch.sqrt(torch.abs(params_1[:, 1] * params_2[:, 1]) + delta)

        # Arithmetic mixing
        epsilon = 0.5 * (torch.abs(params_1[:, 0]) + torch.abs(params_2[:, 0]))
        sigma = 0.5 * (torch.abs(params_1[:, 1]) + torch.abs(params_2[:, 1]))

        r = sigma / distances
        r6 = r**6
        r12 = r6**2
        return 4 * epsilon * (r12 - r6)
