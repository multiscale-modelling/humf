import torch
from torch import nn


class LennardJones(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances, params):
        """Lennard-Jones energies between pairs of sites.

        Parameters
        ----------
        distances : torch.Tensor
            Tensor of shape (num_pairs,) with the distances between pairs of
            sites, in units of Angstrom.
        params: torch.Tensor
            Tensor of shape (2, num_pairs, 2) with the parameters of the sites
            in each pair. Entries (..., 0) contain the well depths epsilon, in
            units of kcal/mol. Entries (..., 1) contain the zero crossing
            sigma, in units of Angstrom.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_pairs,) with the Lennard-Jones energies between pairs of sites,
            in units of kcal/mol.
        """
        # Geometric mixing
        # delta = 1e-8  # Avoids division by zero in the derivative of sqrt.
        # epsilon = torch.sqrt(torch.abs(params[0, :, 0] * params[1, :, 0]) + delta)
        # sigma = torch.sqrt(torch.abs(params[0, :, 1] * params[1, :, 1]) + delta)

        # Arithmetic mixing
        epsilon = 0.5 * (torch.abs(params[0, :, 0]) + torch.abs(params[1, :, 0]))
        sigma = 0.5 * (torch.abs(params[0, :, 1]) + torch.abs(params[1, :, 1]))

        r = sigma / distances
        r6 = r**6
        r12 = r6**2
        # Divide by 2 to avoid double counting.
        return 4 * epsilon * (r12 - r6) / 2
