from math import pi as PI

from torch import nn

ANGSTROM = 1e-10  # m
ELEMENTARY_CHARGE = 1.602176634e-19  # C
KCAL = 4184  # J
MOL = 6.02214076e23  # Avogadro's number

EPSILON_0 = 8.854187817e-12  # C/(Vm)


class Coulomb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distances, charges_1, charges_2):
        """Electrostatic energies between pairs of charges.

        Parameters
        ----------
        distances : torch.Tensor
            Tensor of shape (num_pairs,) with the distances between pairs of charges,
            in units of Angstrom.
        charges_1 : torch.Tensor
            Tensor of shape (num_pairs,) with the charges of the first atom in each pair,
            in units of elementary charge.
        charges_2 : torch.Tensor
            As charges_1, but for the second atom in each pair.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_pairs,) with the electrostatic energies between pairs of charges,
            in units of kcal/mol.
        """
        return (
            (1 / (4 * PI * EPSILON_0))
            * (charges_1 * charges_2 * ELEMENTARY_CHARGE**2 / (distances * ANGSTROM))
            / KCAL
            * MOL
        )
