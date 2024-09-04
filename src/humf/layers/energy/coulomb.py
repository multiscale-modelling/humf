from scipy import constants as c
from torch import nn

# fmt: off
CONVERSION_FACTOR = (
    (c.elementary_charge**2 * c.Avogadro)
    / (4 * c.pi * c.epsilon_0 * c.angstrom * c.kilo * c.calorie)
)
# fmt: on


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
        return CONVERSION_FACTOR * charges_1 * charges_2 / distances
