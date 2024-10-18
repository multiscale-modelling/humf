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

    def forward(self, distances, charges):
        """Electrostatic energies between pairs of charges.

        Parameters
        ----------
        distances : torch.Tensor
            Tensor of shape (num_pairs,) with the distances between pairs of charges,
            in units of Angstrom.
        charges : torch.Tensor
            Tensor of shape (2, num_pairs,) with the charges of the atoms in
            each pair, in units of elementary charge.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_pairs,) with the electrostatic energies
            between pairs of charges, in units of kcal/mol.
        """
        return CONVERSION_FACTOR * charges[0] * charges[1] / distances
