import torch
from torch.nn import Module


class HumfNoClassifier(Module):
    """Humf model without classifiers.

    Works the same as the Humf model, but assumes that interaction and molecule
    classification is done outside of the model, e.g. in a data preprocessing
    step, and that the classification results are included in the data batch.
    """

    def __init__(
        self,
        interaction_sites: Module,
        site_energy: Module,
    ):
        super().__init__()
        self.interaction_sites = interaction_sites
        self.site_energy = site_energy

    def forward(self, batch):
        # TODO: Evaluate intramolecular energy.
        intra_molecular_energy = torch.zeros(batch.batch_size)

        # Evaluate intermolecular energy.
        (
            interaction_site_positions,
            interaction_site_parameteres,
            interaction_site_edge_index,
            interaction_site_batch,
        ) = self.interaction_sites(
            batch,
            batch.interaction_types,
            batch.molecule_types,
            batch.molecule_atom_types,
        )
        intermolecular_energy = self.site_energy(
            batch,
            interaction_site_positions,
            interaction_site_parameteres,
            interaction_site_edge_index,
            interaction_site_batch,
        )

        return intra_molecular_energy + intermolecular_energy
