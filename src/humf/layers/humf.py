import torch
from torch.nn import Module


class Humf(Module):
    def __init__(
        self,
        interaction_classifier: Module,
        molecule_classifier: Module,
        interaction_sites: Module,
        site_energy: Module,
    ):
        super().__init__()
        self.interaction_classifier = interaction_classifier
        self.molecule_classifier = molecule_classifier
        self.interaction_sites = interaction_sites
        self.site_energy = site_energy

    def forward(self, batch):
        interaction_types = self.interaction_classifier(batch)
        molecule_types, molecule_atom_types = self.molecule_classifier(
            batch, interaction_types
        )

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
            interaction_types,
            molecule_types,
            molecule_atom_types,
        )
        intermolecular_energy = self.site_energy(
            batch,
            interaction_site_positions,
            interaction_site_parameteres,
            interaction_site_edge_index,
            interaction_site_batch,
        )

        return intra_molecular_energy + intermolecular_energy
