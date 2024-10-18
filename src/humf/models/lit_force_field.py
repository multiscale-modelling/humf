# pyright: reportIncompatibleMethodOverride=false

import lightning as L
import torch
from torch import Tensor
from torch.nn import Module


class LitForceField(L.LightningModule):
    def __init__(
        self,
        force_field: Module,
        learning_rate: float,
        energy_weight: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["force_field"])
        self.force_field = force_field
        self.learning_rate = learning_rate
        self.energy_weight = energy_weight

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        return self.force_field(batch)

    def training_step(self, batch):
        metrics = self._common_step(batch)
        metrics = {f"train/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, batch_size=batch.batch_size)
        return metrics["train/loss"]

    @torch.enable_grad()
    def validation_step(self, batch):
        metrics = self._common_step(batch)
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, batch_size=batch.batch_size)
        return metrics["val/loss"]

    @torch.enable_grad()
    def test_step(self, batch):
        metrics = self._common_step(batch)
        metrics = {f"test/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, batch_size=batch.batch_size)
        return metrics["test/loss"]

    def _common_step(self, batch):
        predicted_energy, predicted_forces = self(batch)
        target_energy, target_forces = batch.energy, batch.forces
        # Divide energy MSE loss by batch size.
        loss_energy = torch.nn.functional.mse_loss(
            predicted_energy, target_energy, reduction="mean"
        )
        # Divide forces MSE loss by number of atoms in batch.
        loss_forces = torch.nn.functional.mse_loss(
            predicted_forces, target_forces, reduction="sum"
        ) / predicted_forces.size(0)
        loss = loss_energy * self.energy_weight + loss_forces * (1 - self.energy_weight)
        metrics = {
            "loss": loss,
            "loss_energy": loss_energy,
            "loss_forces": loss_forces,
        }
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
