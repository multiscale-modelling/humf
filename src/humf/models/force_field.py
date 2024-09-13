# pyright: reportIncompatibleMethodOverride=false

import lightning as L
import torch
from torch import Tensor
from torch.nn import Module


class ForceField(L.LightningModule):
    def __init__(
        self,
        energy_model: Module,
        learning_rate: float,
        trade_off: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["energy_model"])
        self.energy_model = energy_model
        self.energy_offset = torch.nn.Parameter(torch.tensor(0.0))
        self.learning_rate = learning_rate
        self.trade_off = trade_off

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        batch.pos.requires_grad_(True)

        energy = self.energy_model(batch) + self.energy_offset

        grad_outputs: list[Tensor | None] = [torch.ones_like(energy)]
        forces = torch.autograd.grad(
            [energy],
            [batch.pos],
            grad_outputs=grad_outputs,  # type: ignore
            create_graph=True,
        )[0]
        assert forces is not None
        forces = -forces

        return energy, forces

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
        loss = loss_energy * self.trade_off + loss_forces * (1 - self.trade_off)
        metrics = {
            "loss": loss,
            "loss_energy": loss_energy,
            "loss_forces": loss_forces,
        }
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
