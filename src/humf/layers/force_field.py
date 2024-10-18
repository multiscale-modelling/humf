import torch
from torch import Tensor
from torch.nn import Module


class ForceField(Module):
    def __init__(self, energy_model: Module) -> None:
        super().__init__()
        self.energy_model = energy_model
        self.energy_offset = torch.nn.Parameter(torch.tensor(0.0))

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
