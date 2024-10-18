import torch
from torch.nn import Module


class DistanceBasedClassifier(Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, batch):
        return (batch.edge_distances > self.cutoff).to(torch.int32)
