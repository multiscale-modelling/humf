import torch
from torch_geometric.data import Data


def has_nans(data: Data) -> bool:
    for _, value in data.items():
        if torch.is_tensor(value):
            if torch.isnan(value).any():
                return True
    return False
