import torch


def has_nans(data) -> bool:
    for _, value in data.items():
        if torch.is_tensor(value):
            if torch.isnan(value).any():
                return True
    return False
