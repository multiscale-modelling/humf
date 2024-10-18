import torch


def has_no_nans(data) -> bool:
    for _, value in data.items():
        if torch.is_tensor(value):
            if torch.isnan(value).any():
                return False
    return True
