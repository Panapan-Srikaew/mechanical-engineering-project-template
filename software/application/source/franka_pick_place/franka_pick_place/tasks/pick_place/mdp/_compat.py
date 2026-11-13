import torch


def as_torch(x) -> torch.Tensor:
    return x.torch if hasattr(x, "torch") else x
