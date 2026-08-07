import logging
import os
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class Stats(torch.nn.Module):
    """Simple class to handle stats with pytorch.

    Similar to:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    for handling precision

    (data - mean) / np.sqrt(var + eps)
    """

    def __init__(
        self,
        folder: Optional[str] = None,
        load: bool = True,
        eps=1e-05,
        legacy=False,
    ):
        super().__init__()
        self.legacy = legacy
        # for legacy behavior, the std stats are already cliped to 1e-3

        self.folder = folder
        self.eps = eps
        if folder is not None and load:
            self.load()

    def load(self):
        mean = torch.from_numpy(np.load(os.path.join(self.folder, "mean.npy")))
        std = torch.from_numpy(np.load(os.path.join(self.folder, "std.npy")))
        self.register_from_tensors(mean, std)

    def register_from_tensors(self, mean: torch.Tensor, std: torch.Tensor):
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def normalize(self, data: torch.Tensor, index=None) -> torch.Tensor:
        mean = self.mean.to(device=data.device, dtype=data.dtype)
        std = self.std.to(device=data.device, dtype=data.dtype)

        if index is not None:
            mean = mean[..., index]
            std = std[..., index]

        if self.legacy:
            return (data - mean) / torch.clip(std, 1e-3)

        # adjust std with eps
        return (data - mean) / torch.sqrt(std**2 + self.eps)

    def unnormalize(self, data: torch.Tensor, index=None) -> torch.Tensor:
        mean = self.mean.to(device=data.device, dtype=data.dtype)
        std = self.std.to(device=data.device, dtype=data.dtype)

        if index is not None:
            mean = mean[..., index]
            std = std[..., index]

        if self.legacy:
            return data * torch.clip(std, 1e-3) + mean

        # adjust std with eps
        return data * torch.sqrt(std**2 + self.eps) + mean

    def is_loaded(self):
        return hasattr(self, "mean")

    def get_dim(self):
        return self.mean.shape[0]

    def save(
        self,
        folder: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        if folder is None:
            folder = self.folder
            if folder is None:
                raise ValueError("No folder to save stats")

        if mean is None and std is None:
            try:
                mean = self.mean.cpu().numpy()
                std = self.std.cpu().numpy()
            except AttributeError:
                raise ValueError("Stats were not loaded")

        # don't override stats folder
        os.makedirs(folder, exist_ok=False)

        np.save(os.path.join(folder, "mean.npy"), mean)
        np.save(os.path.join(folder, "std.npy"), std)

    def __eq__(self, other):
        return (self.mean.cpu() == other.mean.cpu()).all() and (
            self.std.cpu() == other.std.cpu()
        ).all()

    # should define a hash value for pytorch, as we defined __eq__
    def __hash__(self):
        # Convert mean and std to bytes for a consistent hash value
        mean_hash = hash(self.mean.detach().cpu().numpy().tobytes())
        std_hash = hash(self.std.detach().cpu().numpy().tobytes())
        return hash((mean_hash, std_hash))

    def __repr__(self):
        return f'Stats(folder="{self.folder}")'
