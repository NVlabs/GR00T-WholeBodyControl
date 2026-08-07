import torch
import torch.nn as nn


class BatchNormNormalizer(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        assert len(insize) == 1, "BatchNormNormalizer only supports 1D observation spaces"
        self._normalizer = nn.SyncBatchNorm(num_features=insize[0], affine=False)

    @property
    def num_features(self):
        return self._normalizer.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])

        x = self._normalizer(x)
        x = x.view(input_shape)
        return x

    def update(self, input: torch.Tensor):
        """Update running stats from input. No-op in eval mode.

        Calls SyncBatchNorm.forward() for its side effect of updating
        running_mean/running_var (and multi-GPU sync). Output is discarded.
        """
        if not self.training:  # do nothing if in evaluation mode
            return
        if len(input.shape) == 3:
            input = input.reshape(-1, input.shape[-1])
        with torch.no_grad():
            self._normalizer(input)

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize using current running stats without updating them."""
        input_shape = input.shape
        if len(input.shape) == 3:
            input = input.reshape(-1, input_shape[-1])
        y = (input - self._normalizer.running_mean) / torch.sqrt(
            self._normalizer.running_var + self._normalizer.eps
        )
        y = torch.clamp(y, min=-5.0, max=5.0)
        if len(input_shape) == 3:
            y = y.view(input_shape)
        return y
