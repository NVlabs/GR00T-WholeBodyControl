"""Running-average utility classes for tracking scalar and tensor metrics.

Three classes are provided at increasing levels of generality:

* ``AverageMeter`` – fixed-capacity windowed mean for batched tensors,
  implemented as an ``nn.Module`` with a registered buffer so it moves with
  the model between devices.
* ``TensorAverageMeter`` – simple accumulate-then-mean helper for a single
  metric.
* ``TensorAverageMeterDict`` – dictionary wrapper around
  ``TensorAverageMeter`` for tracking multiple named metrics simultaneously.
"""

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(nn.Module):
    """Windowed running mean with a configurable sample capacity.

    Maintains a weighted mean over the last ``max_size`` samples.  Implemented
    as an ``nn.Module`` so ``self.mean`` is a registered buffer and follows
    ``.to(device)`` / ``.cuda()`` calls automatically.

    Args:
        in_shape: Shape of the per-sample value tensor (passed to
            ``torch.zeros``).
        max_size: Maximum effective sample count used when computing the
            running mean.  Older samples are down-weighted once this limit
            is reached.
    """

    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        """Incorporate a new batch of values into the running mean.

        Args:
            values: Tensor of shape ``(batch, *in_shape)``.  Empty batches
                are silently skipped.
        """
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        """Reset the meter to its initial empty state."""
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        """Return the current effective sample count."""
        return self.current_size

    def get_mean(self):
        """Return the current mean as a NumPy array on CPU.

        Returns:
            np.ndarray of shape ``in_shape`` with the leading size-1 dimension
            squeezed out.
        """
        return self.mean.squeeze(0).cpu().numpy()


class TensorAverageMeter:
    """Accumulate tensors and compute their mean on demand.

    Unlike ``AverageMeter`` this class keeps all accumulated tensors in a list
    and concatenates them lazily when ``mean()`` is called.  It is best suited
    for metrics that are computed once per rollout step and averaged at the
    end of an epoch.
    """

    def __init__(self):
        self.tensors = []

    def add(self, x):
        """Append a tensor to the accumulator.

        Scalar tensors (0-D) are automatically unsqueezed to 1-D before
        appending so that concatenation works correctly.

        Args:
            x: Tensor to accumulate.
        """
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        """Return the mean of all accumulated tensors.

        Returns:
            A scalar tensor if tensors have been added, or the integer ``0``
            when the accumulator is empty or contains no elements.
        """
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        else:
            return cat.mean()

    def clear(self):
        """Discard all accumulated tensors."""
        self.tensors = []

    def mean_and_clear(self):
        """Compute the mean, clear the accumulator, and return the mean.

        Returns:
            Same as ``mean()``.
        """
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    """Dictionary of ``TensorAverageMeter`` objects for multi-metric tracking.

    Accepts batches of ``{key: tensor}`` dicts and lazily creates a
    ``TensorAverageMeter`` per key.  Uses plain ``dict`` internally (not
    ``defaultdict``) to avoid lambda pickling issues with DDP.
    """

    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        """Append a batch of named metric tensors to their respective meters.

        Args:
            data_dict: Mapping from metric name to tensor value.  New keys are
                registered automatically.
        """
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        """Return a dict mapping each key to its accumulated mean.

        Returns:
            Dict[str, scalar tensor | int] with the same keys as were added.
        """
        mean_dict = {k: v.mean() for k, v in self.data.items()}
        return mean_dict

    def clear(self):
        """Discard all meters and their accumulated data."""
        self.data = {}

    def mean_and_clear(self):
        """Compute means for all keys, clear the meters, and return the means.

        Returns:
            Same as ``mean()``.
        """
        mean = self.mean()
        self.clear()
        return mean
