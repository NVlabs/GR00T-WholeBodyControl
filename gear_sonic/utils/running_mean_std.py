import torch
import torch.nn as nn

"""
updates statistic from a full data

Memory optimization note: If you encounter CUDA out of memory errors, consider setting:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
This can help reduce memory fragmentation.
"""


class RunningMeanStd(nn.Module):

    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.mean_size = insize[0]
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float32))
        self.register_buffer("count", torch.ones((), dtype=torch.float32))

        self.frozen = False
        self.frozen_partial = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def freeze_partial(self, diff):
        self.frozen_partial = True
        self.diff = diff

    def sync_across_gpus(self, accelerator):
        if accelerator.num_processes <= 1:
            return
        # ZL: this is the right formulation but a crude sync. The correct way is to sync
        # batch stats.
        flat_stats = torch.cat(
            [
                self.running_mean.flatten(),  # (D,)
                self.running_var.flatten(),  # (D,)
            ]
        )  # shape = (2D + C,)

        # Gather stats from all processes
        gathered = accelerator.gather(flat_stats[None])  # shape = (world_size, 2D + C)

        # Reshape
        world_size = gathered.shape[0]
        D = self.running_mean.numel()

        means_ = gathered[:, :D].reshape(world_size, *self.running_mean.shape).mean(dim=0)
        vars_ = gathered[:, D : 2 * D].reshape(world_size, *self.running_var.shape).mean(dim=0)

        # Update local stats
        self.running_mean.copy_(means_)
        self.running_var.copy_(vars_)

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        # change shape
        input_shape = input.shape
        if len(input.shape) == 3:
            input = input.reshape(-1, input_shape[-1])

        if self.per_channel:
            if len(self.insize) == 3:
                # Use broadcasting instead of expand_as to avoid memory issues
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1])
                current_var = self.running_var.view([1, self.insize[0], 1, 1])
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1])
                current_var = self.running_var.view([1, self.insize[0], 1])
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]])
                current_var = self.running_var.view([1, self.insize[0]])

        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var + self.epsilon) * y + current_mean
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var + self.epsilon)
            else:
                # Use in-place operations where possible to reduce memory usage
                y = input - current_mean
                y = y / torch.sqrt(current_var + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)

        # update After normalization, so that the values used for training and testing are the same.
        if self.training and not self.frozen:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)

            new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )
            if self.frozen_partial:
                # Only update the last bit (futures)
                self.running_mean[-self.diff :], self.running_var[-self.diff :], self.count = (
                    new_mean[-self.diff :],
                    new_var[-self.diff :],
                    new_count,
                )
            else:
                self.running_mean, self.running_var, self.count = new_mean, new_var, new_count

        if len(input_shape) == 3:
            y = y.view(input_shape)

        return y


class RunningMeanStdObs(nn.Module):

    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert isinstance(insize, dict)
        super().__init__()
        self.running_mean_std = nn.ModuleDict(
            {k: RunningMeanStd(v, epsilon, per_channel, norm_only) for k, v in insize.items()}
        )

    def forward(self, input, unnorm=False):
        res = {k: self.running_mean_std[k](v, unnorm) for k, v in input.items()}
        return res


from collections.abc import Sequence
from copy import deepcopy


class VecNorm(nn.Module):
    """Simple running normalization for observations.

    Keeps track of running mean and variance to normalize observations on-the-fly.

    Args:
        obs_keys: List of observation keys to normalize
        decay: Decay rate for running statistics (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-4)
        device: Device to store statistics on
    """

    def __init__(
        self,
        obs_keys: Sequence[str] | None,
        decay: float = 0.9999,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.obs_keys = obs_keys
        self.decay = decay
        self.eps = eps

        # Running statistics as nn.Parameters
        self.sum: dict[str, nn.Parameter] = nn.ParameterDict({})
        self.ssq: dict[str, nn.Parameter] = nn.ParameterDict({})
        self.cnt: dict[str, nn.Parameter] = nn.ParameterDict({})
        self.mean: dict[str, nn.Parameter] = nn.ParameterDict({})
        self.var: dict[str, nn.Parameter] = nn.ParameterDict({})

        self.initialized = False
        self.frozen = False

    def init_stats(self, obs_dict: dict[str, torch.Tensor]):
        """Initialize running statistics based on observation shapes."""
        if self.obs_keys is None:
            self.obs_keys = list(obs_dict.keys())
        for key in self.obs_keys:
            if key in obs_dict:
                v = obs_dict[key]
                self.sum[key] = nn.Parameter(torch.zeros_like(v[0]), requires_grad=False)
                self.ssq[key] = nn.Parameter(torch.zeros_like(v[0]), requires_grad=False)
                self.cnt[key] = nn.Parameter(torch.zeros_like(v[0]), requires_grad=False)
                self.mean[key] = nn.Parameter(torch.zeros_like(v[0]), requires_grad=False)
                self.var[key] = nn.Parameter(torch.ones_like(v[0]), requires_grad=False)
        self.initialized = True

    def update(self, obs_dict: dict[str, torch.Tensor]):
        """Update running statistics with new observations."""
        if not self.initialized:
            self.init_stats(obs_dict)

        if self.frozen:
            return

        for key in self.obs_keys:
            if key not in obs_dict:
                continue

            x = obs_dict[key]
            sum_x = x.sum(dim=0)
            ssq_x = (x**2).sum(dim=0)
            cnt_x = x.shape[0]

            self.sum[key] = self.sum[key] * self.decay + sum_x
            self.ssq[key] = self.ssq[key] * self.decay + ssq_x
            self.cnt[key] = self.cnt[key] * self.decay + cnt_x

            self.mean[key] = self.sum[key] / self.cnt[key]
            self.var[key] = (self.ssq[key] / self.cnt[key] - self.mean[key] ** 2).clamp(
                min=self.eps
            )

    def normalize(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize observations using running statistics."""
        if not self.initialized:
            self.init_stats(obs_dict)
            return obs_dict.copy()

        normalized = obs_dict.copy()
        for key in self.obs_keys:
            normalized[key] = (normalized[key] - self.mean[key]) / self.var[key].sqrt()

        return normalized

    def denormalize(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Denormalize observations using running statistics."""
        if not self.initialized:
            self.init_stats(obs_dict)
            return obs_dict.copy()

        denormalized = obs_dict.copy()
        for key in self.obs_keys:
            denormalized[key] = denormalized[key] * self.var[key].sqrt() + self.mean[key]

        return denormalized

    def freeze(self):
        """Freeze running statistics updates."""
        self.frozen = True

    def unfreeze(self):
        """Unfreeze running statistics updates."""
        self.frozen = False

    def get_stats(self):
        """Get current running statistics."""
        return {
            "sum": {key: param.data.clone() for key, param in self.sum.items()},
            "ssq": {key: param.data.clone() for key, param in self.ssq.items()},
            "cnt": {key: param.data.clone() for key, param in self.cnt.items()},
            "mean": {key: param.data.clone() for key, param in self.mean.items()},
            "var": {key: param.data.clone() for key, param in self.var.items()},
        }

    def load_stats(self, stats):
        """Load running statistics."""
        for key in stats["sum"]:
            self.sum[key] = nn.Parameter(deepcopy(stats["sum"][key]), requires_grad=False)
            self.ssq[key] = nn.Parameter(deepcopy(stats["ssq"][key]), requires_grad=False)
            self.cnt[key] = nn.Parameter(deepcopy(stats["cnt"][key]), requires_grad=False)
            self.mean[key] = nn.Parameter(deepcopy(stats["mean"][key]), requires_grad=False)
            self.var[key] = nn.Parameter(deepcopy(stats["var"][key]), requires_grad=False)
        self.initialized = True
