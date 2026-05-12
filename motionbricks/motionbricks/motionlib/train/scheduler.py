import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing to a final learning rate."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.num_warmup_steps:
            return [
                base_lr * float(current_step) / float(max(1, self.num_warmup_steps))
                for base_lr in self.base_lrs
            ]
        else:
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return [
                self.final_lr + (base_lr - self.final_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
