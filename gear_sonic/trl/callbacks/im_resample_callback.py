from transformers import TrainerCallback


class ImResampleCallback(TrainerCallback):
    """Callback to resample motion during training. Supports multigpu ."""

    def __init__(self, motion_resample_frequency, skip_resample_frequency=None):
        super().__init__()
        self.motion_resample_frequency = motion_resample_frequency
        self.skip_resample_frequency = skip_resample_frequency

    def on_step_end(self, args, state, control, **kwargs):

        self.env = kwargs.get("env")
        self.accelerator = kwargs.get("accelerator")
        self.device = self.accelerator.device

        should_resample = (state.global_step + 1) % self.motion_resample_frequency == 0
        should_skip = (
            self.skip_resample_frequency is not None
            and (state.global_step + 1) % self.skip_resample_frequency == 0
        )

        if should_resample and not should_skip:
            self.env.resample_motion()
