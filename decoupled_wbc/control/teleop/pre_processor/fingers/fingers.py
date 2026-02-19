from decoupled_wbc.control.teleop.pre_processor.pre_processor import PreProcessor


class FingersPreProcessor(PreProcessor):
    """Dummy class just takes out the fingers from the data."""

    def __init__(self, side: str):
        super().__init__()
        self.side = side

    def __call__(self, data):
        return data[f"{self.side}_fingers"]

    # TODO: calibrate max and min
    def calibrate(self, data, control_device):
        pass
