class EpisodeState:
    """Episode state controller for data collection.

    Manages the state transitions for episode recording:
    - IDLE: Not recording
    - RECORDING: Currently recording data
    - NEED_TO_SAVE: Recording stopped, waiting to save
    """

    def __init__(self):
        self.RECORDING = "recording"
        self.IDLE = "idle"
        self.NEED_TO_SAVE = "need_to_save"

        self.state = self.IDLE

    def change_state(self):
        """Cycle through states: IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE."""
        if self.state == self.IDLE:
            self.state = self.RECORDING
        elif self.state == self.RECORDING:
            self.state = self.NEED_TO_SAVE
        elif self.state == self.NEED_TO_SAVE:
            self.state = self.IDLE

    def reset_state(self):
        """Reset to IDLE state."""
        self.state = self.IDLE

    def get_state(self):
        """Get current state."""
        return self.state
