import gymnasium as gym


class Env:
    """Base interface for all environments in the Gr00t framework"""

    def observe(self) -> dict[str, any]:
        """Read the current state of this environment

        Returns:
            dict: A dictionary of observations
        """
        pass

    def queue_action(self, action: dict[str, any]):
        """Queue an action to be executed

        Args:
            action: A dictionary of action parameters
        """
        pass

    def reset(self, **kwargs):
        """Reset this environment to initial state"""
        pass

    def observation_space(self) -> gym.Space:
        """Get the observation space of this environment

        Returns:
            gym.Space: The observation space
        """
        pass

    def action_space(self) -> gym.Space:
        """Get the action space of this environment

        Returns:
            gym.Space: The action space
        """
        pass

    def close(self):
        """Close and clean up this environment"""
        pass
