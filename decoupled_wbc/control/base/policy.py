from abc import ABC, abstractmethod
from typing import Optional


class Policy(ABC):
    """Base class for implementing control policies in the Gr00t framework.

    A Policy defines how an agent should behave in an environment by mapping observations
    to actions. This abstract base class provides the interface that all concrete policy
    implementations must follow.
    """

    def set_goal(self, goal: dict[str, any]):
        """Set the command from the planner that the policy should follow.

        Args:
            goal: Dictionary containing high-level commands or goals from the planner
        """
        pass

    def set_observation(self, observation: dict[str, any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing the current state/observation of the environment
        """
        self.observation = observation

    @abstractmethod
    def get_action(self, time: Optional[float] = None) -> dict[str, any]:
        """Compute and return the next action at the specified time, based on current observation
        and planner command.

        Args:
            time: Optional "monotonic time" for time-dependent policies

        Returns:
            Dictionary containing the action to be executed
        """

    def close(self):
        """Clean up any resources used by the policy."""
        pass

    def reset(self):
        """Reset the policy to its initial state."""
        pass
