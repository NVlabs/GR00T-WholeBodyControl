from copy import deepcopy
from typing import Optional

import gymnasium as gym

from decoupled_wbc.control.base.policy import Policy


class IdentityPolicy(Policy):
    def __init__(self):
        self.reset()

    def get_action(self, time: Optional[float] = None) -> dict[str, any]:
        return self.goal

    def set_goal(self, goal: dict[str, any]) -> None:
        self.goal = deepcopy(goal)
        self.goal.pop("interpolation_garbage_collection_time", None)
        self.goal.pop("target_time", None)

    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict()

    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict()
