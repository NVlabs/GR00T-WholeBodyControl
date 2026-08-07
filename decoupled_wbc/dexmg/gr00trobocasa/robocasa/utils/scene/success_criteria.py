from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from robosuite.environments import MujocoEnv
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

from robocasa.utils import object_utils as OU

if TYPE_CHECKING:
    from robocasa.utils.scene.scene import SceneObject


class SuccessCriteria(ABC):
    @abstractmethod
    def is_true(self, env: MujocoEnv) -> bool:
        raise NotImplementedError


class IsUpright(SuccessCriteria):
    def __init__(self, obj: SceneObject, threshold=0.8, symmetric: bool = False):
        self._obj = obj
        self._threshold = threshold
        self._symmetric = symmetric

    def is_true(self, env: MujocoEnv) -> bool:
        return OU.check_obj_upright(
            env,
            obj_name=self._obj.mj_obj.name,
            threshold=self._threshold,
            symmetric=self._symmetric,
        )


class IsClose(SuccessCriteria):
    def __init__(
        self,
        obj_a: SceneObject,
        obj_b: SceneObject,
        max_distance: float,
        use_xy_only: bool = False,
    ):
        self._obj_a = obj_a
        self._obj_b = obj_b
        self._max_distance = max_distance
        self._use_xy_only = use_xy_only

    def is_true(self, env: MujocoEnv) -> bool:
        pos_a = env.sim.data.body_xpos[env.obj_body_id[self._obj_a.mj_obj.name]]
        pos_b = env.sim.data.body_xpos[env.obj_body_id[self._obj_b.mj_obj.name]]
        if self._use_xy_only:
            pos_a = pos_a[:2]
            pos_b = pos_b[:2]
        return np.linalg.norm(pos_a - pos_b) <= self._max_distance


class IsInContact(SuccessCriteria):
    def __init__(self, obj_a: SceneObject, obj_b: SceneObject):
        self._obj_a = obj_a
        self._obj_b = obj_b

    def is_true(self, env: MujocoEnv) -> bool:
        return env.check_contact(self._obj_a.mj_obj, self._obj_b.mj_obj)


class IsGripperFar(SuccessCriteria):
    def __init__(self, obj: SceneObject, threshold: float = 0.4):
        self._threshold = threshold
        self._obj = obj

    def is_true(self, env: MujocoEnv) -> bool:
        return OU.any_gripper_obj_far(env, obj_name=self._obj.mj_obj.name, th=self._threshold)


class IsGrasped(SuccessCriteria):
    _GRIPPERS = {"left", "right"}

    def __init__(self, obj: SceneObject, gripper: Optional[str] = None):
        self._obj = obj
        self._grippers = {gripper} if gripper in self._GRIPPERS else self._GRIPPERS

    def is_true(self, env: MujocoEnv) -> bool:
        assert isinstance(env, ManipulationEnv), "Expected a ManipulationEnv instance."
        robot = env.robots[0]
        return any(env._check_grasp(robot.gripper[g], self._obj.mj_obj) for g in self._grippers)


class IsRobotInRange(SuccessCriteria):
    def __init__(self, target: SceneObject, threshold: float, planar: bool):
        self._target = target
        self._threshold = threshold
        self._planar = planar

    def is_true(self, env: MujocoEnv) -> bool:
        robot = env.robots[0]
        robot_pos = env.sim.data.get_body_xpos(robot.robot_model.root_body)
        obj_pos = env.sim.data.body_xpos[env.obj_body_id[self._target.mj_obj.name]]
        if self._planar:
            robot_pos = robot_pos[:2]
            obj_pos = obj_pos[:2]
        return np.linalg.norm(robot_pos - obj_pos) <= self._threshold


class IsPositionInRange(SuccessCriteria):
    def __init__(
        self,
        target: SceneObject,
        axis_index: int,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        self._target = target
        self._axis = axis_index
        self._min = min_val or float("-inf")
        self._max = max_val or float("inf")

    def is_true(self, env: MujocoEnv) -> bool:
        pos = env.sim.data.body_xpos[env.obj_body_id[self._target.mj_obj.name]]
        return self._min <= pos[self._axis] <= self._max


class IsJointQposInRange(SuccessCriteria):
    def __init__(self, obj: SceneObject, joint_id: int, min_val: float, max_val: float):
        self._obj = obj
        self._joint_id = joint_id
        self._min = min_val
        self._max = max_val

    def is_true(self, env: MujocoEnv) -> bool:
        qpos = env.sim.data.get_joint_qpos(self._obj.mj_obj.joints[self._joint_id])
        return self._min <= qpos <= self._max


class NotCriteria(SuccessCriteria):
    def __init__(self, criteria: SuccessCriteria):
        self.criteria = criteria

    def is_true(self, env: MujocoEnv) -> bool:
        return not self.criteria.is_true(env)


class AllCriteria(SuccessCriteria):
    def __init__(self, *criteria: SuccessCriteria):
        self.criteria = criteria

    def is_true(self, env: MujocoEnv) -> bool:
        return all(c.is_true(env) for c in self.criteria)


class AnyCriteria(SuccessCriteria):
    def __init__(self, *criteria: SuccessCriteria):
        self.criteria = criteria

    def is_true(self, env: MujocoEnv) -> bool:
        return any(c.is_true(env) for c in self.criteria)
