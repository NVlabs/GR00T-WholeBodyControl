import numbers
import time as time_module
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import scipy.interpolate as si

from decoupled_wbc.control.base.policy import Policy


class InterpolationPolicy(Policy):
    def __init__(
        self,
        init_time: float,
        init_values: dict[str, np.ndarray],
        max_change_rate: float,
    ):
        """
        Args:
            init_time: The time of recording the initial values.
            init_values: The initial values of the features.
                The keys are the names of the features, and the values
                are the initial values of the features (1D array).
            max_change_rate: The maximum change rate.
        """
        super().__init__()
        self.last_action = init_values  # Vecs are 1D arrays
        self.concat_order = sorted(init_values.keys())
        self.concat_dims = []
        for key in self.concat_order:
            vec = np.array(init_values[key])
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
                init_values[key] = vec
            assert vec.ndim == 1, f"The shape of {key} should be (D,). Got {vec.shape}."
            self.concat_dims.append(vec.shape[0])

        self.init_values_concat = self._concat_vecs(init_values, 1)
        self.max_change_rate = max_change_rate
        self.reset(init_time)

    def reset(self, init_time: float = time_module.monotonic()):
        self.interp = PoseTrajectoryInterpolator(np.array([init_time]), self.init_values_concat)
        self.last_waypoint_time = init_time
        self.max_change_rate = self.max_change_rate

    def _concat_vecs(self, values: dict[str, np.ndarray], length: int) -> np.ndarray:
        """
        Concatenate the vectors into a 2D array to be used for interpolation.
        Args:
            values: The values to concatenate.
            length: The length of the concatenated vectors (time dimension).
        Returns:
            The concatenated vectors (T, D) arrays.
        """
        concat_vecs = []
        for key in self.concat_order:
            if key in values:
                vec = np.array(values[key])
                if vec.ndim == 1:
                    # If the vector is 1D, tile it to the length of the time dimension
                    vec = np.tile(vec, (length, 1))
                assert vec.ndim == 2, f"The shape of {key} should be (T, D). Got {vec.shape}."
                concat_vecs.append(vec)
            else:
                # If the vector is not in the values, use the last action
                # Since the last action is 1D, we need to tile it to the length of the time dimension
                concat_vecs.append(np.tile(self.last_action[key], (length, 1)))
        return np.concatenate(concat_vecs, axis=1)  # Vecs are 2D (T, D) arrays

    def _unconcat_vecs(self, concat_vec: np.ndarray) -> dict[str, np.ndarray]:
        curr_idx = 0
        action = {}
        assert (
            concat_vec.ndim == 1
        ), f"The shape of the concatenated vectors should be (T, D). Got {concat_vec.shape}."
        for key, dim in zip(self.concat_order, self.concat_dims):
            action[key] = concat_vec[curr_idx : curr_idx + dim]
            curr_idx += dim
        return action  # Vecs are 1D arrays

    def __call__(
        self, observation: Dict[str, Any], goal: Dict[str, Any], time: float
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError(
            "`InterpolationPolicy` accepts goal and provide action in two separate methods."
        )

    def set_goal(self, goal: Dict[str, Any]) -> None:
        if "target_time" not in goal:
            return
        assert (
            "interpolation_garbage_collection_time" in goal
        ), "`interpolation_garbage_collection_time` is required."
        target_time = goal.pop("target_time")
        interpolation_garbage_collection_time = goal.pop("interpolation_garbage_collection_time")

        if isinstance(target_time, list):
            for key, vec in goal.items():
                assert isinstance(vec, list)
                assert len(vec) == len(target_time), (
                    f"The length of {key} and `target_time` should be the same. "
                    f"Got {len(vec)} and {len(target_time)}."
                )
        else:
            target_time = [target_time]
            for key in goal:
                goal[key] = [goal[key]]

        # Concatenate all vectors in goal
        concat_vecs = self._concat_vecs(goal, len(target_time))
        assert concat_vecs.shape[0] == len(target_time), (
            f"The length of the concatenated goal and `target_time` should be the same. "
            f"Got {concat_vecs.shape[0]} and {len(target_time)}."
        )

        for tt, vec in zip(target_time, concat_vecs):
            if tt < interpolation_garbage_collection_time:
                continue
            self.interp = self.interp.schedule_waypoint(
                pose=vec,
                time=tt,
                max_change_rate=self.max_change_rate,
                interpolation_garbage_collection_time=interpolation_garbage_collection_time,
                last_waypoint_time=self.last_waypoint_time,
            )
            self.last_waypoint_time = tt

    def get_action(self, time: Optional[float] = None) -> dict[str, Any]:
        """Get the next action based on the (current) monotonic time."""
        if time is None:
            time = time_module.monotonic()
        concat_vec = self.interp(time)
        self.last_action.update(self._unconcat_vecs(concat_vec))
        return self.last_action

    def observation_space(self) -> gym.spaces.Dict:
        """Return the observation space."""
        pass

    def action_space(self) -> gym.spaces.Dict:
        """Return the action space."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)

        times = np.asarray(times)
        poses = np.asarray(poses)

        self.num_joint = len(poses[0])

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.pose_interp = si.interp1d(times, poses, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pose_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            return self.pose_interp.y

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def schedule_waypoint(
        self,
        pose,
        time,
        max_change_rate=np.inf,
        interpolation_garbage_collection_time=None,
        last_waypoint_time=None,
    ) -> "PoseTrajectoryInterpolator":
        if not isinstance(max_change_rate, np.ndarray):
            max_change_rate = np.array([max_change_rate] * self.num_joint)

        assert len(max_change_rate) == self.num_joint
        assert np.max(max_change_rate) > 0

        if last_waypoint_time is not None:
            assert interpolation_garbage_collection_time is not None

        # trim current interpolator to between interpolation_garbage_collection_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time
        if interpolation_garbage_collection_time is not None:
            if time <= interpolation_garbage_collection_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, interpolation_garbage_collection_time < time
            start_time = max(interpolation_garbage_collection_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = interpolation_garbage_collection_time
                else:
                    end_time = max(last_waypoint_time, interpolation_garbage_collection_time)
            else:
                end_time = interpolation_garbage_collection_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # interpolation_garbage_collection_time <= start_time (proven by zhenjia)
        # interpolation_garbage_collection_time <= time (proven by zhenjia)

        # time can't change
        # last_waypoint_time can't change
        # interpolation_garbage_collection_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == interpolation_garbage_collection_time
            else:
                assert end_time == max(last_waypoint_time, interpolation_garbage_collection_time)

        if interpolation_garbage_collection_time is not None:
            assert interpolation_garbage_collection_time <= start_time
            assert interpolation_garbage_collection_time <= time
        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pose_min_duration = np.max(np.abs(end_pose - pose) / max_change_rate)
        duration = max(duration, pose_min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        pose = np.zeros((len(t), self.num_joint))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            pose = self.pose_interp(t)

        if is_single:
            pose = pose[0]
        return pose
