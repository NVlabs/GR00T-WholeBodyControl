import numpy as np
from scipy.interpolate import CubicSpline


class Trajectory:
    """Abstract base class for trajectory objects"""

    def __init__(self, states: np.ndarray):
        """Initialize with the joint waypoints"""
        self.states = states

    @property
    def dof(self) -> int:
        """The number of degrees of freedom"""
        return self.states.shape[1]

    @property
    def n_states(self) -> int:
        """The number of waypoints"""
        return self.states.shape[0]

    @property
    def end_time(self) -> float:
        """The end time of the trajectory"""
        raise NotImplementedError

    def position(self, time: float) -> np.ndarray:
        """Return the position at the given time"""
        raise NotImplementedError

    def velocity(self, time: float) -> np.ndarray:
        """Return the velocity at the given time"""
        raise NotImplementedError

    def acceleration(self, time: float) -> np.ndarray:
        """Return the acceleration at the given time"""
        raise NotImplementedError

    def to_step_waypoints(self, dt: float, type: str = "p") -> np.ndarray:
        """
        Convert this trajectory to a list of joint waypoints,
        either position, velocity, or acceleration at each given time steps
        """
        # Data holder
        total_run_step = int(np.ceil(self.end_time / dt))
        waypoints = np.zeros((total_run_step, self.dof))

        if type == "p":
            func = self.position
        elif type == "v":
            func = self.velocity
        elif type == "a":
            func = self.acceleration
        else:
            raise ValueError(f"Invalid type: {type}, must be 'p', 'v', or 'a'")

        for k in range(total_run_step):
            curr_t = k * dt
            waypoints[k, :] = func(curr_t)
        return waypoints


class SplineTrajectory(Trajectory):
    """Trajectory class that performs cubic spline interpolation
    Require states and corresponding time t_states to perform interpolation
    """

    def __init__(self, states: np.ndarray, t_states: np.ndarray):
        """Initialize with the joint waypoints and corresponding time"""
        super().__init__(states)
        self.t_states = t_states

        # Perform cubic spline interpolation based on the provided states and t_state
        self.trajectory = CubicSpline(self.t_states, self.states, axis=0)

    @property
    def end_time(self) -> float:
        """The end time of the trajectory"""
        return self.t_states[-1]

    def position(self, time: float) -> np.ndarray:
        """Return the position at the given time"""
        # Ensure time is within bounds
        time = self.time_check(time)
        return self.trajectory(time)

    def velocity(self, time: float) -> np.ndarray:
        """Return the velocity at the given time"""
        # Ensure time is within bounds
        time = self.time_check(time)
        return self.trajectory(time, 1)

    def acceleration(self, time: float) -> np.ndarray:
        """Return the acceleration at the given time"""
        # Ensure time is within bounds
        time = self.time_check(time)
        return self.trajectory(time, 2)

    def time_check(self, time: float) -> bool:
        """Check if the given time is within bounds"""
        if time < 0 or time > self.end_time:
            print(
                f"Time {time} outside bounds [0, {self.end_time}]. "
                + f"Clipping it to bounds."
            )
        time = np.clip(time, 0, self.end_time)
        return time


class TOPPRATrajectory(Trajectory):
    """Trajectory class that performs trajectory optimization using TOPP-RA"""

    def __init__(self, states, default_velocity, default_acceleration):
        """
        Initialize with the joint waypoints,
        default velocity, and default acceleration for TOPP-RA
        """
        import toppra as ta  # lazy import

        super().__init__(states)
        self.default_velocity = default_velocity
        self.default_acceleration = default_acceleration

        path_scalars = np.linspace(0, 1, self.n_states)
        path = ta.SplineInterpolator(path_scalars, self.states)

        vel = np.array([self.default_velocity] * self.dof)
        acc = np.array([self.default_acceleration] * self.dof)

        pc_vel = ta.constraint.JointVelocityConstraint(vel)
        pc_acc = ta.constraint.JointAccelerationConstraint(acc)

        instance = ta.algorithm.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeSpline"
        )
        if (jnt_traj := instance.compute_trajectory()) is not None:
            self.trajectory = jnt_traj
        else:
            raise RuntimeError("Failed to parameterize trajectory")

    @property
    def end_time(self) -> float:
        """The end time of the trajectory"""
        return self.trajectory.path_interval[1]

    def position(self, time: float) -> np.ndarray:
        """Return the position at the given time"""
        time = np.clip(time, 0, self.end_time)
        return self.trajectory(time)

    def velocity(self, time: float) -> np.ndarray:
        """Return the velocity at the given time"""
        time = np.clip(time, 0, self.end_time)
        return self.trajectory(time, 1)

    def acceleration(self, time: float) -> np.ndarray:
        """Return the acceleration at the given time"""
        time = np.clip(time, 0, self.end_time)
        return self.trajectory(time, 2)

    def time_check(self, time: float) -> bool:
        """Check if the given time is within bounds"""
        if time < 0 or time > self.end_time:
            print(
                f"Time {time} outside bounds [0, {self.end_time}]. "
                + f"Clipping it to bounds."
            )
        time = np.clip(time, 0, self.end_time)
        return time
