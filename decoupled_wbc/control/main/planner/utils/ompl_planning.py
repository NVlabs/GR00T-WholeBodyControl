import numpy as np
import mujoco
import time
from scipy.spatial import cKDTree
from scipy.interpolate import make_interp_spline

import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou

from simulation.robot import MujocoRobot


class OMPLGeometricPlanner:
    """
    OMPL Geometric Planner class for planning paths.
    Using Mujoco Robot with itsmodel and data for planning.

    By default, using RealVectorStateSpace and PathLengthOptimizationObjective.
    """

    def __init__(
        self,
        robot: MujocoRobot,
        data: mujoco.MjData | None = None,
        planner: str = "RRTConnect",
        validity_resolution: float = 0.01,
        extend_range: float | None = None,
        log: bool = True,
    ):
        """Initialize Planner"""
        # Mujoco Robot with its model and data
        self.robot = robot
        self.model = robot.model
        # create a new data for this planning instead of
        # using the robot instance's data
        if data is None:
            self.data = mujoco.MjData(self.model)
            self.data.qpos[:] = robot.data.qpos[:]
        else:
            self.data = data

        self.robot_geoms = self.robot.robot_geoms
        self.n_dof = self.robot.n_joints
        self.joint_limits = self.robot.joint_limits

        # Set up OMPL planner
        self.planner_name = planner
        self.validity_resolution = validity_resolution
        self.ss, self.si = self.set_up_ompl()
        self.pdef = self.ss.getProblemDefinition()
        self.planner = self.ss.getPlanner()
        if extend_range is not None:
            self.planner.setRange(extend_range)

        if not log:
            ou.setLogLevel(ou.LOG_ERROR)

    def set_up_ompl(self):
        """Setup OMPL planner"""
        # Define space
        space = ob.RealVectorStateSpace(self.n_dof)
        bounds = ob.RealVectorBounds(self.n_dof)
        for i in range(self.n_dof):
            # in case limit is infinite, set to -2pi, 2pi
            low = self.joint_limits[0][i]
            high = self.joint_limits[1][i]
            if low == -np.inf:
                low = -2 * np.pi
            if high == np.inf:
                high = 2 * np.pi
            bounds.setLow(i, low)
            bounds.setHigh(i, high)
        space.setBounds(bounds)

        # Simple Setup
        ss = og.SimpleSetup(space)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(self.validity_resolution)
        ss.setStateValidityChecker(self.validity_checker)
        ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

        # Set planner
        planner = getattr(og, self.planner_name)(si)
        # planner.setSimplifySolutions(False)
        ss.setPlanner(planner)
        return ss, si

    def validity_checker(self, state: ob.State):
        """Check if the state is valid

        By default, check if they are in bounds and if they are collision free
        """
        q = np.array([state[i] for i in range(self.n_dof)], dtype=float)
        self.robot.set_joint_qpos(q)

        # TEMPORARILY DISABLED FOR SIMULATION TESTING ONLY.
        # Restore this before real-robot deployment; otherwise OMPL may return
        # paths that collide with the robot or the environment.
        # in_contact = self.robot.in_contact()
        in_contact = False
        # Check if in bounds
        in_bounds = self.si.satisfiesBounds(state)
        return in_bounds and not in_contact

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        goal_type: str = "upper_body",  # "upper_body", "bimanual", "left", "right"
        ref_traj: np.ndarray | None = None,
        ref_weights: np.ndarray | None = None,
        timeout: float = 10.0,
        smooth_path: bool = True,
        shortcut_path: bool = True,
    ) -> np.ndarray:
        """Plan a path from start to goal"""
        # Convert start and goal to OMPL states
        start_state = self.si.allocState()
        for i in range(self.n_dof):
            start_state[i] = float(start[i])
        self.ss.setStartState(start_state)

        # goal_state = self.si.allocState()
        # goal_state[i] = float(goal[i])
        # self.ss.setGoalState(goal_state)
        goal_state = self.si.allocState()
        if goal_type == "upper_body":
            for i in range(self.n_dof):
                goal_state[i] = float(goal[i])
            self.ss.setGoalState(goal_state)
        elif goal_type == "bimanual":
            raise ValueError("Bimanual goal is not implemented yet")
        elif goal_type == "left":
            raise ValueError("Left goal is not implemented yet")
        elif goal_type == "right":
            self.ss.setGoal(RightGoal(self.si, goal))
        else:
            raise ValueError(f"Invalid goal type: {goal_type}")

        # Define optimization objective
        if ref_traj is not None:
            # Sampler
            self.ss.getStateSpace().setStateSamplerAllocator(
                lambda space: RefStateSampler(space, ref_traj)
            )
            # Cost
            objective = SimilarityObjective(self.si, ref_traj, ref_weights)
            self.pdef.setOptimizationObjective(objective)

        # Set up the planner
        self.ss.setup()

        # Solve
        waypoints = np.array([start])
        status = self.ss.solve(float(timeout))
        if status.asString() == "Exact solution":
            path = self.ss.getSolutionPath()
            objective = self.pdef.getOptimizationObjective()
            if smooth_path:
                ps = og.PathSimplifier(self.si)
                if shortcut_path:
                    try:
                        ps.ropeShortcutPath(path)
                    except Exception:
                        ps.shortcutPath(path)
                ps.smoothBSpline(path)
            states = path.getStates()
            waypoints = np.array(
                [[s[i] for i in range(self.n_dof)] for s in states]
            )

        self.ss.clear()
        return waypoints


class RefStateSampler(ob.StateSampler):
    def __init__(
        self,
        space,
        reference_waypoints,
        p_uniform=0.1,
        spatial_sigma=0.3,
        progress_period=1000,
        progress_sigma=0.1,
        p_progress=0.5,
        seed=None,
    ):
        super().__init__(space)

        self.space = space
        self.dim = space.getDimension()
        self.bounds = space.getBounds()

        self.p_uniform = p_uniform
        self.spatial_sigma = spatial_sigma
        self.progress_period = progress_period
        self.progress_sigma = progress_sigma
        self.p_progress = p_progress

        self.rng = np.random.default_rng(seed)
        self.sample_count = 0

        self._build_reference_model(reference_waypoints)

    def _build_reference_model(self, reference_waypoints):
        wp = np.asarray(reference_waypoints, dtype=float)

        if wp.ndim != 2:
            raise ValueError("reference_waypoints should have shape (N, dim).")
        if wp.shape[1] != self.dim:
            raise ValueError(
                f"Waypoint dimension {wp.shape[1]} does not match state dimension {self.dim}."
            )
        if wp.shape[0] < 2:
            raise ValueError("Need at least two reference waypoints.")

        # Remove consecutive duplicate waypoints.
        diff = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        keep = np.r_[True, diff > 1e-12]
        wp = wp[keep]

        if wp.shape[0] < 2:
            raise ValueError("Reference trajectory is degenerate.")

        # Arc-length parameterization.
        seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        u = np.r_[0.0, np.cumsum(seg_len)]
        u = u / u[-1]

        self.wp = wp
        self.u = u

        # B-spline if scipy exists; otherwise use linear interpolation.
        if make_interp_spline is not None and wp.shape[0] >= 4:
            k = min(3, wp.shape[0] - 1)
            self.spline = make_interp_spline(u, wp, k=k, axis=0)
        else:
            self.spline = None

    def _reference_at(self, t):
        t = float(np.clip(t, 0.0, 1.0))

        if self.spline is not None:
            return np.asarray(self.spline(t), dtype=float)

        # Fallback: linear interpolation dimension by dimension.
        return np.array(
            [np.interp(t, self.u, self.wp[:, j]) for j in range(self.dim)],
            dtype=float,
        )

    def _write_vector_to_state(self, state, x):
        for i in range(self.dim):
            lo = self.bounds.low[i]
            hi = self.bounds.high[i]
            value = float(np.clip(x[i], lo, hi))

            # Most OMPL Python RealVector states support this form.
            # If your binding complains, try: state.values[i] = value
            state[i] = value

    def _sample_uniform_realvector(self, state):
        x = np.empty(self.dim)
        for i in range(self.dim):
            x[i] = self.rng.uniform(self.bounds.low[i], self.bounds.high[i])
        self._write_vector_to_state(state, x)

    def _sample_reference_biased(self, state):
        # Either progress-biased t or globally random t.
        if self.rng.random() < self.p_progress:
            phase = (self.sample_count % self.progress_period) / float(
                self.progress_period
            )
            t = self.rng.normal(loc=phase, scale=self.progress_sigma)
            t = np.clip(t, 0.0, 1.0)
        else:
            t = self.rng.uniform(0.0, 1.0)

        x_ref = self._reference_at(t)

        # Spatial Gaussian around the spline point.
        noise = self.rng.normal(
            loc=0.0, scale=self.spatial_sigma, size=self.dim
        )
        x = x_ref + noise

        self._write_vector_to_state(state, x)

    def sampleUniform(self, state):
        self.sample_count += 1
        # print(self.sample_count)

        if self.rng.random() < self.p_uniform:
            self._sample_uniform_realvector(state)
        else:
            self._sample_reference_biased(state)

    def sampleUniformNear(self, state, near, distance):
        # Keep the default behavior semantically:
        # sample each dimension within [near[i] - distance, near[i] + distance],
        # truncated by bounds.
        x = np.empty(self.dim)
        for i in range(self.dim):
            lo = max(self.bounds.low[i], near[i] - distance)
            hi = min(self.bounds.high[i], near[i] + distance)
            x[i] = self.rng.uniform(lo, hi)
        self._write_vector_to_state(state, x)

    def sampleGaussian(self, state, mean, stdDev):
        x = np.empty(self.dim)
        for i in range(self.dim):
            x[i] = self.rng.normal(loc=mean[i], scale=stdDev)
        self._write_vector_to_state(state, x)


class StateCostIntegralObjective(ob.OptimizationObjective):
    """Original State Cost Integral Objective implementation in python"""

    def __init__(self, si, enable_motion_cost_interpolation):
        super().__init__(si)
        self.si = si
        self.ss = si.getStateSpace()
        self.interpolation = enable_motion_cost_interpolation

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        if self.interpolation:
            cost = self.identityCost()
            nd = self.ss.validSegmentCount(s1, s2)

            temp1 = self.si.cloneState(s1)
            temp2 = self.si.allocState()

            prev_cost = self.stateCost(temp1)
            for j in range(1, nd + 1):
                if j < nd:
                    t = float(j) / float(nd)
                    self.ss.interpolate(s1, s2, t, temp2)
                    curr = temp2
                else:
                    curr = s2

                curr_cost = self.stateCost(curr)
                seg_cost = self.trapezoid(
                    prev_cost, curr_cost, self.si.distance(temp1, curr)
                )
                cost = self.combineCosts(cost, seg_cost)

                if j < nd:
                    self.si.copyState(temp1, temp2)
                prev_cost = curr_cost

        # No interpolation
        else:
            cost = self.trapezoid(
                self.stateCost(s1),
                self.stateCost(s2),
                self.si.distance(s1, s2),
            )
        return cost

    def motionCostBestEstimate(self, s1, s2):
        return self.trapezoid(
            self.stateCost(s1), self.stateCost(s2), self.si.distance(s1, s2)
        )

    @staticmethod
    def trapezoid(c1, c2, dist):
        return ob.Cost(0.5 * dist * (c1.value() + c2.value()))

    def isMotionCostInterpolationEnabled(self):
        return self.interpolation


# class SimilarityObjective(ob.StateCostIntegralObjective):
class SimilarityObjective(StateCostIntegralObjective):
    def __init__(
        self,
        si: ob.SpaceInformation,
        ref_traj: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        super().__init__(si, True)
        self.si = si
        self.n_dof = self.si.getStateSpace().getDimension()

        self.ref_traj = np.asarray(ref_traj)
        if weights is None:
            self.weights = np.ones(self.n_dof)
        else:
            self.weights = np.asarray(weights)

        self.weighted_ref_traj = self.ref_traj * self.weights

        # self.nn = cKDTree(self.weighted_ref_traj)

    def stateCost(self, state):
        config = np.asarray(state[0 : self.n_dof], dtype=float)
        weighted_config = config * self.weights

        # KDTree
        # dist, k = self.nn.query(weighted_config, k=1)
        # return ob.Cost(dist)

        # Squared Euclidean distances
        diff = self.weighted_ref_traj - weighted_config
        dists_sq = np.sum(diff * diff, axis=1)
        min_dist = np.sqrt(np.min(dists_sq))

        return ob.Cost(float(min_dist))


class RightGoal(ob.GoalSampleableRegion):
    GOAL_IDXS = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16]

    def __init__(
        self,
        si: ob.SpaceInformation,
        right_goal: np.ndarray,
        max_goal_samples: int = 1000000,
        threshold: float = 1e-3,
        require_valid: bool = True,
    ):
        super().__init__(si)
        self.si = si
        self.ss = si.getStateSpace()
        self.goal_sampler = si.allocStateSampler()
        self.max_goal_samples = int(max_goal_samples)

        right_goal = np.asarray(right_goal, dtype=float)
        if right_goal.shape[0] == 17:
            right_goal = right_goal[self.GOAL_IDXS]
        self.right_goal = right_goal

        self.require_valid = require_valid
        self.setThreshold(threshold)

    def sampleGoal(self, state: ob.State) -> None:
        # sample a full random state first
        while True:
            self.goal_sampler.sampleUniform(state)
            for goal_i, idx in enumerate(self.GOAL_IDXS):
                state[idx] = float(self.right_goal[goal_i])

            # keep the sampled state inside bounds
            self.ss.enforceBounds(state)

            if not self.require_valid or self.si.isValid(state):
                return

    def maxSampleCount(self) -> int:
        return self.max_goal_samples

    def distanceGoal(self, state: ob.State) -> float:
        dist2 = 0.0
        for goal_i, idx in enumerate(self.GOAL_IDXS):
            d = float(state[idx]) - float(self.right_goal[goal_i])
            dist2 += d * d
        return dist2**0.5
