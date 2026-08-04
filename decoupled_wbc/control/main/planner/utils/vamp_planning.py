import numpy as np
import time

import vamp
from vamp import Sphere, Cuboid, Cylinder
import ompl.geometric as og
import ompl.base as ob

from decoupled_wbc.control.main.planner.utils.ompl_planning import RightGoal
from decoupled_wbc.control.main.planner.utils.ompl_planning import SimilarityObjective, RefStateSampler


class VampStateSpace(ob.RealVectorStateSpace):
    def __init__(self, robot: vamp.robot):
        super().__init__(robot.dimension())
        self.robot = robot
        self.dimension = robot.dimension()
        bounds = ob.RealVectorBounds(self.dimension)
        upper_bounds = robot.upper_bounds()
        lower_bounds = robot.lower_bounds()

        for i in range(self.dimension):
            bounds.setLow(i, lower_bounds[i])
            bounds.setHigh(i, upper_bounds[i])
        self.setBounds(bounds)


class VampStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self, si: ob.SpaceInformation, env: vamp.Environment, robot: vamp.robot
    ):
        super().__init__(si)
        self.env = env
        self.robot = robot
        self.dimension = robot.dimension()

    def isValid(self, state: ob.State) -> bool:
        config = state[0 : self.dimension]
        return self.robot.validate(config, self.env)


class VampMotionValidator(ob.MotionValidator):
    def __init__(
        self, si: ob.SpaceInformation, env: vamp.Environment, robot: vamp.robot
    ):
        super().__init__(si)
        self.env = env
        self.robot = robot
        self.dimension = robot.dimension()

    def checkMotion(self, s1: ob.State, s2: ob.State) -> bool:
        config1 = s1[0 : self.dimension]
        config2 = s2[0 : self.dimension]
        return self.robot.validate_motion(config1, config2, self.env)


class OMPLVAMPPlanner:
    def __init__(
        self,
        env_obj: dict[str, dict],
        robot: str = "g1",
        planner: str = "AORRTC",
        extend_range: float | None = None,
        **kwargs,
    ):
        if robot not in vamp.robots:
            raise RuntimeError(f"Robot {robot} does not exist in VAMP!")
        if robot == "g1_up":
            robot = vamp.g1_up
        else:
            raise RuntimeError(f"Robot {robot} not implemented yet!")
        self.n_dof = robot.dimension()

        # Set up environment
        self.env = self.set_up_env(env_obj)

        # Create the state space from the VAMP robot
        space = VampStateSpace(robot=robot)
        self.si = ob.SpaceInformation(space)

        # Set VAMP-based validators
        motion_validator = VampMotionValidator(self.si, self.env, robot)
        state_validity_checker = VampStateValidityChecker(
            self.si, self.env, robot
        )
        self.si.setMotionValidator(motion_validator)
        self.si.setStateValidityChecker(state_validity_checker)

        # Choose a planner
        self.planner = getattr(og, planner)(self.si)
        if extend_range is not None:
            self.planner.setRange(extend_range)

        # TODO
        # use regular sampler for AORRTC when using similarity objective
        # if planner == "AORRTC":
        #     self.planner.setSimplifySolutions(False)

        # Build SimpleSetup
        self.ss = og.SimpleSetup(self.si)
        self.ss.setPlanner(self.planner)

    def set_up_env(self, env_obj: dict[str, dict]):
        env = vamp.Environment()
        for obj_type, obj in env_obj.items():
            # Cuboid
            if obj_type == "cuboid":
                cuboid = Cuboid(
                    obj["position"],
                    obj["orientation_euler_xyz"],
                    obj["half_extents"],
                )
                cuboid.name = obj["name"]
                env.add_cuboid(cuboid)

            # Cylinder
            elif obj_type == "cylinder":
                cylinder = Cylinder(
                    obj["position"],
                    obj["orientation_euler_xyz"],
                    obj["radius"],
                    obj["length"],
                )
                cylinder.name = obj["name"]
                env.add_capsule(cylinder)

            # Sphere
            elif obj_type == "sphere":
                sphere = Sphere(obj["position"], obj["radius"])
                sphere.name = obj["name"]
                env.add_sphere(sphere)

        return env

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        goal_type: str = "both",  # "left" | "right" | "both"
        ref_traj: np.ndarray | None = None,
        ref_weights: np.ndarray | None = None,
        timeout: float = 1.0,
        smooth_path: bool = True,
        shortcut_path: bool = True,
    ) -> np.ndarray:
        """Plan a path from start to goal"""
        # Convert start to OMPL states
        start_state = self.si.allocState()
        for i in range(self.n_dof):
            start_state[i] = float(start[i])
        self.ss.setStartState(start_state)

        # Define goal
        if goal_type == "both":
            goal_state = self.si.allocState()
            for i in range(self.n_dof):
                goal_state[i] = float(goal[i])
            self.ss.setGoalState(goal_state)
        elif goal_type == "left":
            raise ValueError("Left goal is not implemented yet")
        elif goal_type == "right":
            self.ss.setGoal(RightGoal(self.si, goal))
        else:
            raise ValueError(f"Invalid goal type: {goal_type}")

        # Define optimization objective
        if ref_traj is not None:
            # Sampler
            print("Using reference sampler")
            self.ss.getStateSpace().setStateSamplerAllocator(
                lambda space: RefStateSampler(space, ref_traj)
            )
            objective = SimilarityObjective(self.si, ref_traj, ref_weights)
            self.ss.setOptimizationObjective(objective)

        # Set up the planner
        self.ss.setup()

        # Solve
        waypoints = np.array([start])
        # print(float(timeout))
        status = self.ss.solve(float(timeout))
        if status.asString() == "Exact solution":
            path = self.ss.getSolutionPath()
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

        # TEMP
        # self.ss.clear()
        return waypoints


class VAMPPlanner:
    """
    VAMP Geometric Planner class for planning paths.
    Using VAMP model for planning.
    """

    def __init__(
        self,
        env_obj: dict[str, dict],
        robot: str = "g1",
        planner: str = "aorrtc",
        sampler: str = "xorshift",
        **kwargs,
    ):
        if robot not in vamp.robots:
            raise RuntimeError(f"Robot {robot} does not exist in VAMP!")
        # if planner not in vamp.planners:
        #     raise RuntimeError(f"Planner {planner} does not exist in VAMP!")
        # if sampler not in vamp.samplers:
        #     raise RuntimeError(f"Sampler {sampler} does not exist in VAMP!")

        """Initialize Planner"""
        # Set up VAMP planner
        (vamp_module, planner_func, plan_settings, simp_settings) = (
            vamp.configure_robot_and_planner_with_kwargs(
                robot,
                planner,
                **kwargs,
            )
        )
        self.vamp_module = vamp_module
        self.planner_func = planner_func
        self.plan_settings = plan_settings
        self.simp_settings = simp_settings
        self.sampler = getattr(vamp_module, sampler)()

        # Set up environment
        self.env = self.set_up_env(env_obj)

    def set_up_env(self, env_obj: dict[str, dict]):
        env = vamp.Environment()
        for obj_type, obj in env_obj.items():
            # Cuboid
            if obj_type == "cuboid":
                cuboid = Cuboid(
                    obj["position"],
                    obj["orientation_euler_xyz"],
                    obj["half_extents"],
                )
                cuboid.name = obj["name"]
                env.add_cuboid(cuboid)

            # Cylinder
            elif obj_type == "cylinder":
                cylinder = Cylinder(
                    obj["position"],
                    obj["orientation_euler_xyz"],
                    obj["radius"],
                    obj["length"],
                )
                cylinder.name = obj["name"]
                env.add_capsule(cylinder)

            # Sphere
            elif obj_type == "sphere":
                sphere = Sphere(obj["position"], obj["radius"])
                sphere.name = obj["name"]
                env.add_sphere(sphere)

        return env

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        # goal_type: str = "both",  # "left" | "right" | "both"
        max_iteration: int = 1000000,
        smooth_path: bool = True,
    ):
        """Plan a path from start to goal"""
        # Set up planner settings
        self.plan_settings.max_iterations = int(max_iteration)
        self.plan_settings.max_samples = int(max_iteration)

        # Check validity of start and goal
        if not self.vamp_module.validate(start, self.env):
            raise RuntimeError("Start is not valid!")
        if not self.vamp_module.validate(goal, self.env):
            raise RuntimeError("Goal is not valid!")

        # Solve
        t1 = time.perf_counter()
        waypoints = np.array([start])
        result = self.planner_func(
            start, goal, self.env, self.plan_settings, self.sampler
        )
        t2 = time.perf_counter()
        print(f"Planning Time: {t2 - t1} seconds")

        simplify = None
        if result.solved:
            path = result.path
            if smooth_path:
                simplify = self.vamp_module.simplify(
                    result.path, self.env, self.simp_settings, self.sampler
                )
                path = simplify.path
                path.interpolate_to_resolution(self.vamp_module.resolution())
            waypoints = path.numpy()

        stats = vamp.results_to_dict(result, simplify)
        print(
            f"""
            Planning Time: {stats['planning_time'].microseconds:8d}μs
            Simplify Time: {stats['simplification_time'].microseconds:8d}μs
            Total Time: {stats['total_time'].microseconds:8d}μs
            Planning Iters: {stats['planning_iterations']}
            n Graph States: {stats['planning_graph_size']}
            Path Length:
            Initial: {stats['initial_path_cost']:5.3f}
            Simplified: {stats['simplified_path_cost']:5.3f}"""
        )

        return waypoints
