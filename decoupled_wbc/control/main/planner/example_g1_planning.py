import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco

from decoupled_wbc.control.main.planner.simulation.robot import G1Up
from decoupled_wbc.control.main.planner.simulation.mujoco_utils import render_mp4, sample_qpos
from decoupled_wbc.control.main.planner.utils.ompl_planning import OMPLGeometricPlanner, SimilarityObjective
from decoupled_wbc.control.main.planner.utils.vamp_planning import VAMPPlanner, OMPLVAMPPlanner
from decoupled_wbc.control.main.planner.geometry.trajectory import TOPPRATrajectory


if __name__ == "__main__":
    np.random.seed(200)

    # Setup
    root = os.path.dirname(os.path.abspath(__file__))
    # xml = open(os.path.join(root, "simulation/g1_up.xml")).read()
    # model = mujoco.MjModel.from_xml_string(xml)
    model = mujoco.MjModel.from_xml_path(str(os.path.join(root, "simulation", "g1_up.xml")))
    robot = G1Up(model=model, visualize=True)

    # Planner
    # planner = OMPLGeometricPlanner(robot, planner="AORRTC")
    env_obj = {
        "cuboid": {
            "name": "lab_scene",
            "position": (0.508 + 0.4, 0, -0.36 - 0.073),
            "orientation_euler_xyz": (0, 0, 0),
            "half_extents": (0.762, 0.762, 0.36),
        },
    }

    # planner1 = OMPLGeometricPlanner(robot, planner="RRTstar")
    planner1 = OMPLVAMPPlanner(
        env_obj,
        "g1_up",
        planner="AORRTC",
        # planner="BITstar",
        # planner="RRTstar",
    )
    # planner2 = VAMPPlanner(env_obj, "g1_up", planner="aorrtc")

    # Current as start
    start = robot.get_joint_qpos()
    # Desired goal posture
    goal = start.copy()
    goal[np.arange(0, 3)] = (-0.1, 0, 0.2)
    goal[np.arange(3, 10)] = (-1.5, 1.0, 0.7, -1.0, 0, 0, 0)
    goal[np.arange(10, 17)] = (-0.3, -0.2, 0.3, 0.3, 0, 0, 0.5)

    # Sim reset
    robot.set_joint_qpos(start)
    robot.viewer.sync()

    # Benchmark
    # sample
    start_configs = []
    goal_configs = []
    import vamp
    from vamp import Cuboid

    vamp_robot = vamp.g1_up
    vamp_env = vamp.Environment()
    cuboid = Cuboid(
        (0.508 + 0.4, 0, -0.36 - 0.073), (0, 0, 0), (0.762, 0.762, 0.36)
    )
    cuboid.name = "lab_scene"
    vamp_env.add_cuboid(cuboid)
    while len(start_configs) < 1000 or len(goal_configs) < 1000:
        config = robot.sample_qpos()
        robot.set_joint_qpos(config)
        if (
            not robot.in_contact()
            and robot.in_limits(config)
            and vamp_robot.validate(config, vamp_env)
        ):
            if len(start_configs) < 1000:
                start_configs.append(config)
            else:
                goal_configs.append(config)

    # # solve
    # print(f"Start benchmarking")
    # t1 = time.time()
    # for i in range(1000):
    #     solution = planner0.plan(
    #         start_configs[i], goal_configs[i], "both", timeout=1.0
    #     )
    # t2 = time.time()
    # t_ompl = t2 - t1
    # t1 = time.time()
    # for i in range(1000):
    #     solution = planner1.plan(
    #         start_configs[i], goal_configs[i], "both", timeout=1.0
    #     )
    # t2 = time.time()
    # t_ompl_vamp = t2 - t1
    # t1 = time.time()
    # for i in range(1000):
    #     solution = planner2.plan(
    #         start_configs[i], goal_configs[i], max_iteration=1e5
    #     )
    # t2 = time.time()
    # t_vamp = t2 - t1
    # print()
    # print(f"OMPL geometric planner time: \n{t_ompl} microseconds\n")
    # print(f"OMPL-VAMP planner time: \n{t_ompl_vamp} microseconds\n")
    # print(f"VAMP planner time: \n{t_vamp} microseconds\n")

    # Solve
    root = os.path.dirname(os.path.abspath(__file__))
    ref = np.load(os.path.join(root, "dataset", "reference_col.npy"))
    ref_weights = np.linspace(2.0, 1.0, 10)
    ref_weights = np.concatenate([ref_weights, ref_weights[3:]])
    solution = planner1.plan(
        start, goal, "both", ref, ref_weights, timeout=1.0, shortcut_path=False
    )
    # solution = planner2.plan(start, goal, max_iteration=1e6)
    # solution = planner1.plan(
    #     start, goal, "both", ref, ref_weights, timeout=10.0
    # )

    # solution = planner0.plan(start, goal, "both", timeout=1.0)
    # solution = planner.plan(start, goal, max_iteration=5e4)
    if len(solution) < 2:
        robot.close()
        raise RuntimeError("No solution found!")

    # Check cost
    opt = SimilarityObjective(planner1.si, ref, ref_weights)
    cost = 0.0
    path = planner1.ss.getSolutionPath()
    states = path.getStates()
    for i in range(len(states) - 1):
        cost += opt.motionCost(states[i], states[i + 1]).value()
    print(f"Cost: {cost}")
    planner1.ss.clear()

    # Visualize the waypoints
    try:
        input("Press Enter to start")
        trajectory = TOPPRATrajectory(solution, 2.0, 2.0)
        waypoints = trajectory.to_step_waypoints(model.opt.timestep)
        robot.set_joint_qpos(waypoints[0])
        for waypoint in waypoints:
            robot.set_joint_qpos(waypoint)
            # robot.ctrl_joint_qpos(waypoint)
            robot.viewer.sync()
    finally:
        robot.close()

    # Record Video
    qpos_sequence = [robot.data.qpos.copy() for waypoint in waypoints]
    for i in range(len(waypoints)):
        qpos_sequence[i][robot.joint_ids] = waypoints[i]
    # render_mp4(
    #     model, robot.data, qpos_sequence, model.opt.timestep, "test.mp4"
    # )
