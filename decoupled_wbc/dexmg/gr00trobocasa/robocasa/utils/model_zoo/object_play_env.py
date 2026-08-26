import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from robocasa.utils.model_zoo.mjcf_obj import MJCFObject
from robosuite.models.objects import BoxObject

import robosuite.utils.transform_utils as T


class ObjectPlayEnv(SingleArmEnv):
    def __init__(
        self,
        robots,
        obj_mjcf_path,
        obj_scale=1.0,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        x_range=(-0.03, 0.03),
        y_range=(-0.03, 0.03),
        rotation=None,
        num_objects=1,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self._obj_mjcf_path = obj_mjcf_path
        self._obj_scale = obj_scale
        self._x_range = x_range
        self._y_range = y_range
        self._rotation = rotation
        self._num_objects = num_objects

        self._cam_configs = dict(
            agentview=dict(
                pos=[0, -0.5, 1.35],
                quat=T.convert_quat(
                    np.array([0.3826834, 0, 0, 0.9238795]),
                    to="wxyz",
                ),
            ),
        )

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        reward = 0.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        # xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        # self.robots[0].robot_model.set_base_xpos(xpos)
        ### logic copied from kitchen.py to rotate robot around ###
        for robot, rotation, offset in zip(self.robots, (-np.pi / 2,), (0,)):
            xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
            rot = np.array((0, 0, rotation))
            xpos = T.euler2mat(rot) @ np.array(xpos)
            xpos += np.array((0, offset, 0))
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(rot)

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        self.set_cameras()

        objects = []
        for obj_num in range(self._num_objects):
            if self._obj_mjcf_path == "cube":
                # debugging case using primitive geom
                obj = BoxObject(
                    name="cube_{}".format(obj_num),
                    size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
                    size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
                    rgba=[1, 0, 0, 1],
                )
            else:
                obj = MJCFObject(
                    name="MCJFObj_{}".format(obj_num),
                    mjcf_path=self._obj_mjcf_path,
                    scale=self._obj_scale,
                )

            objects.append(obj)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=objects,
                x_range=self._x_range,
                y_range=self._y_range,
                rotation=self._rotation,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                # # for debugging
                # x_range=(0, 0),
                # y_range=(0, 0),
                # rotation=(0, 0),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    # def _setup_references(self):
    #     """
    #     Sets up references to important components. A reference is typically an
    #     index or a list of indices that point to the corresponding elements
    #     in a flatten array, which is how MuJoCo stores physical simulation data.
    #     """
    #     super()._setup_references()

    #     # Additional object references from this env
    #     self.obj_body_id = self.sim.model.body_name2id(self.mjcf_obj.root_body)

    def set_cameras(self):
        """
        copied from environments/manipulation/kitchen.py
        """
        for cam_name, cam_cfg in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue

            self.mujoco_arena.set_camera(
                camera_name=cam_name, pos=cam_cfg["pos"], quat=cam_cfg["quat"]
            )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # self.sim.model.body_pos[self.sim.model.body_name2id(obj.root_body)] = obj_pos
                # self.sim.model.body_quat[self.sim.model.body_name2id(obj.root_body)] = obj_quat

                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    # def visualize(self, vis_settings):
    #     """
    #     In addition to super call, visualize gripper site proportional to the distance to the cube.
    #
    #     Args:
    #         vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
    #             component should be visualized. Should have "grippers" keyword as well as any other relevant
    #             options specified.
    #     """
    #     # Run superclass method first
    #     super().visualize(vis_settings=vis_settings)
    #
    #     # Color the gripper visualization site according to its distance to the cube
    #     if vis_settings["grippers"]:
    #         self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        return False
