from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import numpy as np
from robosuite.environments import MujocoEnv
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils.placement_samplers import (
    ObjectPositionSampler,
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robocasa.utils.scene.configs import (
    ObjectConfig,
    SceneConfig,
    SceneScaleConfig,
)


class SceneObject:
    def __init__(self, cfg: ObjectConfig):
        self.cfg = cfg
        self._mujoco_object = self._get_object()

    @property
    def mj_obj(self) -> MJCFObject:
        return self._mujoco_object

    def get_sampler_with_args(
        self, env: MujocoEnv, scene_scale: SceneScaleConfig
    ) -> tuple[ObjectPositionSampler, dict]:
        return (
            self._get_sampler(self._mujoco_object, env, scene_scale),
            self.cfg.sampler_config.sampling_args,
        )

    def _get_object(self) -> MJCFObject:
        abs_mjcf_path = (
            self.cfg.mjcf_path
            if Path(self.cfg.mjcf_path).is_absolute()
            else xml_path_completion(self.cfg.mjcf_path, root=robocasa.models.assets_root)
        )
        return MJCFObject(
            name=self.cfg.name,
            mjcf_path=abs_mjcf_path,
            scale=self.cfg.scale,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=self.cfg.density,
            friction=self.cfg.friction,
            margin=self.cfg.margin,
            static=self.cfg.static,
            rgba=self.cfg.rgba,
        )

    def _get_sampler(
        self, obj: MJCFObject, env: MujocoEnv, scene_scale: SceneScaleConfig
    ) -> ObjectPositionSampler:
        x_range, y_range = scene_scale.get_ranges(
            self.cfg.sampler_config.x_range, self.cfg.sampler_config.y_range
        )
        z_offset = scene_scale.get_z_offset(self.cfg.sampler_config.z_offset)
        rotation = scene_scale.get_rotation(self.cfg.sampler_config.rotation)
        reference_pos = scene_scale.get_ref_pos(self.cfg.sampler_config.reference_pos)
        return UniformRandomSampler(
            name=f"{self.cfg.name}_sampler",
            mujoco_objects=obj,
            x_range=x_range,
            y_range=y_range,
            rotation=rotation,
            rng=env.rng,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )


class Scene:
    def __init__(self, env: MujocoEnv, config: SceneConfig, scene_scale: SceneScaleConfig):
        self._env = env
        self._config = config
        self._scene_scale = scene_scale
        self._scene_objects: dict[MJCFObject, SceneObject] = {}
        for scene_obj in self._config.objects:
            self._scene_objects[scene_obj.mj_obj] = scene_obj
        self._scene_sampler = self._get_sampler()

    @property
    def mujoco_objects(self) -> list[MJCFObject]:
        return list(self._scene_objects.keys())

    @property
    def instruction(self) -> str:
        return self._config.instruction

    def reset(self) -> None:
        sim = self._env.sim
        object_placements = self._scene_sampler.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            scene_obj = self._scene_objects[obj]
            if scene_obj.cfg.static:
                body_id = sim.model.body_name2id(obj.root_body)
                sim.model.body_pos[body_id] = obj_pos
                sim.model.body_quat[body_id] = obj_quat
                obj.set_pos(obj_pos)
                obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))
            else:
                joint_name = next(j for j in obj.joints if np.size(sim.data.get_joint_qpos(j)) == 7)
                qpos = np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                # Set qpos0 to support robust mujoco environment reset
                start_i, end_i = sim.model.get_joint_qpos_addr(joint_name)
                sim.data.model.qpos0[start_i:end_i] = qpos
                # Set qpos
                sim.data.set_joint_qpos(
                    joint_name,
                    qpos,
                )

    def success(self) -> bool:
        return self._config.success.is_true(self._env)

    def _get_sampler(self) -> SequentialCompositeSampler:
        sorted_objects = self._sort_scene_objects(list(self._scene_objects.values()))

        scene_sampler = SequentialCompositeSampler(name="SceneSampler", rng=self._env.rng)
        for obj in sorted_objects:
            sampler, args = obj.get_sampler_with_args(self._env, self._scene_scale)
            scene_sampler.append_sampler(sampler=sampler, sample_args=args)

        return scene_sampler

    @staticmethod
    def _sort_scene_objects(scene_objects: list[SceneObject]) -> list[SceneObject]:
        ts = TopologicalSorter()
        for obj in scene_objects:
            ref = obj.cfg.sampler_config.reference.obj if obj.cfg.sampler_config.reference else None
            ts.add(obj, *([ref] if ref else []))
        try:
            return list(ts.static_order())
        except CycleError:
            raise ValueError("Circular reference detected")
