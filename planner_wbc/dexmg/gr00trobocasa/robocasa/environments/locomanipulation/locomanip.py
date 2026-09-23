from abc import abstractmethod
from typing import Optional

from robocasa.environments.locomanipulation.base import LocoManipulationEnv
from robocasa.models.scenes import GroundArena
from robocasa.models.scenes.factory_arena import FactoryArena
from robocasa.utils.scene.configs import SceneConfig, SceneScaleConfig
from robocasa.utils.scene.scene import Scene, SceneObject
from robocasa.utils.scene.success_criteria import SuccessCriteria


class LMEnvBase(LocoManipulationEnv):
    SCENE_SCALE = SceneScaleConfig()

    def __init__(
        self,
        translucent_robot: bool = False,
        use_object_obs: bool = False,
        scene_scale: Optional[SceneScaleConfig] = None,
        *args,
        **kwargs,
    ):
        self.scene_scale = scene_scale or self.SCENE_SCALE
        super().__init__(translucent_robot, use_object_obs, *args, **kwargs)

    def _load_model(self):
        self.scene = Scene(self, self._get_env_config(), self.scene_scale)
        self.mujoco_objects = self.scene.mujoco_objects

        super()._load_model()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            self.scene.reset()

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = {}
        for obj in self.mujoco_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _get_env_config(self) -> SceneConfig:
        return SceneConfig(
            objects=self._get_objects(),
            success=self._get_success_criteria(),
            instruction=self._get_instruction(),
        )

    @abstractmethod
    def _get_objects(self) -> list[SceneObject]:
        raise NotImplementedError

    @abstractmethod
    def _get_success_criteria(self) -> SuccessCriteria:
        raise NotImplementedError

    @abstractmethod
    def _get_instruction(self) -> str:
        raise NotImplementedError

    def _check_success(self):
        return self.scene.success()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene.instruction
        return ep_meta


# noinspection PyAbstractClass
class LMSimpleEnv(LMEnvBase):
    MUJOCO_ARENA_CLS = GroundArena


# noinspection PyAbstractClass
class LMFactoryEnv(LMEnvBase):
    MUJOCO_ARENA_CLS = FactoryArena
