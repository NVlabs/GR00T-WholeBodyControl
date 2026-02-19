import numpy as np
from robocasa.environments.locomanipulation.locomanip import LMFactoryEnv
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneHandedness,
    SceneScaleConfig,
)
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import (
    AllCriteria,
    IsInContact,
    IsUpright,
    SuccessCriteria,
)


class LMBottlePnP(LMFactoryEnv):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, 0.8, 0]),
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )
        self.table_origin = SceneObject(
            ObjectConfig(
                name="table_origin",
                mjcf_path="objects/omniverse/locomanip/factory_ergo_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([1.2, -0.8, 0]),
                    rotation=np.array([np.pi, np.pi]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.4, -0.35]),
                    y_range=np.array([-0.1, 0.1]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsInContact(self.bottle, self.table_target), IsUpright(self.bottle))

    def _get_instruction(self) -> str:
        return "Pick up the bottle from one table and place it on the other."


class LMBoxPnP(LMBottlePnP):
    SCENE_SCALE = SceneScaleConfig(planar_scale=(1, 1), handedness=SceneHandedness.RIGHT)

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.box = SceneObject(
            ObjectConfig(
                name="obj",
                mjcf_path="objects/omniverse/locomanip/cardbox_a1/model.xml",
                static=False,
                scale=0.7,
                density=1,
                friction=(2, 1, 1),
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.35, -0.3]),
                    y_range=np.array([-0.1, 0.1]),
                    rotation=np.array([np.pi * 0.9, np.pi * 1.1]),
                    reference=ReferenceConfig(self.table_origin),
                ),
            )
        )
        return [self.table_origin, self.table_target, self.box]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(IsInContact(self.box, self.table_target), IsUpright(self.box))

    def _get_instruction(self) -> str:
        return "Pick up the box from one table and place it on the other."
