import numpy as np
from robocasa.environments.locomanipulation.base import RobotPoseRandomizer
from robocasa.environments.locomanipulation.locomanip import LMSimpleEnv
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.scene.configs import (
    ObjectConfig,
    ReferenceConfig,
    SamplingConfig,
    SceneScaleConfig,
)
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import (
    AllCriteria,
    AnyCriteria,
    IsGrasped,
    IsInContact,
    IsPositionInRange,
    IsRobotInRange,
    IsUpright,
    NotCriteria,
    SuccessCriteria,
)
from robocasa.utils.visuals_utls import Gradient, randomize_materials_rgba


class LMPickBottle(LMSimpleEnv, DexMGConfigHelper):
    SCENE_SCALE = SceneScaleConfig(planar_scale=1.0)

    TABLE_GRADIENT: Gradient = Gradient(
        np.array([0.68, 0.34, 0.07, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    LIFT_OFFSET = 0.1

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference_pos=np.array([0.4, 0, self.table.mj_obj.top_offset[2]]),
                ),
            )
        )
        return [self.table, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            IsPositionInRange(self.bottle, 2, self.table.mj_obj.top_offset[2] + self.LIFT_OFFSET),
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_bottle"] = int(
            self._check_grasp(self.robots[0].gripper["right"], self.bottle.mj_obj.contact_geoms)
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            self._randomize_table_rgba()

    def _randomize_table_rgba(self):
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
        )


class LMPickBottleHigh(LMPickBottle):
    TABLE_OFFSET = 0.1

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, self.TABLE_OFFSET]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference_pos=np.array(
                        [0.4, 0, self.TABLE_OFFSET + self.table.mj_obj.top_offset[2]]
                    ),
                    reference=ReferenceConfig(obj=self.table),
                ),
            )
        )
        return [self.table, self.bottle]


class LMNavPickBottle(LMPickBottle):
    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

    def _get_instruction(self) -> str:
        return "Walk forward and pick up the bottle from the table."


class LMPickBottleGround(LMPickBottle):
    def _get_objects(self) -> list[SceneObject]:
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference_pos=np.array(
                        [0.4, 0, 0.075]
                    ),  # Base position on ground (z=0.075 is bottle radius)
                ),
            )
        )
        return [self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            IsPositionInRange(self.bottle, 2, self.LIFT_OFFSET, 10),
        )

    def _randomize_table_rgba(self):
        pass


class LMPnPBottle(LMPickBottle):
    LIFT_OFFSET = 0.15

    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 1.2, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        return [self.table, self.table_target, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle, symmetric=True), IsInContact(self.bottle, self.table_target)
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it on the other table."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            target_table=dict(obj_name=self.table_target.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        obj_z = self.sim.data.body_xpos[self.obj_body_id(self.bottle.mj_obj.name)][2]
        target_table_pos = self.sim.data.body_xpos[self.obj_body_id(self.table_target.mj_obj.name)]
        target_table_z = target_table_pos[2] + self.table_target.mj_obj.top_offset[2]
        return dict(obj_off_table=int(obj_z - target_table_z > self.LIFT_OFFSET))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_off_table",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Second subtask for placing on target table
        task.task_spec_0.subtask_2 = dict(
            object_ref="target_table",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def _randomize_table_rgba(self):
        for table in [self.table_target, self.table]:
            randomize_materials_rgba(
                rng=self.rng, mjcf_obj=table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
            )


class LMPickMultipleBottles(LMPickBottle):
    BOTTLE_COLOURS = [(0.3, 0.7, 0.8, 1.0), (0.8, 0.4, 0.3, 1.0)]
    BOTTLES_COUNT = 2
    Y_OFFSET_STEP = 0.1

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 0, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )

        self.bottles = []
        offsets = np.arange(self.BOTTLES_COUNT) - (self.BOTTLES_COUNT - 1) / 2.0
        for i in range(self.BOTTLES_COUNT):
            reference_pos = np.array([0.4, 0, self.table.mj_obj.top_offset[2]])
            reference_pos += np.array([0, self.Y_OFFSET_STEP * offsets[i], 0])
            bottle = SceneObject(
                ObjectConfig(
                    name=f"bottle_{i}",
                    mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                    static=False,
                    scale=0.6,
                    sampler_config=SamplingConfig(
                        x_range=np.array([-0.08, 0.04]),
                        y_range=np.array([-0.04, 0.04]),
                        rotation=np.array([-np.pi, np.pi]),
                        reference_pos=reference_pos,
                    ),
                    rgba=self.BOTTLE_COLOURS[i % len(self.BOTTLE_COLOURS)],
                )
            )
            self.bottles.append(bottle)
        return [self.table, *self.bottles]

    def _get_success_criteria(self) -> SuccessCriteria:
        criteria = []
        for bottle in self.bottles:
            criteria.append(AnyCriteria(IsGrasped(bottle, "right"), IsGrasped(bottle, "left")))
            criteria.append(
                IsPositionInRange(bottle, 2, self.table.mj_obj.top_offset[2] + self.LIFT_OFFSET, 10)
            )
        return AllCriteria(*criteria)

    def _get_instruction(self) -> str:
        return "Pick up bottles."

    def get_object(self):
        return {
            bottle.mj_obj.name: dict(obj_name=bottle.mj_obj.root_body, obj_type="body")
            for bottle in self.bottles
        }

    def get_subtask_term_signals(self):
        return {
            f"grasp_{bottle.mj_obj.name}": int(
                self._check_grasp(self.robots[0].gripper["right"], bottle.mj_obj)
                or self._check_grasp(self.robots[0].gripper["left"], bottle.mj_obj)
            )
            for bottle in self.bottles
        }

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        for i in range(LMPickMultipleBottles.BOTTLES_COUNT):
            subtask = dict(
                object_ref=f"bottle_{i}",
                subtask_term_signal=None,
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            setattr(task.task_spec_0, f"subtask_{i+1}", subtask)
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class LMPnPMultipleBottles(LMPickMultipleBottles):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.table_target = SceneObject(
            ObjectConfig(
                name="table_target",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.02, 0.02]),
                    y_range=np.array([-0.02, 0.02]),
                    reference_pos=np.array([0.5, 1.2, 0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        return [self.table, self.table_target, *self.bottles]

    def _get_success_criteria(self) -> SuccessCriteria:
        criteria = [
            AllCriteria(IsInContact(bottle, self.table_target), IsUpright(bottle, symmetric=True))
            for bottle in self.bottles
        ]
        return AllCriteria(*criteria)

    def _get_instruction(self) -> str:
        return "Pick up bottles from one table and place it on the other."

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        for i in range(LMPnPMultipleBottles.BOTTLES_COUNT):
            bottle_name = f"bottle_{i}"
            subtask = dict(
                object_ref=bottle_name,
                subtask_term_signal=f"{bottle_name}_off_table",
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            setattr(task.task_spec_0, f"subtask_{i+1}", subtask)
        # Next subtask for placing on target table
        task.task_spec_0.subtask_3 = dict(
            object_ref="target_table",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def get_subtask_term_signals(self):
        signals = dict()
        for bottle in self.bottles:
            obj_z = self.sim.data.body_xpos[self.obj_body_id(self.bottle.mj_obj.name)][2]
            target_table_pos = self.sim.data.body_xpos[
                self.obj_body_id(self.table_target.mj_obj.name)
            ]
            target_table_z = target_table_pos[2] + self.table_target.mj_obj.top_offset[2]
            signals[f"{bottle.mj_obj.name}_off_table"] = int(
                obj_z - target_table_z > self.LIFT_OFFSET
            )
        return signals

    def _randomize_table_rgba(self):
        for table in [self.table_target, self.table]:
            randomize_materials_rgba(
                rng=self.rng, mjcf_obj=table.mj_obj, gradient=self.TABLE_GRADIENT, linear=True
            )


class LMPickBottleShelf(LMPickBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.shelf = SceneObject(
            ObjectConfig(
                name="shelf",
                mjcf_path="objects/omniverse/locomanip/lab_shelf/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    rotation=np.array([np.pi / 2, np.pi / 2]),
                    reference_pos=np.array([0.9, 0, 0]),
                ),
            )
        )
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.14, -0.06]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.shelf, spawn_id=2),
                ),
            )
        )
        return [self.shelf, self.bottle]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsGrasped(self.bottle, "right"),
            NotCriteria(IsInContact(self.bottle, self.shelf)),
        )


class LMNavPickBottleShelf(LMPickBottleShelf):
    ROBOT_DISTANCE_THRESHOLD = 1.0

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.1, 0.1), (-0.1, 0.1), (-np.pi / 6, np.pi / 6))

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            NotCriteria(IsRobotInRange(self.shelf, self.ROBOT_DISTANCE_THRESHOLD, True)),
            IsGrasped(self.bottle, "right"),
            NotCriteria(IsInContact(self.bottle, self.shelf)),
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle from the shelf and move backward away from it."


class LMPickBottleShelfLow(LMPickBottleShelf):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.14, -0.06]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.shelf, spawn_id=1),
                ),
            )
        )
        return [self.shelf, self.bottle]


class LMNavPickBottleShelfLow(LMNavPickBottleShelf):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.bottle = SceneObject(
            ObjectConfig(
                name="bottle",
                mjcf_path="objects/omniverse/locomanip/jug_a01/model.xml",
                static=False,
                scale=0.6,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.14, -0.06]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.shelf, spawn_id=1),
                ),
            )
        )
        return [self.shelf, self.bottle]


class LMPnPBottleToPlate(LMPnPBottle):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.plate = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/omniverse/locomanip/plate_1/model.xml",
                scale=1.0,
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.2 - 0.08, -0.2 + 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference=ReferenceConfig(self.table_target),
                ),
            )
        )
        return [self.table, self.table_target, self.bottle, self.plate]

    def _get_success_criteria(self) -> SuccessCriteria:
        return AllCriteria(
            IsUpright(self.bottle, symmetric=True), IsInContact(self.bottle, self.plate)
        )

    def _get_instruction(self) -> str:
        return "Pick up the bottle and place it on the plate."

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.bottle.mj_obj.root_body, obj_type="body"),
            plate=dict(obj_name=self.plate.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        obj_z = self.sim.data.body_xpos[self.obj_body_id(self.bottle.mj_obj.name)][2]
        target_table_pos = self.sim.data.body_xpos[self.obj_body_id(self.table_target.mj_obj.name)]
        target_table_z = target_table_pos[2] + self.table_target.mj_obj.top_offset[2]
        return dict(obj_off_table=int(obj_z - target_table_z > self.LIFT_OFFSET))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_off_table",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Second subtask for placing on plate
        task.task_spec_0.subtask_2 = dict(
            object_ref="plate",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class LMPnPAppleToPlate(LMPnPBottleToPlate):
    def _get_objects(self) -> list[SceneObject]:
        super()._get_objects()
        self.apple = SceneObject(
            ObjectConfig(
                name="apple",
                mjcf_path="objects/omniverse/locomanip/apple_0/model.xml",
                static=False,
                scale=1.0,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.08, 0.04]),
                    y_range=np.array([-0.08, 0.08]),
                    rotation=np.array([-np.pi, np.pi]),
                    reference_pos=np.array([0.4, 0, self.table.mj_obj.top_offset[2]]),
                ),
            )
        )
        return [self.table, self.table_target, self.apple, self.plate]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsInContact(self.apple, self.plate)

    def _get_instruction(self) -> str:
        return "pick up the apple, walk left and place the apple on the plate."

    def get_object(self):
        return dict(
            apple=dict(obj_name=self.apple.mj_obj.root_body, obj_type="body"),
            plate=dict(obj_name=self.plate.mj_obj.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        obj_z = self.sim.data.body_xpos[self.obj_body_id(self.apple.mj_obj.name)][2]
        target_table_pos = self.sim.data.body_xpos[self.obj_body_id(self.table_target.mj_obj.name)]
        target_table_z = target_table_pos[2] + self.table_target.mj_obj.top_offset[2]
        return dict(obj_off_table=int(obj_z - target_table_z > self.LIFT_OFFSET))

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="apple",
            subtask_term_signal="obj_off_table",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Second subtask for placing on plate
        task.task_spec_0.subtask_2 = dict(
            object_ref="plate",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()
