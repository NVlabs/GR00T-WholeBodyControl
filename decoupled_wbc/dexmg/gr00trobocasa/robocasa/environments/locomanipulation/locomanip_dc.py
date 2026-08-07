from robocasa import (
    LMNavPickBottle,
    LMPnPAppleToPlate,
)
from robocasa.models.scenes.lab_arena import LabArena


class LabEnvMixin:
    MUJOCO_ARENA_CLS = LabArena


class LMNavPickBottleDC(LabEnvMixin, LMNavPickBottle): ...


class LMPnPAppleToPlateDC(LabEnvMixin, LMPnPAppleToPlate): ...
