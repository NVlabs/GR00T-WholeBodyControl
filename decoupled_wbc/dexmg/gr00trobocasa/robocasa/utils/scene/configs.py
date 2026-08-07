from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from robocasa.utils.scene.success_criteria import SuccessCriteria

if TYPE_CHECKING:
    from robocasa.utils.scene.scene import SceneObject


@dataclass(frozen=True)
class ReferenceConfig:
    obj: Optional[SceneObject]
    spawn_id: Optional[int] = None
    on_top: bool = True

    def to_dict(self) -> Optional[dict[str, Any]]:
        if self.obj is None:
            return None
        if self.spawn_id is None:
            return {"reference": self.obj.cfg.name, "on_top": self.on_top}
        else:
            return {"reference": (self.obj.cfg.name, self.spawn_id)}


@dataclass(frozen=True)
class SamplingConfig:
    x_range: np.ndarray = np.zeros(2)
    y_range: np.ndarray = np.zeros(2)
    rotation: np.ndarray = np.zeros(2)
    reference_pos: np.ndarray = np.zeros(3)
    z_offset: float = 0
    reference: Optional[ReferenceConfig] = None

    @property
    def sampling_args(self) -> Optional[dict]:
        if self.reference is None:
            return None
        return self.reference.to_dict()


@dataclass(frozen=True)
class ObjectConfig:
    mjcf_path: str
    name: str
    static: bool = False
    scale: float = 1.0
    density: int = 100
    friction: tuple[float, float, float] = (1, 1, 1)
    margin: Optional[float] = None
    rgba: Optional[tuple[float, float, float, float]] = None
    sampler_config: SamplingConfig = SamplingConfig()


@dataclass(frozen=True)
class SceneConfig:
    objects: list[SceneObject]
    success: SuccessCriteria
    instruction: str


class SceneHandedness(Enum):
    UNIVERSAL = "UNIVERSAL"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class SceneScaleConfig:
    planar_scale: Union[float, tuple[float, float]] = 1.0
    vertical_scale: float = 1.0
    handedness: SceneHandedness = SceneHandedness.UNIVERSAL
    handedness_axis: int = 1

    def get_ref_pos(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.astype(float, copy=True)  # ensure float
        pos[self.handedness_axis] *= float(self._handedness_factor)
        pos[:2] = pos[:2] * np.array(self._scale, dtype=float)
        pos[2] *= float(self.vertical_scale)
        return pos

    def get_rotation(self, rot: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if rot is None:
            return None
        rot = rot.astype(float, copy=True)
        rot *= float(self._handedness_factor)
        return rot

    def get_ranges(self, x_range: np.ndarray, y_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_range = x_range.astype(float, copy=True)
        y_range = y_range.astype(float, copy=True)

        if self.handedness_axis == 0:
            x_range *= float(self._handedness_factor)
            x_range.sort()
        elif self.handedness_axis == 1:
            y_range *= float(self._handedness_factor)
            y_range.sort()
        return x_range, y_range

    def get_z_offset(self, z_offset: float) -> float:
        return float(z_offset) * float(self.vertical_scale)

    @property
    def _handedness_factor(self) -> float:
        if self.handedness == "LEFT":  # replace with enum comparison
            return -1.0
        else:
            return 1.0

    @property
    def _scale(self) -> tuple[float, float]:
        if isinstance(self.planar_scale, tuple) and len(self.planar_scale) == 2:
            return float(self.planar_scale[0]), float(self.planar_scale[1])
        elif isinstance(self.planar_scale, (float, int)):
            scale = float(self.planar_scale)
            return scale, scale
        else:
            raise ValueError(
                f"Invalid type for 'planar_scale': expected float or tuple[float, float], "
                f"but got {type(self.planar_scale).__name__}"
            )
