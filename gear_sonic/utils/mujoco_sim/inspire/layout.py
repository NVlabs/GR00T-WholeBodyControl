"""Name-resolved MuJoCo addresses for the native 6D hand contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gear_sonic.utils.hand_control.inspire.config import InspireMapping
from gear_sonic.utils.hand_control.inspire.contract import HAND_DOF


@dataclass(frozen=True)
class SideLayout:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray


@dataclass(frozen=True)
class InspireMujocoLayout:
    left: SideLayout
    right: SideLayout

    @classmethod
    def resolve(cls, model: Any, mapping: InspireMapping) -> "InspireMujocoLayout":
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by CLI dependency checks
            raise RuntimeError("MuJoCo is required; install gear_sonic[sim]") from exc

        def resolve_side(joints) -> SideLayout:
            joint_ids = np.asarray(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, item.mujoco_joint)
                    for item in joints
                ],
                dtype=np.int32,
            )
            actuator_ids = np.asarray(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, item.mujoco_actuator)
                    for item in joints
                ],
                dtype=np.int32,
            )
            if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
                raise ValueError(
                    "mapping contains a MuJoCo joint or actuator absent from the model"
                )
            if (
                len(set(joint_ids.tolist())) != HAND_DOF
                or len(set(actuator_ids.tolist())) != HAND_DOF
            ):
                raise ValueError("mapping must resolve to six unique joints and actuators per hand")
            return SideLayout(
                joint_ids=joint_ids,
                qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
                dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
                actuator_ids=actuator_ids,
            )

        return cls(left=resolve_side(mapping.left), right=resolve_side(mapping.right))

    def read(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(data.qpos[self.left.qpos_addresses], dtype=np.float32).copy(),
            np.asarray(data.qpos[self.right.qpos_addresses], dtype=np.float32).copy(),
            np.asarray(data.qvel[self.left.dof_addresses], dtype=np.float32).copy(),
            np.asarray(data.qvel[self.right.dof_addresses], dtype=np.float32).copy(),
        )
