"""Selected native124 iteration-21204 action semantics inside SONIC V2.

This action term is deliberately separate from both the released SONIC action
term and the legacy two-input native124 policy wrapper.  Its ActionManager
boundary is the selected actor's raw 23-vector in compact hardware/MuJoCo
order.  Processing performs exactly one frozen chain::

    selected raw hardware
      -> selected HOME/SCALE hardware target
      -> plain SONIC raw native action
      -> safe-target V2 once
      -> unbiased hardware target
      -> encoder-bias correction at application

Only Torch operations execute in the control loop.  The support/admission gate
is intentionally not imported here because it is an offline NumPy validator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    _MJLAB_IMPORT_ERROR,
    ActionTerm,
    ActionTermCfg,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_JOINT_NAMES,
    TARGET_DOF,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE as SELECTED_21204_ACTION_SCALE_HARDWARE,
    HOME_Q_HARDWARE as SELECTED_21204_HOME_Q_HARDWARE,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_CONSTANTS_SHA256,
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_FORMULA_SHA256,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_torch,
)

SELECTED_21204_V2_ACTION_CONSTANTS_SHA256 = "a27209ce0a263a681b16820323aa3948c388280c9667d5fab7511435e81a7f48"


def _constant_payload() -> dict[str, object]:
    return {
        "schema": "g1_true23_selected21204_to_sonic_v2_action_constants_v1",
        "selected_home_q_hardware": SELECTED_21204_HOME_Q_HARDWARE.tolist(),
        "selected_action_scale_hardware": (SELECTED_21204_ACTION_SCALE_HARDWARE.tolist()),
        "sonic_default_q_hardware": list(SAFE_TARGET_DEFAULT_Q_HARDWARE),
        "sonic_action_scale_hardware": list(HARDWARE_23_ACTION_SCALE),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "native_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "isaaclab_to_mujoco": list(ISAACLAB_TO_MUJOCO_DOF),
        "mujoco_to_isaaclab": list(MUJOCO_TO_ISAACLAB_DOF),
        "safe_target_constants_sha256": SAFE_TARGET_CONSTANTS_SHA256,
        "safe_target_formula_sha256": SAFE_TARGET_FORMULA_SHA256,
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_selected21204_v2_action_constants() -> str:
    """Fail closed if selected, SONIC, permutation, or V2 constants drift."""

    selected_home = np.asarray(SELECTED_21204_HOME_Q_HARDWARE)
    selected_scale = np.asarray(SELECTED_21204_ACTION_SCALE_HARDWARE)
    if (
        selected_home.shape != (TARGET_DOF,)
        or selected_scale.shape != (TARGET_DOF,)
        or selected_home.dtype != np.float32
        or selected_scale.dtype != np.float32
        or not np.isfinite(selected_home).all()
        or not np.isfinite(selected_scale).all()
        or np.any(selected_scale <= np.float32(0.0))
    ):
        raise RuntimeError("selected-21204 HOME/SCALE constant contract drift")

    permutations = (ISAACLAB_TO_MUJOCO_DOF, MUJOCO_TO_ISAACLAB_DOF)
    if any(
        len(permutation) != TARGET_DOF or sorted(permutation) != list(range(TARGET_DOF))
        for permutation in permutations
    ):
        raise RuntimeError("true23 hardware/native permutation contract drift")
    if any(ISAACLAB_TO_MUJOCO_DOF[MUJOCO_TO_ISAACLAB_DOF[index]] != index for index in range(TARGET_DOF)):
        raise RuntimeError("true23 hardware/native permutations are not inverse")
    if tuple(NATIVE_IL23_JOINT_NAMES[index] for index in ISAACLAB_TO_MUJOCO_DOF) != tuple(HARDWARE_23_JOINT_NAMES):
        raise RuntimeError("true23 hardware/native joint-name order drift")

    actual = _canonical_sha256(_constant_payload())
    if actual != SELECTED_21204_V2_ACTION_CONSTANTS_SHA256:
        raise RuntimeError(
            "selected-21204 to SONIC V2 action constants SHA256 mismatch: "
            f"expected {SELECTED_21204_V2_ACTION_CONSTANTS_SHA256}, got {actual}"
        )
    return actual


def _require_float32_batch(
    value: torch.Tensor,
    *,
    batch_size: int | None,
    context: str,
) -> None:
    expected_shape = f"[{batch_size},{TARGET_DOF}]" if batch_size is not None else f"[batch,{TARGET_DOF}]"
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[-1] != TARGET_DOF
        or (batch_size is not None and value.shape[0] != batch_size)
    ):
        raise ValueError(f"{context} must have shape {expected_shape}")
    if value.dtype != torch.float32:
        raise ValueError(f"{context} must be float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{context} contains NaN or Inf")


@dataclass(frozen=True)
class Selected21204V2ActionDiagnostics:
    """Detached snapshot of every action space in the frozen composite."""

    selected_raw_action_hardware: torch.Tensor
    candidate_target_hardware: torch.Tensor
    plain_sonic_raw_action_native: torch.Tensor
    safe_native_action: torch.Tensor
    final_target_hardware: torch.Tensor
    effective_selected_raw_action_hardware: torch.Tensor
    raw_clip_mask_native: torch.Tensor


def _selected21204_to_sonic_v2(
    selected_raw_action_hardware: torch.Tensor,
    *,
    selected_home_hardware: torch.Tensor,
    selected_scale_hardware: torch.Tensor,
    sonic_default_hardware: torch.Tensor,
    sonic_scale_hardware: torch.Tensor,
    hardware_to_native_indices: torch.Tensor,
) -> Selected21204V2ActionDiagnostics:
    candidate_hardware = selected_home_hardware + selected_scale_hardware * selected_raw_action_hardware
    plain_raw_hardware = (candidate_hardware - sonic_default_hardware) / sonic_scale_hardware
    plain_raw_native = plain_raw_hardware.index_select(
        -1,
        hardware_to_native_indices,
    )
    # Support gate admits only strict abs(raw) < 10.  Boundary values are
    # therefore marked even though torch.clamp itself leaves +/-10 unchanged.
    clip_mask_native = torch.abs(plain_raw_native) >= SAFE_TARGET_RAW_ACTION_CLIP

    # Sole V2 application.  Returned target is already hardware/MuJoCo order;
    # never feed it through the linear SONIC action term afterward.
    safe_native, final_target_hardware = safe_target_transform_torch(plain_raw_native)
    effective_selected_raw_hardware = (final_target_hardware - selected_home_hardware) / selected_scale_hardware
    if not torch.isfinite(effective_selected_raw_hardware).all():
        raise RuntimeError("effective selected-21204 applied action contains NaN or Inf")
    return Selected21204V2ActionDiagnostics(
        selected_raw_action_hardware=selected_raw_action_hardware,
        candidate_target_hardware=candidate_hardware,
        plain_sonic_raw_action_native=plain_raw_native,
        safe_native_action=safe_native,
        final_target_hardware=final_target_hardware,
        effective_selected_raw_action_hardware=(effective_selected_raw_hardware),
        raw_clip_mask_native=clip_mask_native,
    )


def selected21204_raw_hardware_to_sonic_v2_torch(
    selected_raw_action_hardware: torch.Tensor,
) -> Selected21204V2ActionDiagnostics:
    """Pure Torch reference chain for batched selected raw hardware actions."""

    validate_selected21204_v2_action_constants()
    _require_float32_batch(
        selected_raw_action_hardware,
        batch_size=None,
        context="selected-21204 raw hardware action",
    )
    device = selected_raw_action_hardware.device
    return _selected21204_to_sonic_v2(
        selected_raw_action_hardware,
        selected_home_hardware=torch.as_tensor(
            SELECTED_21204_HOME_Q_HARDWARE,
            dtype=torch.float32,
            device=device,
        ),
        selected_scale_hardware=torch.as_tensor(
            SELECTED_21204_ACTION_SCALE_HARDWARE,
            dtype=torch.float32,
            device=device,
        ),
        sonic_default_hardware=torch.as_tensor(
            SAFE_TARGET_DEFAULT_Q_HARDWARE,
            dtype=torch.float32,
            device=device,
        ),
        sonic_scale_hardware=torch.as_tensor(
            HARDWARE_23_ACTION_SCALE,
            dtype=torch.float32,
            device=device,
        ),
        hardware_to_native_indices=torch.as_tensor(
            MUJOCO_TO_ISAACLAB_DOF,
            dtype=torch.long,
            device=device,
        ),
    )


def encoder_biased_hardware_target(
    final_target_hardware: torch.Tensor,
    encoder_bias_hardware: torch.Tensor,
) -> torch.Tensor:
    """Apply same final ``target - encoder_bias`` rule as true23 action term."""

    _require_float32_batch(
        final_target_hardware,
        batch_size=None,
        context="final hardware target",
    )
    _require_float32_batch(
        encoder_bias_hardware,
        batch_size=final_target_hardware.shape[0],
        context="hardware encoder bias",
    )
    if encoder_bias_hardware.device != final_target_hardware.device:
        raise ValueError("hardware encoder bias must share target device")
    return final_target_hardware - encoder_bias_hardware


class Selected21204ToSonicV2ActionCore:
    """Dependency-free state engine used by MJLab ActionTerm and unit tests."""

    def __init__(self, *, num_envs: int, device: str | torch.device) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        validate_selected21204_v2_action_constants()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._selected_home_hardware = torch.as_tensor(
            SELECTED_21204_HOME_Q_HARDWARE,
            dtype=torch.float32,
            device=self.device,
        ).clone()
        self._selected_scale_hardware = torch.as_tensor(
            SELECTED_21204_ACTION_SCALE_HARDWARE,
            dtype=torch.float32,
            device=self.device,
        ).clone()
        self._sonic_default_hardware = torch.as_tensor(
            SAFE_TARGET_DEFAULT_Q_HARDWARE,
            dtype=torch.float32,
            device=self.device,
        )
        self._sonic_scale_hardware = torch.as_tensor(
            HARDWARE_23_ACTION_SCALE,
            dtype=torch.float32,
            device=self.device,
        )
        self._hardware_to_native_indices = torch.as_tensor(
            MUJOCO_TO_ISAACLAB_DOF,
            dtype=torch.long,
            device=self.device,
        )
        self._selected_raw_action_hardware = torch.zeros(
            (num_envs, TARGET_DOF),
            dtype=torch.float32,
            device=self.device,
        )
        self._candidate_target_hardware = torch.empty_like(self._selected_raw_action_hardware)
        self._plain_sonic_raw_action_native = torch.empty_like(self._selected_raw_action_hardware)
        self._safe_native_action = torch.empty_like(self._selected_raw_action_hardware)
        self._final_target_hardware = torch.empty_like(self._selected_raw_action_hardware)
        self._effective_selected_raw_action_hardware = torch.empty_like(self._selected_raw_action_hardware)
        self._raw_clip_mask_native = torch.empty(
            (num_envs, TARGET_DOF),
            dtype=torch.bool,
            device=self.device,
        )
        self._recompute()

    @property
    def raw_action(self) -> torch.Tensor:
        """Original selected raw hardware action at ActionManager boundary."""

        return self._selected_raw_action_hardware

    @property
    def safe_native_action(self) -> torch.Tensor:
        """Applied V2 native action for SONIC previous-action history."""

        return self._safe_native_action

    @property
    def processed_action(self) -> torch.Tensor:
        """Final unbiased hardware target consumed by MJLab."""

        return self._final_target_hardware

    @property
    def effective_selected_raw_action_hardware(self) -> torch.Tensor:
        """Selected coordinates of target actually applied after V2."""

        return self._effective_selected_raw_action_hardware.detach().clone()

    @property
    def diagnostics(self) -> Selected21204V2ActionDiagnostics:
        """Return clones so callers cannot mutate live action state."""

        return Selected21204V2ActionDiagnostics(
            selected_raw_action_hardware=(self._selected_raw_action_hardware.detach().clone()),
            candidate_target_hardware=(self._candidate_target_hardware.detach().clone()),
            plain_sonic_raw_action_native=(self._plain_sonic_raw_action_native.detach().clone()),
            safe_native_action=self._safe_native_action.detach().clone(),
            final_target_hardware=(self._final_target_hardware.detach().clone()),
            effective_selected_raw_action_hardware=(self._effective_selected_raw_action_hardware.detach().clone()),
            raw_clip_mask_native=self._raw_clip_mask_native.detach().clone(),
        )

    def _recompute(self) -> None:
        result = _selected21204_to_sonic_v2(
            self._selected_raw_action_hardware,
            selected_home_hardware=self._selected_home_hardware,
            selected_scale_hardware=self._selected_scale_hardware,
            sonic_default_hardware=self._sonic_default_hardware,
            sonic_scale_hardware=self._sonic_scale_hardware,
            hardware_to_native_indices=self._hardware_to_native_indices,
        )
        self._candidate_target_hardware.copy_(result.candidate_target_hardware)
        self._plain_sonic_raw_action_native.copy_(result.plain_sonic_raw_action_native)
        self._safe_native_action.copy_(result.safe_native_action)
        self._final_target_hardware.copy_(result.final_target_hardware)
        self._effective_selected_raw_action_hardware.copy_(result.effective_selected_raw_action_hardware)
        self._raw_clip_mask_native.copy_(result.raw_clip_mask_native)

    def process_actions(self, actions: torch.Tensor) -> None:
        _require_float32_batch(
            actions,
            batch_size=self.num_envs,
            context="selected-21204 raw hardware action",
        )
        if actions.device != self.device:
            raise ValueError("selected-21204 raw hardware action device mismatch")
        self._selected_raw_action_hardware.copy_(actions)
        self._recompute()

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._selected_raw_action_hardware[env_ids] = 0.0
        self._recompute()

    def applied_target(
        self,
        encoder_bias_hardware: torch.Tensor,
    ) -> torch.Tensor:
        return encoder_biased_hardware_target(
            self._final_target_hardware,
            encoder_bias_hardware,
        )


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class Selected21204HardwareToSonicV2JointPositionActionCfg(ActionTermCfg):
        """Build frozen selected-21204 hardware -> SONIC V2 action term."""

        entity_name: str = "robot"
        actuator_names: tuple[str, ...] = (".*",)

        def build(
            self,
            env: Any,
        ) -> "Selected21204HardwareToSonicV2JointPositionAction":
            return Selected21204HardwareToSonicV2JointPositionAction(
                self,
                env,
            )

    class Selected21204HardwareToSonicV2JointPositionAction(ActionTerm):
        """MJLab action term preserving selected raw hardware input exactly."""

        cfg: Selected21204HardwareToSonicV2JointPositionActionCfg

        def __init__(
            self,
            cfg: Selected21204HardwareToSonicV2JointPositionActionCfg,
            env: Any,
        ) -> None:
            super().__init__(cfg=cfg, env=env)
            validate_selected21204_v2_action_constants()
            target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
            if tuple(target_names) != tuple(HARDWARE_23_JOINT_NAMES):
                raise ValueError(
                    "MJLab actuated joint order differs from selected-21204 "
                    f"hardware true23 contract: {tuple(target_names)!r}"
                )
            self._target_ids = torch.as_tensor(
                target_ids,
                dtype=torch.long,
                device=self.device,
            )
            self._selected_v2_core = Selected21204ToSonicV2ActionCore(
                num_envs=self.num_envs,
                device=self.device,
            )

        @property
        def action_dim(self) -> int:
            return TARGET_DOF

        @property
        def raw_action(self) -> torch.Tensor:
            return self._selected_v2_core.raw_action

        @property
        def processed_action(self) -> torch.Tensor:
            return self._selected_v2_core.processed_action

        @property
        def safe_native_action(self) -> torch.Tensor:
            return self._selected_v2_core.safe_native_action.detach().clone()

        @property
        def selected_raw_action_hardware(self) -> torch.Tensor:
            return self._selected_v2_core.diagnostics.selected_raw_action_hardware

        @property
        def candidate_target_hardware(self) -> torch.Tensor:
            return self._selected_v2_core.diagnostics.candidate_target_hardware

        @property
        def plain_sonic_raw_action_native(self) -> torch.Tensor:
            return self._selected_v2_core.diagnostics.plain_sonic_raw_action_native

        @property
        def final_target_hardware(self) -> torch.Tensor:
            return self._selected_v2_core.diagnostics.final_target_hardware

        @property
        def effective_selected_raw_action_hardware(self) -> torch.Tensor:
            """Use for native124 next-observation previous_action23."""

            return self._selected_v2_core.effective_selected_raw_action_hardware

        @property
        def raw_clip_mask_native(self) -> torch.Tensor:
            return self._selected_v2_core.diagnostics.raw_clip_mask_native

        @property
        def action_diagnostics(self) -> Selected21204V2ActionDiagnostics:
            return self._selected_v2_core.diagnostics

        def process_actions(self, actions: torch.Tensor) -> None:
            self._selected_v2_core.process_actions(actions)

        def apply_actions(self) -> None:
            encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
            applied_target = self._selected_v2_core.applied_target(encoder_bias)
            self._entity.set_joint_position_target(
                applied_target,
                joint_ids=self._target_ids,
            )

        def reset(
            self,
            env_ids: torch.Tensor | slice | None = None,
        ) -> None:
            self._selected_v2_core.reset(env_ids)


else:

    class Selected21204HardwareToSonicV2JointPositionActionCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class Selected21204HardwareToSonicV2JointPositionAction:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def selected21204_v2_action_contract() -> dict[str, object]:
    """Serializable frozen action-chain contract; no readiness claim."""

    return {
        "schema": "g1_true23_selected21204_to_sonic_v2_action_v1",
        "input": "selected21204_raw_hardware_23",
        "candidate_target": "selected21204_home_plus_scale_times_raw_hardware",
        "plain_sonic_action": "native_isaaclab_23",
        "safe_target_v2_application_count": 1,
        "output": "unbiased_hardware_mujoco_23_target",
        "encoder_bias_application": "final_target_hardware_minus_encoder_bias",
        "raw_clip_mask": "abs_plain_sonic_raw_native_greater_than_or_equal_10",
        "action_manager_raw_preserved": True,
        "sonic_previous_action_history": "applied_safe_native_action",
        "native124_previous_action_history": ("effective_selected_raw_action_hardware"),
        "effective_selected_raw_action_formula": (
            "(final_target_hardware-selected21204_home_hardware)/selected21204_scale_hardware"
        ),
        "constant_contract_sha256": validate_selected21204_v2_action_constants(),
        "safe_target_constants_sha256": SAFE_TARGET_CONSTANTS_SHA256,
        "safe_target_formula_sha256": SAFE_TARGET_FORMULA_SHA256,
        "offline_or_simulator_only": True,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


validate_selected21204_v2_action_constants()
