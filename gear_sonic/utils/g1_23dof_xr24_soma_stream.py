"""Exact rolling XR24 -> pinned SOMA evidence and latency analysis.

This module keeps NVIDIA SOMA's Newton solver, objectives, weights, iteration
counts, initialization sequence, feet stabilizer, and joint-limit clamper
unchanged.  It only moves their allocation and CUDA graph capture out of the
per-window path so measured XR24 frames can be processed one at a time.

The module is transport-free.  It never opens XR, ZMQ, DDS, ADB, or Unitree
channels and never authorizes actuation.  A rolling result remains
non-promotable until it agrees with the pinned batch pipeline on an exact
fixture and meets its deadline on the deployment machine.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from gear_sonic.utils.g1_23dof_contract import (
    ALL_REFERENCE_PROFILES,
    CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    REFERENCE_PROFILES,
    SOURCE_DOF,
    SOURCE_IL29_EXCLUDED_INDICES,
)
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    PINNED_CONFIG_SHA256,
    PINNED_SOMA_COMMIT,
    SOMA_MJ29_TO_CANONICAL_IL29,
    validate_raw_capture,
)
from gear_sonic.utils.g1_23dof_semantic_reference import (
    JOINT_ORDER,
    SEMANTIC_REFERENCE_FRAME_KIND,
    SEMANTIC_REFERENCE_SCHEMA_VERSION,
    SOURCE_RETARGETED_DELAYED,
    SOURCE_SAMPLE_PERIOD_NS,
    build_stream_reference_window,
    required_buffer_frames,
    validate_semantic_reference_frame,
    validate_semantic_reference_window,
)
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    XR24_ROLE_TO_SOMA_JOINT,
    _global_transforms,
    _interpolate_pose,
    _local_transforms,
    _normalize_quaternion,
    _quat_conjugate,
    _quat_mul,
    _quat_rotate,
    _rotation_6d_xyzw,
    _validate_pinned_soma_runtime,
    apply_xr24_soma_neutral_calibration,
    build_xr24_soma_neutral_calibration,
    require_xr24_neutral_standing,
)

ROLLING_SOMA_SCHEMA_VERSION = 1
ROLLING_SOMA_KIND = "g1_true23_xr24_pinned_soma_rolling_sample"
ROLLING_SOMA_STATUS = "exact_batch_equivalence_and_deadline_unproven"
ACTIVE_POLICY_FRESHNESS_NS = 60_000_000
CAUSAL_HISTORY_PROFILE = "true23_causal_step1_history_0p02s_v1"
CAUSAL_HISTORY_FRAME_COUNT = 10
CAUSAL_HISTORY_PROOF_FRAME_COUNT = CAUSAL_HISTORY_FRAME_COUNT + 1
CAUSAL_HISTORY_PACKET_SCHEMA_VERSION = 2
CAUSAL_HISTORY_PACKET_KIND = "g1_true23_xr24_soma_causal_history_packet"
CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT = "soma_il29_q_50hz_forward_difference_dq_v1"
CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION = 2
CAUSAL_HISTORY_ENCODER_TERMS_KIND = "g1_true23_causal_history_encoder_terms"
CAUSAL_HISTORY_REFERENCE_TERMS_KIND = "g1_true23_causal_history_reference_terms"
CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT = "robot_imu_quaternion_interpolated_at_pico_q9_v1"
EXPERIMENTAL_SOMA_ITERATION_PROFILES = {
    16: "experimental_soma_ik16_v1",
    12: "experimental_soma_ik12_v1",
}


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _module_sha256() -> str:
    payload = Path(__file__).read_bytes().decode("utf-8")
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def measured_future_latency_proof(
    profile: str,
    *,
    freshness_budget_ns: int = ACTIVE_POLICY_FRESHNESS_NS,
) -> dict[str, Any]:
    """Prove causal delay for a future tensor made only from measurements.

    Selected position ``k`` needs position ``k+1`` to prove its forward
    velocity.  Therefore even an ideal producer needs one real source sample
    beyond the selected future horizon.  Current checked-in stream validation
    deliberately retains one additional already-derived semantic frame.
    """

    if profile not in ALL_REFERENCE_PROFILES:
        raise ValueError(f"unsupported reference profile: {profile}")
    if (
        isinstance(freshness_budget_ns, bool)
        or not isinstance(freshness_budget_ns, int)
        or freshness_budget_ns <= 0
    ):
        raise ValueError("freshness_budget_ns must be a positive integer")
    contract = ALL_REFERENCE_PROFILES[profile]
    selected_horizon_ns = round(contract.horizon_s * 1_000_000_000)
    intrinsic_minimum_delay_ns = selected_horizon_ns + SOURCE_SAMPLE_PERIOD_NS
    checked_semantic_frames = required_buffer_frames(profile)
    # Adapter trace needs one q position beyond every semantic q/dq frame.
    checked_position_samples = checked_semantic_frames + 1
    checked_delay_ns = (checked_position_samples - 1) * SOURCE_SAMPLE_PERIOD_NS
    return {
        "profile": profile,
        "future_frame_step": contract.future_frame_step,
        "selected_horizon_ns": selected_horizon_ns,
        "forward_velocity_proof_ns": SOURCE_SAMPLE_PERIOD_NS,
        "intrinsic_minimum_measured_delay_ns": intrinsic_minimum_delay_ns,
        "checked_semantic_frame_count": checked_semantic_frames,
        "checked_position_sample_count": checked_position_samples,
        "checked_adapter_delay_ns": checked_delay_ns,
        "freshness_budget_ns": freshness_budget_ns,
        "intrinsic_budget_feasible": (intrinsic_minimum_delay_ns <= freshness_budget_ns),
        "checked_adapter_budget_feasible": checked_delay_ns <= freshness_budget_ns,
        "no_fake_future": True,
    }


def all_released_profile_latency_proofs() -> dict[str, dict[str, Any]]:
    """Return deterministic proof for both released temporal profiles."""

    return {
        profile: measured_future_latency_proof(profile)
        for profile in (REFERENCE_PROFILE_NORMAL, REFERENCE_PROFILE_LOW_LATENCY)
    }


def causal_history_stream_contract() -> dict[str, Any]:
    """Mirror training's immutable q0..q10 causal-history contract."""

    body = {
        "schema": "g1_true23_causal_history_profile_v1",
        "profile": CAUSAL_HISTORY_PROFILE,
        "source_sample_rate_hz": 50,
        "source_sample_period_s": 0.02,
        "architecture_initialization_profile": REFERENCE_PROFILE_LOW_LATENCY,
        "encoder_input_dim": 267,
        "lower_body_term_dim": 240,
        "lower_body_term_name": "causal_history_lower_body",
        "lower_body_il29_indices_in_encoder_order": list(CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES),
        "lower_body_order": "mujoco_hardware_left6_then_right6",
        "position_frame_count": CAUSAL_HISTORY_FRAME_COUNT,
        "position_order": "oldest_to_anchor",
        "position_offsets_from_anchor_s": [
            (index - (CAUSAL_HISTORY_FRAME_COUNT - 1)) * 0.02 for index in range(CAUSAL_HISTORY_FRAME_COUNT)
        ],
        "velocity_order": "oldest_to_anchor",
        "velocity_definition": ("forward_difference_q_i_to_q_i_plus_1_over_0p02s"),
        "anchor_frame": "q9",
        "proof_frame": "q10",
        "proof_frame_offset_from_anchor_s": 0.02,
        "anchor_age_at_emission_s": 0.02,
        "reference_channels_anchor": "q9",
        "reference_channels": [
            "lower_body_positions_and_velocities",
            "vr_3point_local_target",
            "vr_3point_local_orientation_target",
            "reference_pelvis_orientation",
            "buffered_robot_anchor_orientation",
            "reward_and_tracking_target",
        ],
        "control_and_proprioception_frame": "q10_current",
        "future_samples_relative_to_emission": False,
        "repeated_or_synthetic_future_frames": False,
        "released_profile_relabel_permitted": False,
        "retraining_required": True,
    }
    return {**body, "contract_sha256": _canonical_sha256(body)}


def soma_rolling_retarget_contract(iterations: int = 24) -> dict[str, Any]:
    if iterations != 24 and iterations not in EXPERIMENTAL_SOMA_ITERATION_PROFILES:
        raise ValueError("SOMA rolling iterations must be 24, 16, or 12")
    body = {
        "schema": "g1_true23_soma_rolling_retarget_contract_v1",
        "profile": (
            "pinned_soma_exact_ik24_v1" if iterations == 24 else EXPERIMENTAL_SOMA_ITERATION_PROFILES[iterations]
        ),
        "soma_commit": PINNED_SOMA_COMMIT,
        "pinned_config_sha256": dict(PINNED_CONFIG_SHA256),
        "ik_solver_iterations": iterations,
        "only_ik_iteration_count_changed": iterations != 24,
        "exact_ik24_equivalence_required": iterations == 24,
        "requires_sim_and_shadow_revalidation": iterations != 24,
        "promotion_eligible": False,
    }
    return {**body, "contract_sha256": _canonical_sha256(body)}


class IncrementalCaptureResampler:
    """Turn validated advancing raw frames into bracketed exact 50 Hz samples."""

    def __init__(self, capture_identity: Mapping[str, Any]):
        required = {"schema_version", "kind", "session_id", "source", "authorization"}
        if set(capture_identity) != required:
            raise ValueError("capture_identity must contain capture fields except frames")
        self._identity = {
            key: capture_identity[key]
            for key in (
                "schema_version",
                "kind",
                "session_id",
                "source",
                "authorization",
            )
        }
        self._previous: Mapping[str, Any] | None = None
        self._next_reference_ns: int | None = None
        self._source_frame_index = 0

    def push(self, frame: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Accept one measured frame; emit only timestamps bracketed by it."""

        if self._previous is None:
            self._previous = frame
            self._next_reference_ns = int(frame["capture_monotonic_ns"])
            return []
        pair = {**self._identity, "frames": [self._previous, frame]}
        validate_raw_capture(pair)
        left = self._previous
        right = frame
        left_ns = int(left["capture_monotonic_ns"])
        right_ns = int(right["capture_monotonic_ns"])
        assert self._next_reference_ns is not None
        if self._next_reference_ns < left_ns:
            raise ValueError("50 Hz phase fell behind newest raw bracket")

        emitted: list[dict[str, Any]] = []
        while self._next_reference_ns <= right_ns:
            reference_ns = self._next_reference_ns
            alpha = (reference_ns - left_ns) / (right_ns - left_ns)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("50 Hz reference timestamp lies outside raw bracket")
            poses = [
                _interpolate_pose(left_pose, right_pose, alpha)
                for left_pose, right_pose in zip(
                    left["body_poses"],
                    right["body_poses"],
                    strict=True,
                )
            ]
            emitted.append(
                {
                    "source_frame_index": self._source_frame_index,
                    "reference_monotonic_ns": reference_ns,
                    "capture_monotonic_ns": right_ns,
                    "raw_bracket_indices": [
                        int(left["frame_index"]),
                        int(right["frame_index"]),
                    ],
                    "raw_interpolation_alpha": alpha,
                    "body_poses": poses,
                }
            )
            self._source_frame_index += 1
            self._next_reference_ns += SOURCE_SAMPLE_PERIOD_NS
        self._previous = frame
        return emitted


class _SomaLocalFrameBuilder:
    def __init__(self, soma_source_root: Path):
        import numpy as np
        from soma_retargeter.assets.bvh import load_bvh

        neutral_path = soma_source_root / "soma_retargeter" / "configs" / "soma" / "soma_zero_frame0.bvh"
        self.skeleton, neutral_animation = load_bvh(str(neutral_path))
        reference_local = self.skeleton.reference_local_transforms.astype(np.float64)
        self._reference_global = _global_transforms(
            self.skeleton.parent_indices,
            reference_local.tolist(),
        )
        neutral_local = np.asarray(
            neutral_animation.local_transforms[0],
            dtype=np.float64,
        )
        self._target_neutral_global = _global_transforms(
            self.skeleton.parent_indices,
            neutral_local.tolist(),
        )
        self._calibration: dict[str, Any] | None = None
        self._soma_indices = {name: self.skeleton.joint_index(name) for _, name in XR24_ROLE_TO_SOMA_JOINT}
        if any(index < 0 for index in self._soma_indices.values()):
            raise RuntimeError("pinned SOMA skeleton lacks required XR24 mapping")

    def calibrate(
        self,
        neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]],
    ) -> dict[str, Any]:
        if self._calibration is not None:
            raise RuntimeError("rolling XR24 SOMA frame builder is already calibrated")
        self._calibration = build_xr24_soma_neutral_calibration(
            neutral_body_pose_frames,
            skeleton=self.skeleton,
            target_neutral_global=self._target_neutral_global,
        )
        return dict(self._calibration)

    def build(self, body_poses: Sequence[Sequence[float]]) -> Any:
        import numpy as np

        if self._calibration is None:
            raise RuntimeError("rolling XR24 SOMA frame builder is not calibrated")

        globals_for_frame = [list(transform) for transform in self._reference_global]
        for role_index, soma_name in XR24_ROLE_TO_SOMA_JOINT:
            globals_for_frame[self._soma_indices[soma_name]] = apply_xr24_soma_neutral_calibration(
                role_index,
                body_poses[role_index],
                self._calibration,
            )
        local = np.asarray(
            _local_transforms(
                self.skeleton.parent_indices,
                globals_for_frame,
            ),
            dtype=np.float32,
        )
        if local.shape != (self.skeleton.num_joints, 7) or not np.isfinite(local).all():
            raise RuntimeError("rolling XR24 -> SOMA local frame is invalid")
        return local


class _ExactSingleEnvironmentExecutor:
    """Persistent form of pinned ``NewtonPipeline.execute`` for one stream."""

    def __init__(self, pipeline: Any):
        import newton.ik as ik
        import numpy as np
        import warp as wp

        self._np = np
        self._wp = wp
        self._pipeline = pipeline
        self._model = pipeline._build_model(1)  # noqa: SLF001 - pinned API
        self._state = self._model.state()
        pipeline.feet_stabilizer.setup_num_envs(1)
        self._env_feet_tx = np.empty(
            (1, len(pipeline.feet_effector_indices), 7),
            dtype=np.float32,
        )
        (
            self._position_objectives,
            self._rotation_objectives,
            joint_limit_objective,
            self._smooth_joint_filter_objective,
        ) = pipeline._create_ik_objectives(  # noqa: SLF001 - pinned API
            1,
            self._model,
            self._state,
        )
        active_objectives = [
            *self._position_objectives,
            *self._rotation_objectives,
        ]
        if pipeline.joint_limit_weight > 0.0:
            active_objectives.append(joint_limit_objective)
        if pipeline.smooth_joint_filter_weight > 0.0:
            active_objectives.append(self._smooth_joint_filter_objective)
        self._ik_solver = ik.IKSolver(
            model=pipeline.ik_model,
            n_problems=1,
            objectives=active_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._joint_q = wp.empty(shape=(1, pipeline.ik_model.joint_coord_count))
        wp.copy(self._joint_q, self._model.joint_q)
        self._ik_solver.reset()

        def single_step() -> None:
            self._ik_solver.step(
                self._joint_q,
                self._joint_q,
                iterations=pipeline.ik_iterations,
            )

        self._single_step = single_step
        self._graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                single_step()
            self._graph = capture.graph
        else:
            single_step()

    def step(self, targets: Any, *, initialization_frame: int | None) -> Any:
        pipeline = self._pipeline
        wp = self._wp
        if initialization_frame is not None:
            denominator = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
            self._smooth_joint_filter_objective.set_weight(
                pipeline.smooth_joint_filter_weight * (initialization_frame / float(denominator))
            )
        for index, target in enumerate(targets):
            self._position_objectives[index].set_target_position(
                0,
                wp.vec3(*target[0:3]),
            )
            self._rotation_objectives[index].set_target_rotation(
                0,
                wp.quat(*target[3:7]),
            )
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._single_step()

        pipeline.feet_stabilizer.reset_state(self._joint_q)
        self._env_feet_tx[0] = self._np.asarray(targets)[pipeline.feet_effector_indices]
        pipeline.feet_stabilizer.solve(self._env_feet_tx)
        if pipeline.post_processing_enabled:
            data = pipeline.joint_limit_clamper.apply(pipeline.feet_stabilizer.current_state()).numpy()
        else:
            data = pipeline.joint_limit_clamper.apply(self._joint_q).numpy()
        return self._np.asarray(data[0], dtype=self._np.float64).reshape(-1)


class _True23BodyTermBuilder:
    """Persistent pinned-G1 FK for synchronized three-point reference terms."""

    def __init__(self, pipeline: Any):
        import newton
        import numpy as np
        import warp as wp

        self._newton = newton
        self._np = np
        self._wp = wp
        self._pipeline = pipeline
        self._model = pipeline._build_model(1)  # noqa: SLF001 - pinned API
        self._state = self._model.state()
        self._joint_qd = wp.zeros(self._model.joint_dof_count, dtype=wp.float32)
        leaf_names = [str(label).rsplit("/", 1)[-1] for label in pipeline.robot_builder.body_label]
        required = (
            "pelvis",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
            "torso_link",
        )
        if any(name not in leaf_names for name in required):
            raise RuntimeError("pinned G1 model lacks required reference body")
        self._pelvis_index, left, right, torso = (leaf_names.index(name) for name in required)
        self._torso_index = torso
        self._body_indices = (left, right, torso)
        self._offsets = (
            (0.18, -0.025, 0.0),
            (0.18, 0.025, 0.0),
            (0.0, 0.0, 0.35),
        )

    def build(
        self,
        root7_mj29: Sequence[float],
        timing: Mapping[str, Any],
    ) -> dict[str, Any]:
        q = self._np.asarray(root7_mj29, dtype=self._np.float32).copy()
        if q.shape != (SOURCE_DOF + 7,):
            raise RuntimeError("rolling body FK input is not root7 + MJ29")
        for source_index, canonical_index in enumerate(SOMA_MJ29_TO_CANONICAL_IL29):
            if canonical_index in SOURCE_IL29_EXCLUDED_INDICES:
                q[7 + source_index] = 0.0
        joint_q = self._wp.array(q, dtype=self._wp.float32)
        self._newton.eval_fk(
            self._model,
            joint_q,
            self._joint_qd,
            self._state,
        )
        body_q = self._np.asarray(
            self._state.body_q.numpy(),
            dtype=self._np.float64,
        ).reshape(self._pipeline.num_body_count, 7)
        pelvis = body_q[self._pelvis_index]
        torso = body_q[self._torso_index]
        pelvis_inverse = _quat_conjugate(pelvis[3:])
        positions: list[float] = []
        orientations: list[float] = []
        for body_index, offset in zip(
            self._body_indices,
            self._offsets,
            strict=True,
        ):
            body = body_q[body_index]
            point_offset = _quat_rotate(body[3:], offset)
            point = [body[axis] + point_offset[axis] for axis in range(3)]
            delta = [point[axis] - pelvis[axis] for axis in range(3)]
            positions.extend(_quat_rotate(pelvis_inverse, delta))
            orientations.extend(_quat_mul(pelvis_inverse, body[3:]))
        return {
            "source_frame_index": int(timing["source_frame_index"]),
            "reference_monotonic_ns": int(timing["reference_monotonic_ns"]),
            "capture_monotonic_ns": int(timing["capture_monotonic_ns"]),
            "vr_3point_local_target": positions,
            "vr_3point_local_orn_target": orientations,
            # Native124 policies declare torso_link as motion anchor.  Unitree
            # IMU quaternion is torso-mounted, so both sides of relative
            # orientation must use torso rather than pelvis.
            "reference_anchor_quaternion_xyzw": torso[3:].tolist(),
        }


class _PersistentSomaTargetBuilder:
    """Pinned scaler kernels with persistent Warp buffers and CUDA graph."""

    def __init__(self, pipeline: Any):
        import numpy as np
        from soma_retargeter.robotics.human_to_robot_scaler import (
            HumanToRobotScaler,
        )
        import soma_retargeter.utils.pose_utils as pose_utils
        import warp as wp

        scaler = pipeline.human_robot_scaler
        skeleton = scaler.skeleton
        self._np = np
        self._wp = wp
        self._joint_count = int(skeleton.num_joints)
        self._target_indices = list(pipeline.target_effector_indices)
        self._local_pose = wp.empty(shape=(1, self._joint_count), dtype=wp.transform)
        self._global_pose = wp.empty(shape=(1, self._joint_count), dtype=wp.transform)
        self._effectors = wp.empty(
            shape=(1, len(scaler.mapped_joint_indices)),
            dtype=wp.transform,
        )
        self._parent_indices = wp.array(skeleton.parent_indices, dtype=wp.int32)

        @wp.kernel
        def compute_global_pose_kernel(
            in_num_joints: wp.int32,
            in_parent_indices: wp.array(dtype=wp.int32),
            in_local_pose: wp.array2d(dtype=wp.transform),
            out_result: wp.array2d(dtype=wp.transform),
        ):
            frame_index = wp.tid()
            pose_utils.wp_compute_global_pose(
                in_num_joints,
                wp.transform_identity(),
                in_parent_indices,
                in_local_pose[frame_index],
                out_result[frame_index],
            )

        @wp.kernel
        def compute_scaled_effectors_kernel(
            in_num_mapped_joints: wp.int32,
            in_global_pose: wp.array2d(dtype=wp.transform),
            in_mapped_joint_indices: wp.array(dtype=wp.int32),
            in_mapped_joint_scales: wp.array(dtype=wp.float32),
            in_mapped_joint_offsets: wp.array(dtype=wp.transform),
            out_result: wp.array2d(dtype=wp.transform),
        ):
            frame_index = wp.tid()
            HumanToRobotScaler.wp_compute_scaled_effectors(
                in_num_mapped_joints,
                in_global_pose[frame_index],
                in_mapped_joint_indices,
                in_mapped_joint_scales,
                in_mapped_joint_offsets,
                True,
                out_result[frame_index],
            )

        def launch() -> None:
            wp.launch(
                compute_global_pose_kernel,
                dim=1,
                inputs=[
                    self._joint_count,
                    self._parent_indices,
                    self._local_pose,
                ],
                outputs=[self._global_pose],
            )
            wp.launch(
                compute_scaled_effectors_kernel,
                dim=1,
                inputs=[
                    len(scaler.mapped_joint_indices),
                    self._global_pose,
                    scaler.mapped_joint_indices,
                    scaler.mapped_joint_scales,
                    scaler.mapped_joint_offsets,
                ],
                outputs=[self._effectors],
            )

        self._launch = launch
        self._graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                launch()
            self._graph = capture.graph
        else:
            launch()

    def build(self, local_frame: Any) -> Any:
        local = self._np.asarray(local_frame, dtype=self._np.float32)
        if local.shape != (self._joint_count, 7):
            raise RuntimeError("persistent SOMA target frame shape changed")
        self._local_pose.assign(local[None, ...])
        if self._graph is not None:
            self._wp.capture_launch(self._graph)
        else:
            self._launch()
        return self._effectors.numpy()[0, self._target_indices, :]


class PinnedSomaRollingRetargeter:
    """Persistent exact-config SOMA solver with one output per 50 Hz sample."""

    def __init__(
        self,
        *,
        soma_source_root: Path,
        _solver_iterations: int = 24,
    ):
        if _solver_iterations != 24 and _solver_iterations not in (EXPERIMENTAL_SOMA_ITERATION_PROFILES):
            raise ValueError("unsupported SOMA solver iteration count")
        self.soma_source_root = soma_source_root.resolve()
        self.solver_iterations = _solver_iterations
        self.retarget_contract = soma_rolling_retarget_contract(_solver_iterations)
        self.runtime_report = _validate_pinned_soma_runtime(self.soma_source_root)
        self._frame_builder = _SomaLocalFrameBuilder(self.soma_source_root)
        self._pipeline: Any | None = None
        self._executor: _ExactSingleEnvironmentExecutor | None = None
        self._body_builder: _True23BodyTermBuilder | None = None
        self._target_builder: _PersistentSomaTargetBuilder | None = None
        self._last_frame_index: int | None = None
        self._last_reference_ns: int | None = None
        self._durations_ns: list[int] = []
        self._target_durations_ns: list[int] = []
        self._solver_durations_ns: list[int] = []
        self._body_term_durations_ns: list[int] = []
        self.joint_limit_verified_sample_count = 0
        self._joint_limit_lower: Any | None = None
        self._joint_limit_upper: Any | None = None
        self._joint_limit_dof_to_coord: Any | None = None

    def _verify_joint_limits(self, row: Any) -> None:
        assert self._pipeline is not None
        assert self._joint_limit_lower is not None
        assert self._joint_limit_upper is not None
        assert self._joint_limit_dof_to_coord is not None
        violations = 0
        for dof_index, coord_index in enumerate(self._joint_limit_dof_to_coord):
            coordinate = int(coord_index)
            if coordinate < 7:
                continue
            value = float(row[coordinate])
            minimum = float(self._joint_limit_lower[dof_index])
            maximum = float(self._joint_limit_upper[dof_index])
            if value < minimum - 1.0e-6 or value > maximum + 1.0e-6:
                violations += 1
        if violations:
            raise RuntimeError(f"SOMA post-processing left {violations} joint-limit violations")
        self.joint_limit_verified_sample_count += 1

    def _bootstrap(self, local_frame: Any) -> tuple[Any, int]:
        import numpy as np
        from soma_retargeter.animation.animation_buffer import AnimationBuffer
        from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
        import warp as wp

        pipeline = NewtonPipeline(self._frame_builder.skeleton)
        pipeline.ik_iterations = self.solver_iterations
        first = AnimationBuffer(
            self._frame_builder.skeleton,
            1,
            50.0,
            local_transforms=np.asarray(local_frame, dtype=np.float32)[None, ...],
        )
        pipeline.add_input_motions(
            [first],
            [wp.transform_identity()],
            scale_animation=True,
        )
        targets = pipeline.input_targets[0]
        expected = pipeline.num_initialization_frames + pipeline.num_stabilization_frames + 1
        if len(targets) != expected:
            raise RuntimeError("pinned SOMA initialization target count drift")
        executor = _ExactSingleEnvironmentExecutor(pipeline)
        row = None
        remove_count = expected - 1
        for frame_index, target in enumerate(targets):
            row = executor.step(
                target,
                initialization_frame=(frame_index if frame_index <= remove_count else None),
            )
        assert row is not None
        self._pipeline = pipeline
        self._executor = executor
        self._body_builder = _True23BodyTermBuilder(pipeline)
        self._target_builder = _PersistentSomaTargetBuilder(pipeline)
        clamper = pipeline.joint_limit_clamper
        self._joint_limit_lower = clamper.joint_limit_lower.numpy()
        self._joint_limit_upper = clamper.joint_limit_upper.numpy()
        self._joint_limit_dof_to_coord = clamper.dof_to_coord.numpy()
        return row, expected

    def _target_from_local_frame(self, local_frame: Any) -> Any:
        assert self._target_builder is not None
        return self._target_builder.build(local_frame)

    def calibrate(
        self,
        neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]],
    ) -> dict[str, Any]:
        """Bind native XR24 role frames before solver or stream starts."""

        if self._executor is not None or self._last_frame_index is not None:
            raise RuntimeError("rolling SOMA calibration must precede initialization")
        return self._frame_builder.calibrate(neutral_body_pose_frames)

    def prime(
        self,
        neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]],
    ) -> dict[str, Any]:
        """Initialize the exact solver before the live stream clock begins.

        The returned row is deliberately not a semantic sample: it has no
        fabricated frame index or timestamp.  The first subsequent ``push``
        remains frame zero at its real bracketed capture time.
        """

        if self._executor is not None or self._last_frame_index is not None:
            raise RuntimeError("rolling SOMA retargeter is already initialized")
        if len(neutral_body_pose_frames) != 10:
            raise ValueError("rolling SOMA prime requires exactly 10 neutral frames")
        standing_reports = [require_xr24_neutral_standing(body_poses) for body_poses in neutral_body_pose_frames]
        calibration = self.calibrate(neutral_body_pose_frames)
        body_poses = neutral_body_pose_frames[-1]
        local_frame = self._frame_builder.build(body_poses)
        started_ns = time.monotonic_ns()
        row, initialization_steps = self._bootstrap(local_frame)
        finished_ns = time.monotonic_ns()
        self._verify_joint_limits(row)
        return {
            "initialization_solver_steps": initialization_steps,
            "prime_duration_ns": finished_ns - started_ns,
            "joint_limits_verified": True,
            "neutral_standing_gate": {
                "pass": all(report["pass"] for report in standing_reports),
                "hold_frame_count": len(standing_reports),
            },
            "neutral_calibration_sha256": calibration["calibration_sha256"],
            "semantic_sample_emitted": False,
            "stream_clock_started": False,
        }

    def push(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Retarget one bracketed sample, preserving exact solver continuity."""

        frame_index = int(sample["source_frame_index"])
        reference_ns = int(sample["reference_monotonic_ns"])
        if self._last_frame_index is not None and (
            frame_index != self._last_frame_index + 1
            or reference_ns != self._last_reference_ns + SOURCE_SAMPLE_PERIOD_NS
        ):
            raise ValueError("rolling SOMA input is not contiguous exact 50 Hz")
        local_frame = self._frame_builder.build(sample["body_poses"])
        started_ns = time.monotonic_ns()
        initialization_steps = 0
        if self._executor is None:
            row, initialization_steps = self._bootstrap(local_frame)
            target_finished_ns = started_ns
            solver_started_ns = started_ns
        else:
            targets = self._target_from_local_frame(local_frame)
            target_finished_ns = time.monotonic_ns()
            solver_started_ns = target_finished_ns
            row = self._executor.step(targets, initialization_frame=None)
        solver_finished_ns = time.monotonic_ns()
        if row.shape != (SOURCE_DOF + 7,):
            raise RuntimeError("rolling SOMA output is not root7 + MJ29")
        if not all(math.isfinite(float(value)) for value in row):
            raise RuntimeError("rolling SOMA output contains non-finite values")
        self._verify_joint_limits(row)
        assert self._body_builder is not None
        body_term = self._body_builder.build(row, sample)
        finished_ns = time.monotonic_ns()
        duration_ns = finished_ns - started_ns
        self._durations_ns.append(duration_ns)
        self._target_durations_ns.append(target_finished_ns - started_ns)
        self._solver_durations_ns.append(solver_finished_ns - solver_started_ns)
        self._body_term_durations_ns.append(finished_ns - solver_finished_ns)
        self._last_frame_index = frame_index
        self._last_reference_ns = reference_ns
        mj29 = [float(value) for value in row[7:]]
        il29 = [0.0] * SOURCE_DOF
        for source_index, target_index in enumerate(SOMA_MJ29_TO_CANONICAL_IL29):
            il29[target_index] = mj29[source_index]
        return {
            "schema_version": ROLLING_SOMA_SCHEMA_VERSION,
            "kind": ROLLING_SOMA_KIND,
            "source_frame_index": frame_index,
            "reference_monotonic_ns": reference_ns,
            "capture_monotonic_ns": int(sample["capture_monotonic_ns"]),
            "raw_bracket_indices": list(sample["raw_bracket_indices"]),
            "raw_interpolation_alpha": _finite(
                sample["raw_interpolation_alpha"],
                "raw_interpolation_alpha",
            ),
            "joint_root7_mj29": [float(value) for value in row],
            "joint_pos_mj29": mj29,
            "joint_pos_il29": il29,
            "compute_started_monotonic_ns": started_ns,
            "compute_finished_monotonic_ns": finished_ns,
            "compute_duration_ns": duration_ns,
            "target_build_duration_ns": target_finished_ns - started_ns,
            "solver_duration_ns": solver_finished_ns - solver_started_ns,
            "body_term_duration_ns": finished_ns - solver_finished_ns,
            "initialization_solver_steps": initialization_steps,
            "body_term": body_term,
            "producer_sha256": _module_sha256(),
            "retarget_contract_sha256": self.retarget_contract["contract_sha256"],
            "status": ROLLING_SOMA_STATUS,
            "promotion_eligible": False,
        }

    def timing_summary(self, *, exclude_bootstrap: bool = True) -> dict[str, Any]:
        """Return steady solver timing without claiming deadline success."""

        import numpy as np

        values = self._durations_ns[1:] if exclude_bootstrap else self._durations_ns
        if not values:
            raise ValueError("no rolling steady-state timing samples")
        array = np.asarray(values, dtype=np.float64)
        p50 = float(np.percentile(array, 50))
        p95 = float(np.percentile(array, 95))
        p99 = float(np.percentile(array, 99))
        maximum = float(np.max(array))
        mean = float(np.mean(array))
        target_array = np.asarray(
            self._target_durations_ns[1:] if exclude_bootstrap else self._target_durations_ns,
            dtype=np.float64,
        )
        solver_array = np.asarray(
            self._solver_durations_ns[1:] if exclude_bootstrap else self._solver_durations_ns,
            dtype=np.float64,
        )
        body_array = np.asarray(
            self._body_term_durations_ns[1:] if exclude_bootstrap else self._body_term_durations_ns,
            dtype=np.float64,
        )
        simulated_finish_ns = 0.0
        queue_response_ns: list[float] = []
        queue_backlog_ns: list[float] = []
        for index, duration_ns in enumerate(values):
            arrival_ns = float(index * SOURCE_SAMPLE_PERIOD_NS)
            simulated_finish_ns = max(arrival_ns, simulated_finish_ns) + float(duration_ns)
            response_ns = simulated_finish_ns - arrival_ns
            queue_response_ns.append(response_ns)
            queue_backlog_ns.append(max(0.0, response_ns - SOURCE_SAMPLE_PERIOD_NS))
        response_array = np.asarray(queue_response_ns, dtype=np.float64)
        backlog_array = np.asarray(queue_backlog_ns, dtype=np.float64)
        return {
            "sample_count": len(values),
            "deadline_ns": SOURCE_SAMPLE_PERIOD_NS,
            "mean_ns": mean,
            "p50_ns": p50,
            "p95_ns": p95,
            "p99_ns": p99,
            "max_ns": maximum,
            "mean_fps": 1_000_000_000.0 / mean,
            "target_build_mean_ns": float(np.mean(target_array)),
            "target_build_p99_ns": float(np.percentile(target_array, 99)),
            "solver_mean_ns": float(np.mean(solver_array)),
            "solver_p99_ns": float(np.percentile(solver_array, 99)),
            "body_term_mean_ns": float(np.mean(body_array)),
            "body_term_p99_ns": float(np.percentile(body_array, 99)),
            "deadline_met_every_sample": maximum <= SOURCE_SAMPLE_PERIOD_NS,
            "p99_deadline_met": p99 <= SOURCE_SAMPLE_PERIOD_NS,
            "single_worker_queue_model": {
                "arrival_period_ns": SOURCE_SAMPLE_PERIOD_NS,
                "response_mean_ns": float(np.mean(response_array)),
                "response_p95_ns": float(np.percentile(response_array, 95)),
                "response_p99_ns": float(np.percentile(response_array, 99)),
                "response_max_ns": float(np.max(response_array)),
                "max_backlog_ns": float(np.max(backlog_array)),
                "on_time_fraction": float(np.mean(response_array <= SOURCE_SAMPLE_PERIOD_NS)),
                "queue_cleared_by_end": bool(backlog_array[-1] == 0.0),
            },
            "promotion_eligible": False,
        }


class ExperimentalSomaRollingRetargeter(PinnedSomaRollingRetargeter):
    """Distinct non-promotable IK-iteration experiment; never pinned exact."""

    def __init__(self, *, soma_source_root: Path, solver_iterations: int):
        if solver_iterations not in EXPERIMENTAL_SOMA_ITERATION_PROFILES:
            raise ValueError("experimental solver iterations must be 16 or 12")
        super().__init__(
            soma_source_root=soma_source_root,
            _solver_iterations=solver_iterations,
        )


class RollingSemanticProducer:
    """Derive exact q/dq frames and honest delayed windows from rolling SOMA."""

    def __init__(self, *, source_session_id: str, profile: str):
        if not source_session_id:
            raise ValueError("source_session_id must be non-empty")
        if profile not in REFERENCE_PROFILES:
            raise ValueError(f"unsupported reference profile: {profile}")
        self.source_session_id = source_session_id
        self.profile = profile
        self._needed = required_buffer_frames(profile)
        self._frames: deque[dict[str, Any]] = deque(maxlen=self._needed)
        self._body_terms: deque[dict[str, Any]] = deque(maxlen=self._needed)
        self._previous: Mapping[str, Any] | None = None

    def push(self, rolling: Mapping[str, Any]) -> dict[str, Any] | None:
        """Emit previous q only after current q proves forward velocity."""

        if rolling.get("kind") != ROLLING_SOMA_KIND:
            raise ValueError("rolling SOMA sample kind mismatch")
        if rolling.get("promotion_eligible") is not False:
            raise ValueError("rolling SOMA sample must remain non-promotable")
        if self._previous is None:
            self._previous = rolling
            return None
        previous = self._previous
        if (
            int(rolling["source_frame_index"]) != int(previous["source_frame_index"]) + 1
            or int(rolling["reference_monotonic_ns"])
            != int(previous["reference_monotonic_ns"]) + SOURCE_SAMPLE_PERIOD_NS
        ):
            raise ValueError("rolling semantic input is not contiguous 50 Hz")
        current_q = list(rolling["joint_pos_il29"])
        previous_q = list(previous["joint_pos_il29"])
        if len(current_q) != SOURCE_DOF or len(previous_q) != SOURCE_DOF:
            raise ValueError("rolling semantic input is not complete IL29")
        velocity = [
            (float(current) - float(prior)) * 50.0 for prior, current in zip(previous_q, current_q, strict=True)
        ]
        frame = {
            "schema_version": SEMANTIC_REFERENCE_SCHEMA_VERSION,
            "kind": SEMANTIC_REFERENCE_FRAME_KIND,
            "source_kind": SOURCE_RETARGETED_DELAYED,
            "source_session_id": self.source_session_id,
            "producer_sha256": str(previous["producer_sha256"]),
            "source_frame_index": int(previous["source_frame_index"]),
            "reference_monotonic_ns": int(previous["reference_monotonic_ns"]),
            "capture_monotonic_ns": int(previous["capture_monotonic_ns"]),
            "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "joint_order": JOINT_ORDER,
            "complete_joint_mask_il29": [True] * SOURCE_DOF,
            "joint_values_semantics": "full_reference_no_fill",
            "velocity_semantics": "source_50hz_forward_difference",
            "temporal_semantics": "measured_delayed_reference",
            "joint_pos_il29": [float(value) for value in previous_q],
            "joint_vel_il29": velocity,
        }
        validate_semantic_reference_frame(frame)
        body_term = dict(previous["body_term"])
        if (
            body_term["source_frame_index"] != frame["source_frame_index"]
            or body_term["reference_monotonic_ns"] != frame["reference_monotonic_ns"]
        ):
            raise RuntimeError("rolling body terms are not semantic-frame aligned")
        self._frames.append(frame)
        self._body_terms.append(body_term)
        produced_ns = int(rolling["compute_finished_monotonic_ns"])
        result: dict[str, Any] = {
            "semantic_frame": frame,
            "body_term": body_term,
            "velocity_proof_source_frame_index": int(rolling["source_frame_index"]),
            "velocity_proof_reference_monotonic_ns": int(rolling["reference_monotonic_ns"]),
            "produced_monotonic_ns": produced_ns,
            "reference_window": None,
            "measured_delay_ns": None,
            "promotion_eligible": False,
        }
        if len(self._frames) == self._needed:
            window = build_stream_reference_window(
                list(self._frames),
                profile=self.profile,
                emitted_monotonic_ns=produced_ns,
            )
            playback_index = int(window["playback"]["frame_index"])
            aligned_body = next(
                term for term in self._body_terms if int(term["source_frame_index"]) == playback_index
            )
            result["body_term"] = aligned_body
            result["reference_window"] = window
            result["measured_delay_ns"] = int(window["playback"]["emission_lag_ns"])
        self._previous = rolling
        return result


class CausalHistorySemanticProducer:
    """Build exact q0..q9 history with q10 forward-difference proof."""

    def __init__(self, *, source_session_id: str):
        if not source_session_id:
            raise ValueError("source_session_id must be non-empty")
        self.source_session_id = source_session_id
        self._rows: deque[dict[str, Any]] = deque(maxlen=CAUSAL_HISTORY_PROOF_FRAME_COUNT)
        self._last_frame_index: int | None = None
        self._last_reference_ns: int | None = None
        self._producer_sha256: str | None = None

    def push(self, rolling: Mapping[str, Any]) -> dict[str, Any] | None:
        """Accept one measured SOMA q; emit after 11 contiguous q samples."""

        if rolling.get("kind") != ROLLING_SOMA_KIND:
            raise ValueError("causal-history input kind mismatch")
        if rolling.get("promotion_eligible") is not False:
            raise ValueError("causal-history input must remain non-promotable")
        frame_index = int(rolling["source_frame_index"])
        reference_ns = int(rolling["reference_monotonic_ns"])
        if self._last_frame_index is not None and (
            frame_index != self._last_frame_index + 1
            or reference_ns != self._last_reference_ns + SOURCE_SAMPLE_PERIOD_NS
        ):
            raise ValueError("causal-history input is not contiguous exact 50 Hz")
        joint_pos = [_finite(value, "causal-history joint_pos_il29") for value in rolling["joint_pos_il29"]]
        if len(joint_pos) != SOURCE_DOF:
            raise ValueError("causal-history input is not complete IL29")
        producer_sha256 = str(rolling["producer_sha256"])
        if self._producer_sha256 is None:
            self._producer_sha256 = producer_sha256
        elif producer_sha256 != self._producer_sha256:
            raise ValueError("causal-history SOMA producer changed mid-stream")
        body_term = dict(rolling["body_term"])
        if (
            int(body_term["source_frame_index"]) != frame_index
            or int(body_term["reference_monotonic_ns"]) != reference_ns
        ):
            raise ValueError("causal-history body term is not q-aligned")
        row = dict(rolling)
        row["joint_pos_il29"] = joint_pos
        row["body_term"] = body_term
        self._rows.append(row)
        self._last_frame_index = frame_index
        self._last_reference_ns = reference_ns
        if len(self._rows) < CAUSAL_HISTORY_PROOF_FRAME_COUNT:
            return None

        rows = list(self._rows)
        # Training selects the first 12 *MuJoCo/hardware* joints, represented
        # here in canonical IL29 order.  Canonical IL29[:12] is not lower
        # body: it includes waist roll/pitch and left shoulder pitch.
        positions = [
            [row["joint_pos_il29"][index] for index in CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES] for row in rows[:-1]
        ]
        next_positions = [
            [row["joint_pos_il29"][index] for index in CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES] for row in rows[1:]
        ]
        velocities = [
            [
                (next_value - value) * (1_000_000_000.0 / SOURCE_SAMPLE_PERIOD_NS)
                for value, next_value in zip(
                    position,
                    next_position,
                    strict=True,
                )
            ]
            for position, next_position in zip(
                positions,
                next_positions,
                strict=True,
            )
        ]
        lower_body = [value for row in positions for value in row] + [value for row in velocities for value in row]
        flattened_positions = lower_body[:120]
        proof_lower_body_position = [
            rows[-1]["joint_pos_il29"][index] for index in CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES
        ]
        if len(lower_body) != 240 or not all(math.isfinite(value) for value in lower_body):
            raise RuntimeError("causal-history lower-body tensor is invalid")
        anchor = rows[-2]
        proof = rows[-1]
        q_ref23_native = [float(anchor["joint_pos_il29"][index]) for index in NATIVE_IL23_TO_CANONICAL_IL29]
        qd_ref23_native = [
            (float(proof["joint_pos_il29"][index]) - float(anchor["joint_pos_il29"][index])) * 50.0
            for index in NATIVE_IL23_TO_CANONICAL_IL29
        ]
        anchor_body_term = dict(anchor["body_term"])
        contract = causal_history_stream_contract()
        return {
            "schema_version": CAUSAL_HISTORY_PACKET_SCHEMA_VERSION,
            "kind": CAUSAL_HISTORY_PACKET_KIND,
            "profile": CAUSAL_HISTORY_PROFILE,
            "contract_sha256": contract["contract_sha256"],
            "source_session_id": self.source_session_id,
            "soma_producer_sha256": self._producer_sha256,
            "position_source_frame_indices": [int(row["source_frame_index"]) for row in rows[:-1]],
            "velocity_proof_source_frame_indices": [int(row["source_frame_index"]) for row in rows[1:]],
            "position_reference_monotonic_ns": [int(row["reference_monotonic_ns"]) for row in rows[:-1]],
            "velocity_proof_reference_monotonic_ns": [int(row["reference_monotonic_ns"]) for row in rows[1:]],
            "anchor_source_frame_index": int(anchor["source_frame_index"]),
            "anchor_reference_monotonic_ns": int(anchor["reference_monotonic_ns"]),
            "proof_source_frame_index": int(proof["source_frame_index"]),
            "proof_reference_monotonic_ns": int(proof["reference_monotonic_ns"]),
            "proof_capture_monotonic_ns": int(proof["capture_monotonic_ns"]),
            "produced_monotonic_ns": int(proof["compute_finished_monotonic_ns"]),
            "intrinsic_measurement_delay_ns": (
                int(proof["reference_monotonic_ns"]) - int(anchor["reference_monotonic_ns"])
            ),
            "control_derivative_contract": (CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT),
            "sdk_derivatives_consumed": False,
            "positions_repeated_or_synthesized": False,
            "causal_history_positions_lower_body": flattened_positions,
            "proof_lower_body_position": proof_lower_body_position,
            "causal_history_lower_body": lower_body,
            "anchor_joint_pos_il29": list(anchor["joint_pos_il29"]),
            "proof_joint_pos_il29": list(proof["joint_pos_il29"]),
            "q_ref23_native": q_ref23_native,
            "qd_ref23_native": qd_ref23_native,
            "anchor_body_term": anchor_body_term,
            "control_and_proprioception_frame": "q10_current",
            "promotion_eligible": False,
        }


def validate_causal_history_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate causal namespace and recompute every q0..q10 derivative."""

    required = {
        "schema_version",
        "kind",
        "profile",
        "contract_sha256",
        "source_session_id",
        "soma_producer_sha256",
        "position_source_frame_indices",
        "velocity_proof_source_frame_indices",
        "position_reference_monotonic_ns",
        "velocity_proof_reference_monotonic_ns",
        "anchor_source_frame_index",
        "anchor_reference_monotonic_ns",
        "proof_source_frame_index",
        "proof_reference_monotonic_ns",
        "proof_capture_monotonic_ns",
        "produced_monotonic_ns",
        "intrinsic_measurement_delay_ns",
        "control_derivative_contract",
        "sdk_derivatives_consumed",
        "positions_repeated_or_synthesized",
        "causal_history_positions_lower_body",
        "proof_lower_body_position",
        "causal_history_lower_body",
        "anchor_joint_pos_il29",
        "proof_joint_pos_il29",
        "q_ref23_native",
        "qd_ref23_native",
        "anchor_body_term",
        "control_and_proprioception_frame",
        "promotion_eligible",
    }
    if not isinstance(packet, Mapping) or set(packet) != required:
        raise ValueError("causal-history packet keys mismatch")
    contract = causal_history_stream_contract()
    if (
        packet["schema_version"] != CAUSAL_HISTORY_PACKET_SCHEMA_VERSION
        or packet["kind"] != CAUSAL_HISTORY_PACKET_KIND
        or packet["profile"] != CAUSAL_HISTORY_PROFILE
        or packet["contract_sha256"] != contract["contract_sha256"]
    ):
        raise ValueError("causal-history packet contract mismatch")
    if (
        packet["control_derivative_contract"] != CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT
        or packet["sdk_derivatives_consumed"] is not False
        or packet["positions_repeated_or_synthesized"] is not False
        or packet["control_and_proprioception_frame"] != "q10_current"
        or packet["promotion_eligible"] is not False
    ):
        raise ValueError("causal-history packet safety semantics mismatch")
    position_indices = [int(value) for value in packet["position_source_frame_indices"]]
    proof_indices = [int(value) for value in packet["velocity_proof_source_frame_indices"]]
    if (
        len(position_indices) != CAUSAL_HISTORY_FRAME_COUNT
        or proof_indices != [value + 1 for value in position_indices]
        or position_indices != list(range(position_indices[0], position_indices[0] + 10))
        or int(packet["anchor_source_frame_index"]) != position_indices[-1]
        or int(packet["proof_source_frame_index"]) != proof_indices[-1]
    ):
        raise ValueError("causal-history q0..q10 frame proof mismatch")
    position_times = [int(value) for value in packet["position_reference_monotonic_ns"]]
    proof_times = [int(value) for value in packet["velocity_proof_reference_monotonic_ns"]]
    if (
        len(position_times) != CAUSAL_HISTORY_FRAME_COUNT
        or proof_times != [value + SOURCE_SAMPLE_PERIOD_NS for value in position_times]
        or any(right - left != SOURCE_SAMPLE_PERIOD_NS for left, right in zip(position_times, position_times[1:]))
        or int(packet["anchor_reference_monotonic_ns"]) != position_times[-1]
        or int(packet["proof_reference_monotonic_ns"]) != proof_times[-1]
        or int(packet["intrinsic_measurement_delay_ns"]) != SOURCE_SAMPLE_PERIOD_NS
    ):
        raise ValueError("causal-history q0..q10 timestamp proof mismatch")
    positions = [
        _finite(value, "causal-history position proof") for value in packet["causal_history_positions_lower_body"]
    ]
    proof_position = [
        _finite(value, "causal-history q10 position proof") for value in packet["proof_lower_body_position"]
    ]
    lower_body = [
        _finite(value, "causal-history lower-body tensor") for value in packet["causal_history_lower_body"]
    ]
    if len(positions) != 120 or len(proof_position) != 12 or len(lower_body) != 240:
        raise ValueError("causal-history lower-body dimensions mismatch")
    position_rows = [positions[index : index + 12] for index in range(0, 120, 12)]
    proof_rows = [*position_rows[1:], proof_position]
    expected_velocities = [
        (next_value - value) * 50.0
        for row, next_row in zip(position_rows, proof_rows, strict=True)
        for value, next_value in zip(row, next_row, strict=True)
    ]
    expected = positions + expected_velocities
    if any(
        not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1.0e-10)
        for value, expected_value in zip(lower_body, expected, strict=True)
    ):
        raise ValueError("causal-history velocity is not q0..q10 forward difference")
    q_ref23 = [_finite(value, "causal-history q_ref23_native") for value in packet["q_ref23_native"]]
    qd_ref23 = [_finite(value, "causal-history qd_ref23_native") for value in packet["qd_ref23_native"]]
    if len(q_ref23) != 23 or len(qd_ref23) != 23:
        raise ValueError("causal-history native23 reference dimensions mismatch")
    anchor_il29 = [
        _finite(value, "causal-history anchor_joint_pos_il29") for value in packet["anchor_joint_pos_il29"]
    ]
    proof_il29 = [
        _finite(value, "causal-history proof_joint_pos_il29") for value in packet["proof_joint_pos_il29"]
    ]
    if len(anchor_il29) != 29 or len(proof_il29) != 29:
        raise ValueError("causal-history IL29 proof dimensions mismatch")
    if any(
        not math.isclose(value, anchor_il29[index], rel_tol=0.0, abs_tol=1.0e-10)
        for value, index in zip(
            position_rows[-1],
            CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
            strict=True,
        )
    ) or any(
        not math.isclose(value, proof_il29[index], rel_tol=0.0, abs_tol=1.0e-10)
        for value, index in zip(
            proof_position,
            CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
            strict=True,
        )
    ):
        raise ValueError("causal-history lower-body selector/order differs from encoder ABI")
    expected_q_ref23 = [anchor_il29[index] for index in NATIVE_IL23_TO_CANONICAL_IL29]
    expected_qd_ref23 = [
        (proof_il29[index] - anchor_il29[index]) * 50.0 for index in NATIVE_IL23_TO_CANONICAL_IL29
    ]
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-7)
        for value, expected in zip(q_ref23, expected_q_ref23, strict=True)
    ) or any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-6)
        for value, expected in zip(qd_ref23, expected_qd_ref23, strict=True)
    ):
        raise ValueError("causal-history native23 reference is not bound to q9/q10")
    body = packet["anchor_body_term"]
    if not isinstance(body, Mapping) or (
        int(body["source_frame_index"]) != position_indices[-1]
        or int(body["reference_monotonic_ns"]) != position_times[-1]
    ):
        raise ValueError("causal-history body terms are not q9-aligned")
    for name, size in (
        ("vr_3point_local_target", 9),
        ("vr_3point_local_orn_target", 12),
        ("reference_anchor_quaternion_xyzw", 4),
    ):
        values = [_finite(value, f"anchor_body_term.{name}") for value in body[name]]
        if len(values) != size:
            raise ValueError(f"anchor_body_term.{name} dimension mismatch")
    return {
        "profile": CAUSAL_HISTORY_PROFILE,
        "anchor_source_frame_index": position_indices[-1],
        "proof_source_frame_index": proof_indices[-1],
        "encoder_lower_body_dim": len(lower_body),
        "sdk_derivatives_consumed": False,
        "sha256": _canonical_sha256(packet),
        "promotion_eligible": False,
    }


def complete_causal_history_encoder_packet(
    *,
    semantic_packet: Mapping[str, Any],
    robot_anchor_quaternion_wxyz: Sequence[float],
    robot_anchor_monotonic_ns: int,
    robot_anchor_source_contract: str,
) -> dict[str, Any]:
    """Bind q9 body/robot anchors while control and proprioception stay q10."""

    validate_causal_history_packet(semantic_packet)
    anchor_ns = int(semantic_packet["anchor_reference_monotonic_ns"])
    if (
        isinstance(robot_anchor_monotonic_ns, bool)
        or not isinstance(robot_anchor_monotonic_ns, int)
        or robot_anchor_monotonic_ns != anchor_ns
    ):
        raise ValueError("robot anchor must be interpolated at exact causal q9")
    if robot_anchor_source_contract != CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT:
        raise ValueError("causal robot anchor source contract mismatch")
    if len(robot_anchor_quaternion_wxyz) != 4:
        raise ValueError("robot anchor quaternion must be w,x,y,z")
    robot_xyzw = _normalize_quaternion(
        (
            robot_anchor_quaternion_wxyz[1],
            robot_anchor_quaternion_wxyz[2],
            robot_anchor_quaternion_wxyz[3],
            robot_anchor_quaternion_wxyz[0],
        )
    )
    body = semantic_packet["anchor_body_term"]
    reference_xyzw = _normalize_quaternion(body["reference_anchor_quaternion_xyzw"])
    relative = _quat_mul(_quat_conjugate(robot_xyzw), reference_xyzw)
    motion_anchor_ori_b = _rotation_6d_xyzw(relative)
    lower_body = list(semantic_packet["causal_history_lower_body"])
    encoder_input = [
        *lower_body,
        *body["vr_3point_local_target"],
        *body["vr_3point_local_orn_target"],
        *motion_anchor_ori_b,
    ]
    if len(encoder_input) != 267 or not all(math.isfinite(float(value)) for value in encoder_input):
        raise RuntimeError("causal-history encoder input is invalid")
    terms = {
        "schema_version": CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION,
        "kind": CAUSAL_HISTORY_ENCODER_TERMS_KIND,
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "reference_contract_sha256": semantic_packet["contract_sha256"],
        "pico_anchor_source_frame_index": int(semantic_packet["anchor_source_frame_index"]),
        "pico_anchor_monotonic_ns": anchor_ns,
        "robot_anchor_monotonic_ns": robot_anchor_monotonic_ns,
        "robot_anchor_source_contract": robot_anchor_source_contract,
        "control_source_frame_index": int(semantic_packet["proof_source_frame_index"]),
        "control_monotonic_ns": int(semantic_packet["proof_reference_monotonic_ns"]),
        "causal_history_lower_body": lower_body,
        "vr_3point_local_target": list(body["vr_3point_local_target"]),
        "vr_3point_local_orn_target": list(body["vr_3point_local_orn_target"]),
        "motion_anchor_ori_b": motion_anchor_ori_b,
        "control_derivative_contract": (CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT),
        "sdk_derivatives_consumed": False,
    }
    return {
        "encoder_terms": terms,
        "encoder_input": encoder_input,
        "encoder_input_sha256": _canonical_sha256(encoder_input),
        "semantic_packet_sha256": _canonical_sha256(semantic_packet),
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "anchor_frame": "q9",
        "control_and_proprioception_frame": "q10_current",
        "promotion_eligible": False,
    }


def causal_history_reference_terms(
    semantic_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Create robot-independent transport; C++ joins q9/q10 LowState."""

    validate_causal_history_packet(semantic_packet)
    body = semantic_packet["anchor_body_term"]
    return {
        "schema_version": CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION,
        "kind": CAUSAL_HISTORY_REFERENCE_TERMS_KIND,
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "reference_contract_sha256": semantic_packet["contract_sha256"],
        "pico_anchor_source_frame_index": int(semantic_packet["anchor_source_frame_index"]),
        "pico_anchor_monotonic_ns": int(semantic_packet["anchor_reference_monotonic_ns"]),
        "control_source_frame_index": int(semantic_packet["proof_source_frame_index"]),
        "control_monotonic_ns": int(semantic_packet["proof_reference_monotonic_ns"]),
        "causal_history_lower_body": list(semantic_packet["causal_history_lower_body"]),
        "vr_3point_local_target": list(body["vr_3point_local_target"]),
        "vr_3point_local_orn_target": list(body["vr_3point_local_orn_target"]),
        "reference_anchor_quaternion_xyzw": list(body["reference_anchor_quaternion_xyzw"]),
        "anchor_joint_pos_il29": list(semantic_packet["anchor_joint_pos_il29"]),
        "proof_joint_pos_il29": list(semantic_packet["proof_joint_pos_il29"]),
        "q_ref23_native": list(semantic_packet["q_ref23_native"]),
        "qd_ref23_native": list(semantic_packet["qd_ref23_native"]),
        "control_derivative_contract": (CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT),
        "sdk_derivatives_consumed": False,
    }


def complete_profile_bound_encoder_packet(
    *,
    semantic_packet: Mapping[str, Any],
    expected_profile: str,
    robot_anchor_quaternion_wxyz: Sequence[float],
    robot_anchor_monotonic_ns: int,
) -> dict[str, Any]:
    """Complete 267 terms only with an exactly synchronized robot anchor."""

    from gear_sonic.utils.g1_23dof_live_shadow import (
        ENCODER_TERMS_KIND,
        ENCODER_TERMS_SCHEMA_VERSION,
    )

    window = semantic_packet.get("reference_window")
    if not isinstance(window, Mapping):
        raise ValueError("semantic packet has no complete reference window")
    summary = validate_semantic_reference_window(window)
    if summary["profile"] != expected_profile:
        raise ValueError("semantic window profile disagrees with artifact profile")
    body = semantic_packet["body_term"]
    anchor_ns = int(window["playback"]["frame_monotonic_ns"])
    if int(body["reference_monotonic_ns"]) != anchor_ns or int(body["source_frame_index"]) != int(
        window["playback"]["frame_index"]
    ):
        raise ValueError("body terms are not aligned to semantic playback anchor")
    if (
        isinstance(robot_anchor_monotonic_ns, bool)
        or not isinstance(robot_anchor_monotonic_ns, int)
        or robot_anchor_monotonic_ns != anchor_ns
    ):
        raise ValueError("robot anchor must be interpolated at exact semantic anchor")
    if len(robot_anchor_quaternion_wxyz) != 4:
        raise ValueError("robot anchor quaternion must be w,x,y,z")
    robot_xyzw = _normalize_quaternion(
        (
            robot_anchor_quaternion_wxyz[1],
            robot_anchor_quaternion_wxyz[2],
            robot_anchor_quaternion_wxyz[3],
            robot_anchor_quaternion_wxyz[0],
        )
    )
    reference_xyzw = _normalize_quaternion(body["reference_anchor_quaternion_xyzw"])
    relative = _quat_mul(_quat_conjugate(robot_xyzw), reference_xyzw)
    terms = {
        "schema_version": ENCODER_TERMS_SCHEMA_VERSION,
        "kind": ENCODER_TERMS_KIND,
        "pico_source_frame_index": int(body["source_frame_index"]),
        "pico_source_monotonic_ns": anchor_ns,
        "future_frame_offsets_s": list(window["future_frame_offsets_s"]),
        "command_multi_future_lower_body": list(window["command_multi_future_lower_body"]),
        "vr_3point_local_target": list(body["vr_3point_local_target"]),
        "vr_3point_local_orn_target": list(body["vr_3point_local_orn_target"]),
        "motion_anchor_ori_b": _rotation_6d_xyzw(relative),
    }
    return {
        "encoder_terms": terms,
        "semantic_anchor_monotonic_ns": anchor_ns,
        "robot_anchor_monotonic_ns": robot_anchor_monotonic_ns,
        "produced_monotonic_ns": int(semantic_packet["produced_monotonic_ns"]),
        "reference_profile": expected_profile,
        "promotion_eligible": False,
    }


def compare_rolling_to_batch(
    rolling_rows: Sequence[Mapping[str, Any]],
    batch_root7_mj29: Sequence[Sequence[float]],
    *,
    absolute_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare rolling output against same-capture pinned batch output."""

    if len(rolling_rows) != len(batch_root7_mj29):
        raise ValueError("rolling and batch sample counts differ")
    if absolute_tolerance < 0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative")
    differences = [
        abs(float(rolling) - float(batch))
        for rolling_row, batch_row in zip(
            rolling_rows,
            batch_root7_mj29,
            strict=True,
        )
        for rolling, batch in zip(
            rolling_row["joint_root7_mj29"],
            batch_row,
            strict=True,
        )
    ]
    maximum = max(differences, default=0.0)
    return {
        "sample_count": len(rolling_rows),
        "value_count": len(differences),
        "absolute_tolerance": absolute_tolerance,
        "max_abs_error": maximum,
        "exact_within_tolerance": maximum <= absolute_tolerance,
        "rolling_sha256": _canonical_sha256([row["joint_root7_mj29"] for row in rolling_rows]),
        "batch_sha256": _canonical_sha256(batch_root7_mj29),
        "promotion_eligible": False,
    }


__all__ = [
    "ACTIVE_POLICY_FRESHNESS_NS",
    "CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT",
    "CAUSAL_HISTORY_ENCODER_TERMS_KIND",
    "CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION",
    "CAUSAL_HISTORY_PACKET_KIND",
    "CAUSAL_HISTORY_PROFILE",
    "CAUSAL_HISTORY_REFERENCE_TERMS_KIND",
    "CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES",
    "CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT",
    "CausalHistorySemanticProducer",
    "EXPERIMENTAL_SOMA_ITERATION_PROFILES",
    "ExperimentalSomaRollingRetargeter",
    "IncrementalCaptureResampler",
    "PinnedSomaRollingRetargeter",
    "RollingSemanticProducer",
    "all_released_profile_latency_proofs",
    "causal_history_stream_contract",
    "causal_history_reference_terms",
    "compare_rolling_to_batch",
    "complete_causal_history_encoder_packet",
    "complete_profile_bound_encoder_packet",
    "measured_future_latency_proof",
    "soma_rolling_retarget_contract",
    "validate_causal_history_packet",
]
