"""Fit one native-23 SONIC head across crawl, elbow crawl, and dance.

Teacher rollouts are offline evidence only.  Every exported candidate keeps the
native true23 encoder/decoder ABI and changes only the final affine decoder
layer.  Output remains simulator-only and cannot authorize robot control.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from gear_sonic.scripts.fit_g1_true23_sonic_crawl_head import (
    EMBEDDED_RAW_CLIP,
    RIDGE_GRID,
    _fit_delta,
    _hidden_session,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanTrue23MujocoController,
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import (
    ExactHashSonicPolicy,
    _reference_policy_frame,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history

ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)
VALIDATION_TAIL_ROWS = 100
RAW_ACTION_BOUND = 10.0
DEFAULT_ENCODER = Path("artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx")
DEFAULT_DATASETS = (
    (
        "hand_crawl",
        Path("artifacts/g1_true23/sonic_library_true23_hand_original_speed_v1/hand_crawling.true23.npz"),
        Path(
            "artifacts/g1_true23/sonic_library_true23_released_adapter_v9_crawl_dataset/"
            "hand_crawling.true23.physical.npz"
        ),
    ),
    (
        "elbow_crawl",
        Path("artifacts/g1_true23/sonic_library_true23_elbow_physical_reference_v1/elbow_crawling.true23.npz"),
        Path(
            "artifacts/g1_true23/sonic_library_true23_released_adapter_v11_elbow_dataset/"
            "elbow_crawling.true23.physical.npz"
        ),
    ),
    (
        "happy_dance",
        Path("artifacts/g1_true23/sonic_library_true23_happy_physical_reference_v1/happy_dance.true23.npz"),
        Path(
            "artifacts/g1_true23/sonic_library_true23_released_adapter_v10_happy_dataset/"
            "happy_dance.true23.physical.npz"
        ),
    ),
)


def _dataset(
    *,
    root: Path,
    motion_path: Path,
    rollout_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    decoder_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(rollout_path, allow_pickle=False) as archive:
        pre_qpos = np.asarray(archive["pre_qpos"], dtype=np.float64)
        pre_qvel = np.asarray(archive["pre_qvel"], dtype=np.float64)
        labels = np.asarray(archive["applied_raw_native23"], dtype=np.float32)
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    if (
        frame_count < VALIDATION_TAIL_ROWS + 12
        or pre_qpos.shape != (frame_count, 30)
        or pre_qvel.shape != (frame_count, 29)
        or labels.shape != (frame_count, 23)
        or not np.isfinite(pre_qpos).all()
        or not np.isfinite(pre_qvel).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("native23 teacher rollout arrays mismatch")

    policy = ExactHashSonicPolicy(
        encoder_path,
        decoder_path,
        expected_decoder_sha256=decoder_sha256,
    )
    controller = CleanTrue23MujocoController(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=policy,
    )
    frames = [_reference_policy_frame(motion, index) for index in range(10)]
    decoder_rows: list[np.ndarray] = []
    output_labels: list[np.ndarray] = []
    previous_safe = np.zeros(23, dtype=np.float32)
    for frame_index in range(10, frame_count):
        controller.data.qpos[:] = pre_qpos[frame_index]
        controller.data.qvel[:] = pre_qvel[frame_index]
        controller.previous_safe_native = previous_safe.copy()
        controller.module.mj_forward(controller.model, controller.data)
        frames = [*frames[1:], controller._policy_frame()]  # noqa: SLF001
        packet = motion_reference_terms(motion, frame_index - 1)
        buffered_pelvis = (
            np.asarray(motion["body_quat_w"][9, 0], dtype=np.float64)
            if frame_index == 10
            else pre_qpos[frame_index - 1, 3:7]
        )
        encoder267 = encoder267_from_reference(packet, buffered_pelvis)
        token64 = np.asarray(
            policy.encoder.run(
                None,
                {policy.encoder_input: encoder267.reshape(1, 267)},
            )[0],
            dtype=np.float32,
        ).reshape(64)
        decoder_rows.append(np.concatenate((token64, term_major_history(frames))).astype(np.float32))
        output_labels.append(labels[frame_index])
        previous_safe, _ = safe_target_transform_numpy(labels[frame_index])

    decoder994 = np.asarray(decoder_rows, dtype=np.float32)
    label_array = np.asarray(output_labels, dtype=np.float32)
    hidden_session = _hidden_session(decoder_path)
    hidden_input = hidden_session.get_inputs()[0].name
    hidden = np.concatenate(
        [
            np.asarray(
                hidden_session.run(
                    ["/Mul_7_output_0"],
                    {hidden_input: row[None]},
                )[0]
            )
            for row in decoder994
        ],
        axis=0,
    ).astype(np.float32)
    if hidden.shape != (len(decoder994), 512):
        raise ValueError("native23 decoder hidden shape mismatch")
    return decoder994, hidden, label_array


def select_ridge(
    hidden: np.ndarray,
    residual: np.ndarray,
    group_lengths: Sequence[int],
) -> tuple[float, list[dict[str, float]]]:
    if hidden.ndim != 2 or residual.shape != (len(hidden), 23):
        raise ValueError("multimotion ridge arrays mismatch")
    if sum(group_lengths) != len(hidden) or any(length <= VALIDATION_TAIL_ROWS for length in group_lengths):
        raise ValueError("multimotion ridge group lengths mismatch")
    validation = np.zeros(len(hidden), dtype=bool)
    cursor = 0
    for length in group_lengths:
        validation[cursor + length - VALIDATION_TAIL_ROWS : cursor + length] = True
        cursor += length
    candidates: list[dict[str, float]] = []
    for ridge in RIDGE_GRID:
        delta_w, delta_b = _fit_delta(hidden[~validation], residual[~validation], ridge)
        prediction = hidden[validation] @ delta_w.T + delta_b
        rmse = float(np.sqrt(np.mean(np.square(prediction - residual[validation]))))
        candidates.append({"ridge": float(ridge), "validation_rmse": rmse})
    selected = min(candidates, key=lambda item: item["validation_rmse"])
    return float(selected["ridge"]), candidates


def _alpha_name(alpha: float) -> str:
    if alpha not in ALPHAS:
        raise ValueError("unsupported multimotion head alpha")
    return f"alpha{int(round(alpha * 100)):03d}"


def _export_candidate(
    *,
    base_decoder: Path,
    output_path: Path,
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    alpha: float,
) -> None:
    if os.path.lexists(output_path):
        raise FileExistsError(f"multimotion candidate exists: {output_path}")
    model = onnx.load(str(base_decoder), load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    base_w = numpy_helper.to_array(initializers["layers.8.weight"]).astype(np.float32)
    base_b = numpy_helper.to_array(initializers["layers.8.bias"]).astype(np.float32)
    new_w = (base_w.astype(np.float64) + alpha * delta_w).astype(np.float32)
    new_b = (base_b.astype(np.float64) + alpha * delta_b).astype(np.float32)
    initializers["layers.8.weight"].CopyFrom(numpy_helper.from_array(new_w, "layers.8.weight"))
    initializers["layers.8.bias"].CopyFrom(numpy_helper.from_array(new_b, "layers.8.bias"))
    if model.graph.node[-1].op_type != "Gemm" or model.graph.node[-1].output != ["action"]:
        raise ValueError("base decoder final affine graph changed")
    model.graph.node[-1].output[0] = "action_unbounded"
    model.graph.initializer.extend(
        (
            numpy_helper.from_array(np.asarray(-EMBEDDED_RAW_CLIP, dtype=np.float32), "raw_clip_min"),
            numpy_helper.from_array(np.asarray(EMBEDDED_RAW_CLIP, dtype=np.float32), "raw_clip_max"),
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Clip",
            ("action_unbounded", "raw_clip_min", "raw_clip_max"),
            ("action",),
            name="/bounded_raw_action",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-decoder", type=Path, required=True)
    parser.add_argument("--expected-base-decoder-sha256", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    base_decoder = (args.base_decoder if args.base_decoder.is_absolute() else root / args.base_decoder).resolve(
        strict=True
    )
    encoder_path = (root / DEFAULT_ENCODER).resolve(strict=True)
    output_prefix = args.output_prefix if args.output_prefix.is_absolute() else root / args.output_prefix
    manifest_path = args.output_manifest if args.output_manifest.is_absolute() else root / args.output_manifest
    if len(args.expected_base_decoder_sha256) != 64:
        raise ValueError("expected base decoder SHA256 mismatch")
    if sha256_file(base_decoder) != args.expected_base_decoder_sha256:
        raise ValueError("base decoder SHA256 mismatch")
    if os.path.lexists(manifest_path):
        raise FileExistsError("multimotion manifest exists")

    groups: list[dict[str, Any]] = []
    decoder_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for name, motion_relative, rollout_relative in DEFAULT_DATASETS:
        motion = (root / motion_relative).resolve(strict=True)
        rollout = (root / rollout_relative).resolve(strict=True)
        decoder994, hidden, group_labels = _dataset(
            root=root,
            motion_path=motion,
            rollout_path=rollout,
            encoder_path=encoder_path,
            decoder_path=base_decoder,
            decoder_sha256=args.expected_base_decoder_sha256,
        )
        groups.append(
            {
                "name": name,
                "rows": len(hidden),
                "motion_sha256": sha256_file(motion),
                "rollout_sha256": sha256_file(rollout),
            }
        )
        decoder_rows.append(decoder994)
        hidden_rows.append(hidden)
        labels.append(group_labels)

    decoder994 = np.concatenate(decoder_rows, axis=0)
    hidden = np.concatenate(hidden_rows, axis=0)
    teacher = np.concatenate(labels, axis=0)
    base_session = ort.InferenceSession(str(base_decoder), providers=["CPUExecutionProvider"])
    base_input = base_session.get_inputs()[0].name
    base_output = np.concatenate(
        [np.asarray(base_session.run(None, {base_input: row[None]})[0]) for row in decoder994],
        axis=0,
    ).astype(np.float32)
    residual = teacher - base_output
    group_lengths = [int(group["rows"]) for group in groups]
    selected_ridge, ridge_grid = select_ridge(hidden, residual, group_lengths)
    delta_w, delta_b = _fit_delta(hidden, residual, selected_ridge)

    candidates: list[dict[str, Any]] = []
    cursor_by_group = np.cumsum([0, *group_lengths])
    for alpha in ALPHAS:
        name = _alpha_name(alpha)
        output_path = Path(f"{output_prefix}.{name}.decoder.onnx")
        _export_candidate(
            base_decoder=base_decoder,
            output_path=output_path,
            delta_w=delta_w,
            delta_b=delta_b,
            alpha=alpha,
        )
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        prediction = np.concatenate(
            [np.asarray(session.run(None, {input_name: row[None]})[0]) for row in decoder994],
            axis=0,
        ).astype(np.float32)
        per_group = {}
        for index, group in enumerate(groups):
            start = int(cursor_by_group[index])
            stop = int(cursor_by_group[index + 1])
            per_group[group["name"]] = {
                "rmse": float(np.sqrt(np.mean(np.square(prediction[start:stop] - teacher[start:stop])))),
                "raw_abs_max": float(np.max(np.abs(prediction[start:stop]))),
            }
        candidates.append(
            {
                "alpha": alpha,
                "name": name,
                "decoder_path": str(output_path),
                "decoder_sha256": sha256_file(output_path),
                "rmse": float(np.sqrt(np.mean(np.square(prediction - teacher)))),
                "raw_abs_max": float(np.max(np.abs(prediction))),
                "raw_bound_passed": bool(np.max(np.abs(prediction)) < RAW_ACTION_BOUND),
                "per_group": per_group,
            }
        )

    report = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_native23_multimotion_last_affine_ridge_v1",
        "base_decoder_path": str(base_decoder),
        "base_decoder_sha256": args.expected_base_decoder_sha256,
        "encoder_path": str(encoder_path),
        "encoder_sha256": sha256_file(encoder_path),
        "row_count": len(hidden),
        "groups": groups,
        "selected_ridge": selected_ridge,
        "ridge_grid": ridge_grid,
        "base_rmse": float(np.sqrt(np.mean(np.square(base_output - teacher)))),
        "delta_weight_frobenius": float(np.linalg.norm(delta_w)),
        "delta_bias_l2": float(np.linalg.norm(delta_b)),
        "changed_components": ["decoder_final_affine", "bounded_raw_output_clip"],
        "candidates": candidates,
        "authorization": {
            "offline_teacher_distillation_only": True,
            "runtime_output_dof": 23,
            "source_29dof_physics_used_at_runtime": False,
            "simulator_candidate_only": True,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
