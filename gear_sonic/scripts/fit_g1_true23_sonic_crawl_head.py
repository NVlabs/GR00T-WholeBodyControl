"""Fit a crawl-specific native true23 SONIC final affine from 23D rollout data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanSonicPolicy,
    CleanTrue23MujocoController,
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import _reference_policy_frame
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history

RIDGE_GRID = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
HEAD_ALPHA_GRID = (1.0, 0.9, 0.75, 0.5, 0.25)
RAW_ACTION_BOUND = 10.0
EMBEDDED_RAW_CLIP = 9.8


def _hidden_session(decoder_path: Path) -> ort.InferenceSession:
    model = onnx.load(str(decoder_path), load_external_data=True)
    model.graph.output.append(helper.make_tensor_value_info("/Mul_7_output_0", TensorProto.FLOAT, [1, 512]))
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _dataset(
    *,
    root: Path,
    motion_path: Path,
    rollout_path: Path,
    encoder_path: Path,
    decoder_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(rollout_path, allow_pickle=False) as archive:
        pre_qpos = np.asarray(archive["pre_qpos"], dtype=np.float64)
        pre_qvel = np.asarray(archive["pre_qvel"], dtype=np.float64)
        labels = np.asarray(archive["applied_raw_native23"], dtype=np.float32)
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    if (
        frame_count < 12
        or pre_qpos.shape != (frame_count, 30)
        or pre_qvel.shape != (frame_count, 29)
        or labels.shape != (frame_count, 23)
    ):
        raise ValueError("true23 adapter training arrays shape mismatch")
    policy = CleanSonicPolicy(encoder_path, decoder_path)
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
        frames = [*frames[1:], controller._policy_frame()]
        q9 = frame_index - 1
        packet = motion_reference_terms(motion, q9)
        buffered_pelvis = (
            np.asarray(motion["body_quat_w"][9, 0], dtype=np.float64)
            if frame_index == 10
            else pre_qpos[frame_index - 1, 3:7]
        )
        encoder267 = encoder267_from_reference(packet, buffered_pelvis)
        token64 = np.asarray(
            policy.encoder.run(None, {policy.encoder_input: encoder267.reshape(1, 267)})[0],
            dtype=np.float32,
        ).reshape(64)
        decoder_rows.append(np.concatenate((token64, term_major_history(frames))).astype(np.float32))
        output_labels.append(labels[frame_index])
        previous_safe, _ = safe_target_transform_numpy(labels[frame_index])
    decoder994 = np.asarray(decoder_rows, dtype=np.float32)
    label_array = np.asarray(output_labels, dtype=np.float32)
    hidden_session = _hidden_session(decoder_path)
    input_name = hidden_session.get_inputs()[0].name
    hidden = np.concatenate(
        [np.asarray(hidden_session.run(["/Mul_7_output_0"], {input_name: row[None]})[0]) for row in decoder994],
        axis=0,
    ).astype(np.float32)
    return decoder994, hidden, label_array


def _fit_delta(
    hidden: np.ndarray,
    residual: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((hidden.astype(np.float64), np.ones(len(hidden), dtype=np.float64)))
    gram = design.T @ design / len(design)
    rhs = design.T @ residual.astype(np.float64) / len(design)
    coefficients = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    return coefficients[:-1].T, coefficients[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-decoder", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--head-alpha", type=float)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--base-decoder", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    output_decoder = args.output_decoder if args.output_decoder.is_absolute() else root / args.output_decoder
    output_manifest = args.output_manifest if args.output_manifest.is_absolute() else root / args.output_manifest
    if os.path.lexists(output_decoder) or os.path.lexists(output_manifest):
        raise FileExistsError("crawl head output exists")
    encoder_path = root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
    base_decoder = (
        root / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx"
        if args.base_decoder is None
        else (args.base_decoder if args.base_decoder.is_absolute() else root / args.base_decoder)
    )
    motion_path = (
        root / "artifacts/g1_true23/sonic_library_true23_hand_original_speed_v1/hand_crawling.true23.npz"
        if args.motion is None
        else (args.motion if args.motion.is_absolute() else root / args.motion)
    )
    rollout_path = (
        root
        / (
            "artifacts/g1_true23/sonic_library_true23_released_adapter_v9_crawl_dataset/"
            "hand_crawling.true23.physical.npz"
        )
        if args.rollout is None
        else (args.rollout if args.rollout.is_absolute() else root / args.rollout)
    )
    decoder994, hidden, labels = _dataset(
        root=root,
        motion_path=motion_path,
        rollout_path=rollout_path,
        encoder_path=encoder_path,
        decoder_path=base_decoder,
    )
    base_session = ort.InferenceSession(str(base_decoder), providers=["CPUExecutionProvider"])
    base_input = base_session.get_inputs()[0].name
    base_output = np.concatenate(
        [np.asarray(base_session.run(None, {base_input: row[None]})[0]) for row in decoder994],
        axis=0,
    ).astype(np.float32)
    split = len(hidden) - 100
    candidates = []
    for ridge in RIDGE_GRID:
        delta_w, delta_b = _fit_delta(hidden[:split], labels[:split] - base_output[:split], ridge)
        prediction = base_output[split:] + hidden[split:] @ delta_w.T + delta_b
        rmse = float(np.sqrt(np.mean(np.square(prediction - labels[split:]))))
        candidates.append((rmse, ridge))
    _, selected_ridge = min(candidates)
    delta_w, delta_b = _fit_delta(hidden, labels - base_output, selected_ridge)
    alpha_candidates = []
    for alpha in HEAD_ALPHA_GRID:
        alpha_prediction = np.clip(
            base_output + alpha * (hidden @ delta_w.T + delta_b),
            -EMBEDDED_RAW_CLIP,
            EMBEDDED_RAW_CLIP,
        )
        alpha_candidates.append(
            {
                "alpha": alpha,
                "rmse": float(np.sqrt(np.mean(np.square(alpha_prediction - labels)))),
                "raw_abs_max": float(np.max(np.abs(alpha_prediction))),
            }
        )
    bounded_candidates = [item for item in alpha_candidates if item["raw_abs_max"] < RAW_ACTION_BOUND]
    if not bounded_candidates:
        raise RuntimeError("no crawl head alpha satisfies strict native raw-action bound")
    selected_alpha = float(min(bounded_candidates, key=lambda item: item["rmse"])["alpha"])
    if args.head_alpha is not None:
        if not 0.0 <= args.head_alpha <= 1.0:
            raise ValueError("head alpha must be within [0, 1]")
        selected_alpha = float(args.head_alpha)
    delta_w *= selected_alpha
    delta_b *= selected_alpha
    model = onnx.load(str(base_decoder), load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    base_w = numpy_helper.to_array(initializers["layers.8.weight"]).astype(np.float32)
    base_b = numpy_helper.to_array(initializers["layers.8.bias"]).astype(np.float32)
    new_w = (base_w.astype(np.float64) + delta_w).astype(np.float32)
    new_b = (base_b.astype(np.float64) + delta_b).astype(np.float32)
    initializers["layers.8.weight"].CopyFrom(numpy_helper.from_array(new_w, "layers.8.weight"))
    initializers["layers.8.bias"].CopyFrom(numpy_helper.from_array(new_b, "layers.8.bias"))
    if model.graph.node[-1].output != ["action"] or model.graph.node[-1].op_type != "Gemm":
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
    output_decoder.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_decoder))
    onnx.checker.check_model(str(output_decoder), full_check=True)
    candidate_session = ort.InferenceSession(str(output_decoder), providers=["CPUExecutionProvider"])
    candidate_input = candidate_session.get_inputs()[0].name
    prediction = np.concatenate(
        [np.asarray(candidate_session.run(None, {candidate_input: row[None]})[0]) for row in decoder994],
        axis=0,
    )
    report = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_crawl_native_bounded_last_affine_ridge_v2",
        "base_decoder_sha256": sha256_file(base_decoder),
        "candidate_decoder_sha256": sha256_file(output_decoder),
        "encoder_sha256": sha256_file(encoder_path),
        "motion_sha256": sha256_file(motion_path),
        "rollout_sha256": sha256_file(rollout_path),
        "row_count": len(hidden),
        "selected_ridge": selected_ridge,
        "selected_head_alpha": selected_alpha,
        "validation_grid": [{"ridge": ridge, "rmse": rmse} for rmse, ridge in candidates],
        "head_alpha_grid": alpha_candidates,
        "base_rmse": float(np.sqrt(np.mean(np.square(base_output - labels)))),
        "candidate_rmse": float(np.sqrt(np.mean(np.square(prediction - labels)))),
        "candidate_max_abs_error": float(np.max(np.abs(prediction - labels))),
        "candidate_raw_abs_max": float(np.max(np.abs(prediction))),
        "candidate_raw_bound": RAW_ACTION_BOUND,
        "candidate_raw_bound_gate_passed": bool(np.max(np.abs(prediction)) < RAW_ACTION_BOUND),
        "embedded_raw_clip_abs": EMBEDDED_RAW_CLIP,
        "delta_weight_frobenius": float(np.linalg.norm(delta_w)),
        "delta_bias_l2": float(np.linalg.norm(delta_b)),
        "changed_components": ["final_affine", "bounded_raw_output_clip"],
        "authorization": {
            "offline_behavior_cloning_only": True,
            "simulator_candidate_only": True,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
