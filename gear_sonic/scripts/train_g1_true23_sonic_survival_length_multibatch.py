"""Average four late-region survival-score batches, then run a bounded fine line search."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.scripts import (
    train_g1_true23_sonic_survival_length_score as single,
    train_g1_true23_sonic_task_space_ppo_full_support as parent,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    audit_task_space_ppo_env_cfg,
    make_task_space_ppo_env_cfg,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import (
    STATE_PARAMETER_NAMES,
    TARGET_DIRECTION_L2,
    ZERO_COUNTS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_survival_length_multibatch_v1.json"
)
CONTRACT_SHA256 = "43d9d79b98aecacded2f24d903522267867e2e62597eed44c040ae7c5b6391f9"
BATCH_COUNT = 4
ACTION_SEEDS = (20260805, 20260806, 20260807, 20260808)
SCALES = (-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
TOTAL_TRANSITIONS = BATCH_COUNT * single.TOTAL_TRANSITIONS
RESULT_FILENAME = "survival_length_multibatch_result.json"
FAILURE_FILENAME = "survival_length_multibatch_failure.json"
DIRECTION_FILENAME = "survival_length_multibatch_direction.pt"
CANDIDATE_FILENAME = "sonic_survival_length_multibatch_candidate.pt"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("survival-length multibatch contract mismatch")
    contract = fs._strict_json(path, "survival-length multibatch contract")
    collection = contract.get("collection", {})
    gradient = contract.get("gradient", {})
    gate = contract.get("evaluation_gate", {})
    if (
        contract.get("kind") != "g1_true23_sonic_survival_length_multibatch_contract_v1"
        or contract.get("seed") != fs.FIXED_SEED
        or collection.get("batch_count") != BATCH_COUNT
        or collection.get("num_envs_per_batch") != single.NUM_ENVS
        or collection.get("steps_per_batch") != single.COLLECTION_STEPS
        or collection.get("total_transitions") != TOTAL_TRANSITIONS
        or collection.get("action_seeds") != list(ACTION_SEEDS)
        or gradient.get("parameters") != list(fs.TRAINABLE_ACTOR_PARAMETERS)
        or gradient.get("direction_l2_target") != TARGET_DIRECTION_L2
        or gradient.get("screen_scales") != list(SCALES)
        or gate.get("baseline_policy_sha256") != single.SOURCE_POLICY_SHA256
        or gate.get("required_zero_counts") != list(ZERO_COUNTS)
    ):
        raise ValueError("survival-length multibatch contract semantic mismatch")
    return contract


def _sources(root: Path) -> dict[str, str]:
    relatives = (
        CONTRACT_RELATIVE_PATH,
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_length_multibatch.py"),
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_length_score.py"),
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_score_line_search.py"),
    )
    return {relative.as_posix(): sha256_file((root / relative).resolve(strict=True)) for relative in relatives}


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        base = single.preflight(root)
        if base.get("ready") is not True:
            raise RuntimeError("survival-length multibatch parent preflight not ready")
        material = {
            "parent_material_manifest_sha256": base["material_manifest_sha256"],
            "sources": _sources(root),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_multibatch_preflight_v1",
            "ready": True,
            "contract": contract,
            "parent_preflight": base,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha256(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_multibatch_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _materials_unchanged(root: Path, audit: Mapping[str, Any], phase: str) -> None:
    current = preflight(root)
    if current.get("ready") is not True or current.get("material_manifest_sha256") != audit.get(
        "material_manifest_sha256"
    ):
        raise RuntimeError(f"survival-length multibatch materials changed at {phase}")


def aggregate_directions(
    directions: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if len(directions) != BATCH_COUNT:
        raise ValueError("survival-length multibatch direction count mismatch")
    averaged: dict[str, torch.Tensor] = {}
    for name in STATE_PARAMETER_NAMES:
        tensors = [direction[name].detach().cpu().float().contiguous() for direction in directions]
        if any(tensor.shape != tensors[0].shape or not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("survival-length multibatch direction tensor mismatch")
        averaged[name] = torch.stack(tensors).mean(dim=0).contiguous()
    raw_l2 = math.sqrt(sum(float(torch.sum(value.double().square()).item()) for value in averaged.values()))
    if not math.isfinite(raw_l2) or raw_l2 < TARGET_DIRECTION_L2 * 0.05:
        raise RuntimeError("survival-length multibatch directions cancel")
    factor = TARGET_DIRECTION_L2 / raw_l2
    normalized = {name: torch.mul(value, factor).float().contiguous() for name, value in averaged.items()}
    normalized_l2 = math.sqrt(
        sum(float(torch.sum(value.double().square()).item()) for value in normalized.values())
    )
    pairwise_cosines: list[list[float]] = []
    for left in directions:
        row: list[float] = []
        left_flat = torch.cat([left[name].double().flatten() for name in STATE_PARAMETER_NAMES])
        for right in directions:
            right_flat = torch.cat([right[name].double().flatten() for name in STATE_PARAMETER_NAMES])
            row.append(
                float(
                    torch.dot(left_flat, right_flat)
                    / (torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat))
                )
            )
        pairwise_cosines.append(row)
    return normalized, {
        "aggregation": "equal_mean_of_per_batch_l2_normalized_negative_gradients_then_l2_renormalize",
        "batch_count": BATCH_COUNT,
        "raw_mean_direction_l2": raw_l2,
        "normalization_factor": factor,
        "normalized_direction_l2": normalized_l2,
        "direction_state_sha256": fs._state_sha256(normalized),
        "pairwise_direction_cosines": pairwise_cosines,
    }


def construct_states(
    baseline: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor]
) -> dict[float, dict[str, torch.Tensor]]:
    states: dict[float, dict[str, torch.Tensor]] = {}
    for scale in SCALES:
        state = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
        for name in STATE_PARAMETER_NAMES:
            state[name] = torch.add(state[name], direction[name], alpha=scale).contiguous()
        states[scale] = state
    return states


def _clean(record: Mapping[str, Any]) -> bool:
    return (
        all(record.get(name) == 0 for name in ZERO_COUNTS)
        and not isinstance(record.get("episode_return"), bool)
        and isinstance(record.get("episode_return"), (int, float))
        and math.isfinite(float(record["episode_return"]))
        and isinstance(record.get("completed_transitions"), int)
        and record.get("terminal_q9") == 8 + int(record["completed_transitions"])
    )


def assess_screen(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [record.get("scale") for record in records] != list(SCALES):
        raise ValueError("survival-length multibatch evaluation order mismatch")
    baseline = records[SCALES.index(0.0)]
    baseline_ok = (
        _clean(baseline)
        and baseline.get("policy_state_sha256") == single.SOURCE_POLICY_SHA256
        and 355 <= int(baseline["completed_transitions"]) <= 360
        and baseline.get("termination_names") == ["anchor_pos"]
    )
    allowed = {"ee_body_pos", "anchor_pos", "anchor_ori", "time_out"}
    eligible = (
        [
            record
            for record in records
            if record.get("scale") != 0.0
            and _clean(record)
            and record.get("termination_names") in ([name] for name in allowed)
            and int(record["completed_transitions"]) >= int(baseline["completed_transitions"]) + 10
        ]
        if baseline_ok
        else []
    )
    selected = max(
        eligible,
        key=lambda record: (
            int(record["completed_transitions"]),
            float(record["episode_return"]),
            -abs(float(record["scale"])),
        ),
        default=None,
    )
    return {
        "baseline_passed": baseline_ok,
        "baseline_completed_transitions": baseline.get("completed_transitions"),
        "baseline_terminal_q9": baseline.get("terminal_q9"),
        "screen_candidate_found": selected is not None,
        "selected_scale": selected.get("scale") if selected is not None else None,
        "selected_policy_state_sha256": selected.get("policy_state_sha256") if selected is not None else None,
        "selected_completed_transitions": selected.get("completed_transitions") if selected is not None else None,
        "selected_terminal_q9": selected.get("terminal_q9") if selected is not None else None,
    }


def assess_confirmation(baseline: Mapping[str, Any], candidate: Mapping[str, Any], selected_sha256: str) -> bool:
    return (
        _clean(baseline)
        and baseline.get("policy_state_sha256") == single.SOURCE_POLICY_SHA256
        and 355 <= int(baseline["completed_transitions"]) <= 360
        and baseline.get("termination_names") == ["anchor_pos"]
        and _clean(candidate)
        and candidate.get("policy_state_sha256") == selected_sha256
        and candidate.get("termination_names") in (["ee_body_pos"], ["anchor_pos"], ["anchor_ori"], ["time_out"])
        and int(candidate["completed_transitions"]) >= int(baseline["completed_transitions"]) + 8
    )


def _create_output_subdirectories(run_dir: Path) -> None:
    for name in ("batches", "gradient", "evaluations", "checkpoints"):
        (run_dir / name).mkdir(mode=0o755, exist_ok=True)


def run(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("survival-length multibatch preflight failed")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    base_contract = fs.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    baseline = single._load_source_policy()
    run_dir = parent._create_run_dir_exclusive(requested_run_dir)
    _create_output_subdirectories(run_dir)
    parent._write_json_exclusive(run_dir / "preflight.json", audit)
    parent._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])
    directions: list[dict[str, torch.Tensor]] = []
    batch_evidence: list[dict[str, Any]] = []
    task_audit: dict[str, Any] | None = None
    primes: list[dict[str, Any]] = []
    _materials_unchanged(root, audit, "before_collection")
    for batch_index, action_seed in enumerate(ACTION_SEEDS):
        random.seed(fs.FIXED_SEED)
        np.random.seed(fs.FIXED_SEED % (2**32))
        torch.manual_seed(fs.FIXED_SEED)
        torch.cuda.manual_seed_all(fs.FIXED_SEED)
        actor = single._actor(baseline, topology)
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=single.NUM_ENVS)
        cfg.seed = fs.FIXED_SEED
        current_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=single.NUM_ENVS)
        task_audit = current_audit if task_audit is None else task_audit
        env = ManagerBasedRlEnv(cfg=cfg, device=single.DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            primes.append(prime)
            random.seed(action_seed)
            np.random.seed(action_seed % (2**32))
            torch.manual_seed(action_seed)
            torch.cuda.manual_seed_all(action_seed)
            direction, evidence = single.collect_and_score(actor=actor, wrapped=wrapped)
        finally:
            env.close()
        evidence = {**evidence, "batch_index": batch_index, "action_seed": action_seed}
        directions.append(direction)
        batch_evidence.append(evidence)
        parent._write_json_exclusive(run_dir / "batches" / f"batch_{batch_index}.json", evidence)
        del actor
        torch.cuda.empty_cache()
    _materials_unchanged(root, audit, "after_collection")
    direction, aggregation = aggregate_directions(directions)
    single._write_torch(
        run_dir / "gradient" / DIRECTION_FILENAME,
        {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_length_multibatch_direction_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_checkpoint_sha256": single.SOURCE_CHECKPOINT_SHA256,
            "direction_state_dict": direction,
            "direction_state_sha256": aggregation["direction_state_sha256"],
            "optimizer_steps": 0,
            "critic_updates": 0,
        },
    )
    states = construct_states(baseline, direction)
    records: list[dict[str, Any]] = []
    for index, scale in enumerate(SCALES):
        record = single._evaluate_state(
            state=states[scale],
            scale=scale,
            topology=topology,
            motion=motion,
            material_manifest_sha256=audit["material_manifest_sha256"],
        )
        record["contract_sha256"] = CONTRACT_SHA256
        parent._write_json_exclusive(run_dir / "evaluations" / f"screen_{index}.json", record)
        records.append(record)
    screen = assess_screen(records)
    confirmations: dict[str, Any] | None = None
    candidate: dict[str, str] | None = None
    if screen["screen_candidate_found"] is True:
        selected_scale = float(screen["selected_scale"])
        baseline_confirmation = single._evaluate_state(
            state=states[0.0],
            scale=0.0,
            topology=topology,
            motion=motion,
            material_manifest_sha256=audit["material_manifest_sha256"],
        )
        candidate_confirmation = single._evaluate_state(
            state=states[selected_scale],
            scale=selected_scale,
            topology=topology,
            motion=motion,
            material_manifest_sha256=audit["material_manifest_sha256"],
        )
        confirmed = assess_confirmation(
            baseline_confirmation, candidate_confirmation, str(screen["selected_policy_state_sha256"])
        )
        confirmations = {
            "baseline": baseline_confirmation,
            "candidate": candidate_confirmation,
            "confirmed": confirmed,
        }
        parent._write_json_exclusive(run_dir / "evaluations" / "confirmation.json", confirmations)
        if confirmed:
            checkpoint = {
                "schema_version": 1,
                "kind": "g1_true23_sonic_survival_length_multibatch_candidate_v1",
                "contract_sha256": CONTRACT_SHA256,
                "source_checkpoint_sha256": single.SOURCE_CHECKPOINT_SHA256,
                "policy_state_dict": states[selected_scale],
                "policy_state_sha256": screen["selected_policy_state_sha256"],
                "selected_scale": selected_scale,
                "aggregation": aggregation,
                "training_transitions": TOTAL_TRANSITIONS,
                "gradient_computations": BATCH_COUNT,
                "optimizer_steps": 0,
                "critic_updates": 0,
                "support_qualified": False,
                "promotion_eligible": False,
                "deployment_ready": False,
                "hardware_authorized": False,
            }
            path = run_dir / "checkpoints" / CANDIDATE_FILENAME
            single._write_torch(path, checkpoint)
            candidate = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_survival_length_multibatch_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "task_audit": task_audit,
        "primes": primes,
        "batch_evidence": batch_evidence,
        "aggregation": aggregation,
        "evaluations": records,
        "screen_assessment": screen,
        "confirmations": confirmations,
        "candidate": candidate,
        "training_transitions": TOTAL_TRANSITIONS,
        "gradient_computations": BATCH_COUNT,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _materials_unchanged(root, audit, "before_result")
    parent._write_json_exclusive(run_dir / RESULT_FILENAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    train = sub.add_parser("train")
    train.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    train.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run(args.repository_root, args.run_dir)
    except Exception as error:
        run_dir = args.run_dir.expanduser().resolve(strict=False)
        if run_dir.is_dir() and not run_dir.is_symlink():
            path = run_dir / FAILURE_FILENAME
            if not os.path.lexists(path):
                parent._write_json_exclusive(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "g1_true23_sonic_survival_length_multibatch_failure_v1",
                        "contract_sha256": CONTRACT_SHA256,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "candidate": None,
                        "support_qualified": False,
                        "promotion_eligible": False,
                        "deployment_ready": False,
                        "hardware_authorized": False,
                    },
                )
        print(f"survival-length multibatch failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
