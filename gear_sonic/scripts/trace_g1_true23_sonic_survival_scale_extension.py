"""Extend the admitted survival-score direction with seven deterministic scales."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as parent
from gear_sonic.scripts.train_g1_true23_sonic_survival_score_line_search import _evaluate_state
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import STATE_PARAMETER_NAMES, ZERO_COUNTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_sonic_survival_scale_extension_v1.json")
CONTRACT_SHA256 = "2c8c091aa6463b83bdb3dbcffb5c25c0cdc21340e0553ddeb416c90019972b43"
BASELINE_PATH = Path(
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_v1_seed20260805/checkpoints/sonic_task_space_full_support_model_0.pt"
)
BASELINE_SHA256 = "83682858ed907f7353668bf9c35088b2619824ce6491cd6542871202457cc344"
SURVIVAL_RESULT_PATH = Path(
    "/root/g1_true23_runs/sonic_survival_score_line_search_v1_seed20260805/survival_score_result.json"
)
SURVIVAL_RESULT_SHA256 = "22874a8a3271b778c3d2d0f36d8813835c0c1b9b58e20165c016b3a70496aed7"
SURVIVAL_CHECKPOINT_PATH = Path(
    "/root/g1_true23_runs/sonic_survival_score_line_search_v1_seed20260805/checkpoints/sonic_survival_score_candidate.pt"
)
SURVIVAL_CHECKPOINT_SHA256 = "60dc4affcdb8f5874650eed976b30a1558159f2ceb3c683fece4725eac82f8cf"
BASELINE_POLICY_SHA256 = fs.INITIAL_OVERLAY_POLICY_STATE_SHA256
SCALE1_POLICY_SHA256 = "e846f3d08595dbb383c6a524caa4f551ff26e958792ac48c5b13e1f643f7d1a2"
SCALES = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
RESULT_FILENAME = "survival_scale_extension_result.json"
FAILURE_FILENAME = "survival_scale_extension_failure.json"
CHECKPOINT_FILENAME = "sonic_survival_scale_extension_candidate.pt"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _strict_file(path: Path, expected: str, context: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{context} must be regular file")
    resolved = lexical.resolve(strict=True)
    if sha256_file(resolved) != expected:
        raise ValueError(f"{context} SHA256 mismatch")
    return resolved


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("survival extension contract mismatch")
    contract = fs._strict_json(path, "survival extension contract")
    if (
        contract.get("kind") != "g1_true23_sonic_survival_scale_extension_contract_v1"
        or contract.get("seed") != fs.FIXED_SEED
        or contract.get("construction", {}).get("changed_tensors") != list(STATE_PARAMETER_NAMES)
        or contract.get("construction", {}).get("scales") != list(SCALES)
        or contract.get("evaluation_gate", {}).get("required_zero_counts") != list(ZERO_COUNTS)
    ):
        raise ValueError("survival extension contract semantics mismatch")
    return contract


def _sources(root: Path) -> dict[str, str]:
    relatives = (
        CONTRACT_RELATIVE_PATH,
        Path("gear_sonic/scripts/trace_g1_true23_sonic_survival_scale_extension.py"),
    )
    return {item.as_posix(): sha256_file((root / item).resolve(strict=True)) for item in relatives}


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        parent_audit = parent.preflight(root)
        if parent_audit.get("ready") is not True:
            raise RuntimeError("survival extension parent preflight not ready")
        evidence = {
            "baseline": str(_strict_file(BASELINE_PATH, BASELINE_SHA256, "extension baseline")),
            "survival_result": str(
                _strict_file(SURVIVAL_RESULT_PATH, SURVIVAL_RESULT_SHA256, "extension survival result")
            ),
            "survival_checkpoint": str(
                _strict_file(
                    SURVIVAL_CHECKPOINT_PATH,
                    SURVIVAL_CHECKPOINT_SHA256,
                    "extension survival checkpoint",
                )
            ),
        }
        material = {
            "parent_material_manifest_sha256": parent_audit["material_manifest"]["material_manifest_sha256"],
            "evidence": evidence,
            "sources": _sources(root),
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_scale_extension_preflight_v1",
            "ready": True,
            "contract": contract,
            "contract_sha256": CONTRACT_SHA256,
            "parent_preflight": parent_audit,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha256(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_scale_extension_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
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
        raise RuntimeError(f"survival extension materials changed at {phase}")


def construct_states() -> tuple[dict[float, dict[str, torch.Tensor]], dict[str, Any]]:
    baseline_checkpoint = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
    survival_checkpoint = torch.load(SURVIVAL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    baseline = baseline_checkpoint["policy_state_dict"]
    scale1 = survival_checkpoint["policy_state_dict"]
    if (
        inspect_true23_policy_state(baseline_checkpoint, reference_profile=fs.REFERENCE_PROFILE)
        != BASELINE_POLICY_SHA256
        or inspect_true23_policy_state({"policy_state_dict": scale1}, reference_profile=fs.REFERENCE_PROFILE)
        != SCALE1_POLICY_SHA256
        or set(baseline) != set(scale1)
    ):
        raise ValueError("survival extension endpoint policy identity mismatch")
    changed = [name for name in baseline if not torch.equal(baseline[name], scale1[name])]
    if set(changed) != set(STATE_PARAMETER_NAMES) or len(changed) != len(STATE_PARAMETER_NAMES):
        raise ValueError("survival extension changed tensor set mismatch")
    direction = {name: (scale1[name] - baseline[name]).contiguous() for name in STATE_PARAMETER_NAMES}
    states: dict[float, dict[str, torch.Tensor]] = {}
    hashes: dict[str, str] = {}
    for scale in SCALES:
        state = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
        for name in STATE_PARAMETER_NAMES:
            state[name] = torch.add(state[name], direction[name], alpha=scale).contiguous()
        states[scale] = state
        hashes[str(scale)] = inspect_true23_policy_state(
            {"policy_state_dict": state}, reference_profile=fs.REFERENCE_PROFILE
        )
    direction_l2 = torch.linalg.vector_norm(
        torch.cat([direction[name].reshape(-1).double() for name in STATE_PARAMETER_NAMES])
    ).item()
    return states, {
        "baseline_policy_state_sha256": BASELINE_POLICY_SHA256,
        "scale1_policy_state_sha256": SCALE1_POLICY_SHA256,
        "direction_state_sha256": fs._state_sha256(direction),
        "direction_l2": direction_l2,
        "policy_state_sha256_by_scale": hashes,
    }


def assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [record.get("scale") for record in records] != list(SCALES):
        raise ValueError("survival extension evaluation order mismatch")
    eligible = [
        record
        for record in records
        if record.get("completed_transitions", -1) >= 182
        and record.get("terminal_q9", -1) >= 190
        and record.get("termination_names") in (["ee_body_pos"], ["anchor_pos"], ["anchor_ori"], ["time_out"])
        and all(record.get(name) == 0 for name in ZERO_COUNTS)
        and not isinstance(record.get("episode_return"), bool)
        and isinstance(record.get("episode_return"), (int, float))
        and math.isfinite(float(record["episode_return"]))
    ]
    selected = max(
        eligible,
        key=lambda record: (
            int(record["completed_transitions"]),
            float(record["episode_return"]),
            -float(record["scale"]),
        ),
        default=None,
    )
    return {
        "candidate_selected": selected is not None,
        "selected_scale": selected.get("scale") if selected is not None else None,
        "selected_policy_state_sha256": selected.get("policy_state_sha256") if selected is not None else None,
        "selected_completed_transitions": selected.get("completed_transitions") if selected is not None else None,
        "selected_terminal_q9": selected.get("terminal_q9") if selected is not None else None,
        "selected_termination_names": selected.get("termination_names") if selected is not None else None,
        "diagnostic_candidate_only": True,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"survival extension checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(value), temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(repository_root: Path, requested_run_dir: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("survival extension preflight failed")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    base = fs.load_task_space_ppo_contract(root)
    topology = (root / base["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(strict=True)
    motion = (root / base["environment"]["motion_relative_path"]).resolve(strict=True)
    run_dir = parent._create_run_dir_exclusive(requested_run_dir)
    parent._write_json_exclusive(run_dir / "preflight.json", audit)
    parent._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])
    states, construction = construct_states()
    parent._write_json_exclusive(run_dir / "construction.json", construction)
    records: list[dict[str, Any]] = []
    for index, scale in enumerate(SCALES):
        _materials_unchanged(root, audit, f"before_evaluation_{index}")
        record = _evaluate_state(
            state=states[scale],
            scale=scale,
            topology=topology,
            motion=motion,
            material_manifest_sha256=audit["material_manifest_sha256"],
        )
        record["contract_sha256"] = CONTRACT_SHA256
        parent._write_json_exclusive(run_dir / "evaluations" / f"evaluation_{index}.json", record)
        records.append(record)
    assessment = assess(records)
    candidate: dict[str, str] | None = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        checkpoint = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_survival_scale_extension_candidate_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_survival_result_sha256": SURVIVAL_RESULT_SHA256,
            "policy_state_dict": states[scale],
            "policy_state_sha256": assessment["selected_policy_state_sha256"],
            "selected_scale": scale,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        path = run_dir / "checkpoints" / CHECKPOINT_FILENAME
        _write_torch(path, checkpoint)
        candidate = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_survival_scale_extension_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "construction": construction,
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "training_transitions": 0,
        "gradient_computations": 0,
        "optimizer_steps": 0,
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
    trace = sub.add_parser("trace")
    trace.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    trace.add_argument("--run-dir", type=Path, required=True)
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
                        "kind": "g1_true23_sonic_survival_scale_extension_failure_v1",
                        "contract_sha256": CONTRACT_SHA256,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "candidate": None,
                        "support_qualified": False,
                        "promotion_eligible": False,
                        "deployment_ready": False,
                        "hardware_authorized": False,
                    },
                )
        print(f"survival extension failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
