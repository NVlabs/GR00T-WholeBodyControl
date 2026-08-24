"""Recollect and exact-verify the prior rank256 exact-impulse survival direction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_disturbance_survival_score as engine,
    train_g1_true23_sonic_rank256_exact_impulse_survival_score as exact,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_exact_survival_direction_recollection_v1.json"
)
CONTRACT_SHA256 = "e1aecda562838a53e591b8bd09887180c1ed38d76d90811d3860d238244c58a6"
RESULT_FILENAME = "exact_survival_direction_recollection_result.json"
DIRECTION_FILENAME = "exact_survival_direction_recollected.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("direction recollection contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parent = body.get("parent_result", {})
    parent_path = Path(parent.get("path", "")).resolve(strict=True)
    failed = body.get("failed_recollection", {})
    failed_path = Path(failed.get("path", "")).resolve(strict=True)
    if (
        body.get("kind") != "g1_true23_sonic_rank256_exact_survival_direction_recollection_contract_v1"
        or parent_path.is_symlink()
        or sha256_file(parent_path) != parent.get("sha256")
        or failed_path.is_symlink()
        or sha256_file(failed_path) != failed.get("sha256")
        or failed.get("exact_reproduction") is not False
        or failed.get("direction_artifact") is not None
        or body.get("collection", {}).get("all_envs_receive_exact_vector") is not True
        or body.get("boundaries", {}).get("evaluation_runs") != 0
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("direction recollection contract semantic/input mismatch")
    return body


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    contract = _load_contract(root)
    if os.path.lexists(requested_run_dir):
        raise FileExistsError("direction recollection run directory exists")
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

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
        audit_task_space_ppo_env_cfg,
        make_task_space_ppo_env_cfg,
    )

    with exact._scope(root):  # noqa: SLF001
        audit = engine.preflight(root)
        if audit.get("ready") is not True:
            raise RuntimeError("direction recollection engine preflight failed")
        # Match original engine lifecycle exactly: preflight before seeding.
        configure_torch_backends(allow_tf32=False, deterministic=True)
        torch.cuda.set_device(0)
        random.seed(fs.FIXED_SEED)
        np.random.seed(fs.FIXED_SEED % (2**32))
        torch.manual_seed(fs.FIXED_SEED)
        torch.cuda.manual_seed_all(fs.FIXED_SEED)
        baseline, overlay = engine._rank256_state(root, audit["contract"])  # noqa: SLF001
        base_contract = task_space.load_task_space_ppo_contract(root)
        topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
            strict=True
        )
        motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
        run_dir = full_support._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
        full_support._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=engine.NUM_ENVS)
        cfg.seed = fs.FIXED_SEED
        task_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=engine.NUM_ENVS)
        env = ManagerBasedRlEnv(cfg=cfg, device=engine.parent.DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            actor = engine._actor(baseline, topology)  # noqa: SLF001
            vectors = engine._impulse_vectors(  # noqa: SLF001
                engine.NUM_ENVS,
                torch.device(engine.parent.DEVICE),
            )
            engine._materials_unchanged(root, audit, "before_collection")  # noqa: SLF001
            with engine._collection_scope():  # noqa: SLF001
                direction, evidence = engine.COLLECT_AND_SCORE(
                    actor=actor,
                    wrapped=engine._ImpulseProxy(wrapped, vectors),  # noqa: SLF001
                )
        finally:
            env.close()
        scaled = engine._scaled_direction(direction)  # noqa: SLF001
        evidence["scaled_direction_state_sha256"] = fs._state_sha256(scaled)  # noqa: SLF001
        evidence["scaled_direction_l2_target"] = engine.DIRECTION_SCALE * float(evidence["target_direction_l2"])
        expected = contract["expected_evidence"]
        reproduction = {
            name: evidence.get(name) == expected[name]
            for name in (
                "gradient_state_sha256",
                "direction_state_sha256",
                "scaled_direction_state_sha256",
            )
        }
        reproduction["scaled_direction_l2"] = (
            abs(float(evidence["scaled_direction_l2_target"]) - float(expected["scaled_direction_l2"])) <= 1.0e-12
        )
        exact_match = all(reproduction.values())
        artifact = None
        if exact_match:
            path = run_dir / "checkpoints" / DIRECTION_FILENAME
            engine._write_checkpoint(  # noqa: SLF001
                path,
                {
                    "schema_version": 1,
                    "kind": "g1_true23_sonic_rank256_exact_survival_direction_recollection_v1",
                    "contract_sha256": CONTRACT_SHA256,
                    "direction_state_dict": scaled,
                    "direction_state_sha256": evidence["scaled_direction_state_sha256"],
                    "direction_l2": evidence["scaled_direction_l2_target"],
                    "optimizer_steps": 0,
                    "critic_updates": 0,
                    "hardware_authorized": False,
                },
            )
            artifact = {"path": str(path), "sha256": sha256_file(path)}
        report = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_exact_survival_direction_recollection_result_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_overlay": overlay,
            "task_audit": task_audit,
            "prime": prime,
            "gradient_evidence": evidence,
            "expected_evidence": expected,
            "reproduction_by_field": reproduction,
            "exact_reproduction": exact_match,
            "direction_artifact": artifact,
            "training_transitions": engine.TOTAL_TRANSITIONS,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        full_support._write_json_exclusive(run_dir / RESULT_FILENAME, report)  # noqa: SLF001
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.repository_root, args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report["exact_reproduction"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
