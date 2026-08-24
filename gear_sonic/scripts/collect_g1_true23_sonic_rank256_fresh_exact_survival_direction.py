"""Collect one fresh, explicitly non-reproduced exact-push survival direction."""

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
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_fresh_exact_survival_direction_v1.json"
)
CONTRACT_SHA256 = "ccd1b2e333915b220de195f4da3bbd0ff38613c91c18921488835f6cb395b369"
RESULT_FILENAME = "fresh_exact_survival_direction_result.json"
DIRECTION_FILENAME = "fresh_exact_survival_direction.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("fresh exact-survival contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    collection = body.get("collection", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_fresh_exact_survival_direction_contract_v1"
        or collection.get("num_envs") != engine.NUM_ENVS
        or collection.get("steps") != engine.COLLECTION_STEPS
        or collection.get("all_envs_receive_exact_vector") is not True
        or collection.get("impulse_transition") != engine.IMPULSE_TRANSITION
        or collection.get("historical_byte_reproduction_required") is not False
        or collection.get("fresh_direction_identity_required") is not True
        or boundaries.get("evaluation_runs") != 0
        or boundaries.get("robot_or_network_commands_permitted") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("fresh exact-survival contract semantic mismatch")
    for name, entry in body["parents"].items():
        parent = Path(entry["path"]).resolve(strict=True)
        if parent.is_symlink() or sha256_file(parent) != entry["sha256"]:
            raise ValueError(f"fresh exact-survival parent mismatch: {name}")
    return body


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    contract = _load_contract(root)
    if os.path.lexists(requested_run_dir):
        raise FileExistsError("fresh exact-survival run directory exists")
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
            raise RuntimeError("fresh exact-survival engine preflight failed")
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
            vector = np.asarray(contract["collection"]["exact_vector"], dtype=np.float32)
            vectors = torch.from_numpy(np.repeat(vector.reshape(1, 6), engine.NUM_ENVS, axis=0)).to(
                device=engine.parent.DEVICE
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
        scaled_sha = fs._state_sha256(scaled)  # noqa: SLF001
        scaled_l2 = engine.DIRECTION_SCALE * float(evidence["target_direction_l2"])
        evidence["scaled_direction_state_sha256"] = scaled_sha
        evidence["scaled_direction_l2_target"] = scaled_l2
        path = run_dir / "checkpoints" / DIRECTION_FILENAME
        engine._write_checkpoint(  # noqa: SLF001
            path,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_rank256_fresh_exact_survival_direction_v1",
                "contract_sha256": CONTRACT_SHA256,
                "direction_state_dict": scaled,
                "direction_state_sha256": scaled_sha,
                "direction_l2": scaled_l2,
                "historical_byte_reproduction_claimed": False,
                "optimizer_steps": 0,
                "critic_updates": 0,
                "hardware_authorized": False,
            },
        )
        artifact = {"path": str(path), "sha256": sha256_file(path)}
        report = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_fresh_exact_survival_direction_result_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_overlay": overlay,
            "task_audit": task_audit,
            "prime": prime,
            "gradient_evidence": evidence,
            "direction_artifact": artifact,
            "historical_byte_reproduction_claimed": False,
            "training_transitions": engine.TOTAL_TRANSITIONS,
            "evaluation_runs": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "candidate_selected": False,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        engine._materials_unchanged(root, audit, "before_result")  # noqa: SLF001
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
