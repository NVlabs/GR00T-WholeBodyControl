"""Qualify hash-bound recovery/late SONIC blend in clean-observation MJLab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from gear_sonic.scripts import (
    qualify_g1_true23_sonic_rank256_phase_balance as phase,
    train_g1_true23_sonic_rank256_disturbance_survival_score as disturbance,
)
from gear_sonic.trl.mjlab import sonic_task_space_ppo_runner as task_space
from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import (
    BLEND_DURATION_STEPS,
    BLEND_START_Q9,
    LATE_CHECKPOINT_PATH,
    LATE_CHECKPOINT_SHA256,
    LATE_POLICY_SHA256,
    RECOVERY_CHECKPOINT_PATH,
    RECOVERY_CHECKPOINT_SHA256,
    RECOVERY_POLICY_SHA256,
    SonicRecoveryBlendPolicy,
    load_hash_bound_actor,
)

NUM_ENVS = 128
EPISODE_STEPS = 510
SEED = 835868017
DEVICE = "cuda:0"
DEFAULT_OUTPUT = Path("artifacts/g1_true23/sonic_recovery_blend_clean_q360_width10_qualification_v2.json")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite recovery blend evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(repository_root: Path, output: Path) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    root = repository_root.expanduser().resolve(strict=True)
    contract, _shifted, _overlay, topology, motion = phase._inputs(root)  # noqa: SLF001
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    _seed(SEED)
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    cfg.seed = SEED
    noisy_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=NUM_ENVS)
    cfg.observations["tokenizer"].enable_corruption = False
    cfg.observations["policy"].enable_corruption = False
    if (
        cfg.observations["tokenizer"].enable_corruption is not False
        or cfg.observations["policy"].enable_corruption is not False
    ):
        raise RuntimeError("recovery blend failed to disable artificial observation corruption")
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    simulator_steps = 0
    try:
        wrapped_base = RslRlVecEnvWrapper(env, clip_actions=None)
        vectors = torch.zeros((NUM_ENVS, 6), dtype=torch.float32, device=DEVICE)
        exact = torch.tensor(contract["qualification"]["disturbance_vectors"][0], device=DEVICE)
        opposite = torch.tensor(contract["qualification"]["disturbance_vectors"][1], device=DEVICE)
        vectors[32:64] = exact
        vectors[64:96] = opposite
        rng = np.random.default_rng(SEED)
        maxima = np.asarray([0.5, 0.5, 0.2, 0.52, 0.52, 0.78], dtype=np.float32)
        random_vectors = rng.uniform(-maxima, maxima, size=(32, 6)).astype(np.float32)
        vectors[96:128] = torch.from_numpy(random_vectors).to(DEVICE)
        wrapped = disturbance._ImpulseProxy(wrapped_base, vectors)  # noqa: SLF001
        prime = prime_sonic_true23_training_environment(wrapped)
        recovery_actor = load_hash_bound_actor(
            RECOVERY_CHECKPOINT_PATH,
            topology_checkpoint_path=topology,
            expected_checkpoint_sha256=RECOVERY_CHECKPOINT_SHA256,
            expected_policy_sha256=RECOVERY_POLICY_SHA256,
            device=DEVICE,
        )
        late_actor = load_hash_bound_actor(
            LATE_CHECKPOINT_PATH,
            topology_checkpoint_path=topology,
            expected_checkpoint_sha256=LATE_CHECKPOINT_SHA256,
            expected_policy_sha256=LATE_POLICY_SHA256,
            device=DEVICE,
        )
        actor = SonicRecoveryBlendPolicy(recovery_actor, late_actor, env).to(DEVICE)
        observations = wrapped.get_observations().to(DEVICE)
        active = torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE)
        lengths = torch.full((NUM_ENVS,), EPISODE_STEPS, dtype=torch.long, device=DEVICE)
        nonfinite_count = 0
        raw_clip_count = 0
        for step in range(EPISODE_STEPS):
            with torch.no_grad():
                action = actor(observations, stochastic_output=False)
                nonfinite_count += int(torch.count_nonzero(~torch.isfinite(action)).detach().cpu())
                raw_clip_count += int(
                    torch.count_nonzero(torch.amax(torch.abs(action), dim=1) >= 10.0).detach().cpu()
                )
                observations, _rewards, dones, extras = wrapped.step(action)
                simulator_steps += NUM_ENVS
                extras["log"] = {}
                observations = observations.to(DEVICE)
                done = dones.to(dtype=torch.bool, device=DEVICE)
                newly_done = active & done
                lengths[newly_done] = step + 1
                active &= ~done
        values = lengths.detach().cpu().tolist()
    finally:
        env.close()

    groups: dict[str, dict[str, Any]] = {}
    for name, start in (("nominal", 0), ("exact", 32), ("opposite", 64), ("random", 96)):
        subset = values[start : start + 32]
        groups[name] = {
            "completed_transitions": subset,
            "minimum": min(subset),
            "median": float(np.median(subset)),
            "survivors": sum(value == EPISODE_STEPS for value in subset),
        }
    passed = (
        groups["nominal"]["survivors"] == 32
        and groups["exact"]["survivors"] == 32
        and groups["opposite"]["survivors"] == 32
        and groups["random"]["survivors"] >= 31
        and nonfinite_count == 0
        and raw_clip_count == 0
    )
    report = {
        "schema_version": 2,
        "kind": "g1_true23_sonic_recovery_blend_clean_qualification_v2",
        "passed": passed,
        "claim": "simulator_clean_observation_recovery_blend_candidate_only",
        "controller": {
            "recovery_checkpoint_path": str(RECOVERY_CHECKPOINT_PATH),
            "recovery_checkpoint_sha256": RECOVERY_CHECKPOINT_SHA256,
            "recovery_policy_sha256": RECOVERY_POLICY_SHA256,
            "late_checkpoint_path": str(LATE_CHECKPOINT_PATH),
            "late_checkpoint_sha256": LATE_CHECKPOINT_SHA256,
            "late_policy_sha256": LATE_POLICY_SHA256,
            "blend_start_q9": BLEND_START_Q9,
            "blend_duration_steps": BLEND_DURATION_STEPS,
            "blend_formula": "smoothstep(alpha)=alpha^2*(3-2*alpha)",
            "history_reset_at_handoff": False,
            "action_substitution": False,
        },
        "environment": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "episode_steps": EPISODE_STEPS,
            "prime": prime,
            "base_task_audit_before_clean_mutation": noisy_audit,
            "artificial_tokenizer_corruption": False,
            "artificial_policy_corruption": False,
            "physical_domain_randomization": False,
        },
        "groups": groups,
        "all": {
            "survivors": sum(value == EPISODE_STEPS for value in values),
            "minimum": min(values),
            "median": float(np.median(values)),
            "simulator_transition_calls": simulator_steps,
        },
        "nonfinite_action_count": nonfinite_count,
        "raw_clip_required_count": raw_clip_count,
        "training_updates": 0,
        "optimizer_steps": 0,
        "teacher_labels_used": False,
        "single_policy_export_available": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    _write_json_exclusive(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.repository_root, args.output)
    print(json.dumps({"passed": report["passed"], "all": report["all"]}, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
