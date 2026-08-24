"""Fit one bounded survival-score direction from rank256 under q250 impulses."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_survival_length_score as parent,
    train_g1_true23_sonic_task_space_ppo_full_support as full_support,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import ZERO_COUNTS

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_disturbance_survival_score_v1.json"
)
CONTRACT_SHA256 = "82e65b98e20bd4070a904cdb14e779ed63a8ea639ff81a2cedbd90aa0b681d2d"
KIND = "g1_true23_sonic_rank256_disturbance_survival_score_result_v1"
NUM_ENVS = 128
COLLECTION_STEPS = 510
TOTAL_TRANSITIONS = NUM_ENVS * COLLECTION_STEPS
IMPULSE_TRANSITION = 241
EXPLORATION_STD = 0.05
DIRECTION_SCALE = 0.25
SCALES = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
NOMINAL_SEED = 835868017
DISTURBANCE_SEED = 2069156915
FAILED_VECTOR = np.asarray(
    [-0.109284965708, 0.348842165152, -0.026012141987, 0.137065557322, -0.499675156302, -0.717607152875],
    dtype=np.float32,
)
RESULT_FILENAME = "rank256_disturbance_survival_score_result.json"
FAILURE_FILENAME = "rank256_disturbance_survival_score_failure.json"
CHECKPOINT_FILENAME = "rank256_disturbance_survival_score_candidate.pt"
DIRECTION_FILENAME = "rank256_disturbance_survival_score_direction.pt"
COLLECT_AND_SCORE = parent.collect_and_score
EXTRA_SOURCE_RELATIVE_PATHS: tuple[Path, ...] = ()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("rank256 disturbance-survival contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, Mapping):
        raise TypeError("rank256 disturbance-survival contract must be mapping")
    collection = body.get("collection", {})
    gradient = body.get("gradient", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_disturbance_survival_score_contract_v1"
        or collection.get("num_envs") != NUM_ENVS
        or collection.get("steps") != COLLECTION_STEPS
        or collection.get("total_transitions") != TOTAL_TRANSITIONS
        or collection.get("impulse_global_transition") != IMPULSE_TRANSITION
        or collection.get("exploration_std") != EXPLORATION_STD
        or gradient.get("scales") != list(SCALES)
        or gradient.get("optimizer_steps") != 0
        or gradient.get("critic_updates") != 0
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("rank256 disturbance-survival contract semantic mismatch")
    for name, entry in body["inputs"].items():
        raw = Path(entry["path"])
        path = raw if raw.is_absolute() else root / raw
        path = path.resolve(strict=True)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"rank256 disturbance-survival input mismatch: {name}")
    return body


def _rank256_state(root: Path, contract: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    import onnx

    inputs = contract["inputs"]
    source_path = Path(inputs["source_checkpoint"]["path"]).resolve(strict=True)
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    validated = task_space.validate_mjlab_training_checkpoint(payload)
    source = validated["policy_state_dict"]
    state = {name: tensor.detach().cpu().to(torch.float32).contiguous().clone() for name, tensor in source.items()}
    if len(state) != 29 or "std" not in state:
        raise ValueError("rank256 source policy tensor namespace mismatch")
    decoder_path = (root / inputs["candidate_decoder"]["path"]).resolve(strict=True)
    initializers = task_space._onnx_initializers(onnx.load(str(decoder_path)))  # noqa: SLF001
    expected = {f"layers.{layer}.{suffix}" for layer in range(9) for suffix in ("weight", "bias")}
    if set(initializers) != expected:
        raise ValueError("rank256 decoder initializer namespace mismatch")
    for layer in range(8):
        module = layer * 2
        for suffix in ("weight", "bias"):
            source_name = f"actor_module.decoders.g1_dyn.module.{module}.{suffix}"
            if not np.array_equal(state[source_name].numpy(), initializers[f"layers.{layer}.{suffix}"]):
                raise ValueError(f"rank256 decoder trunk mismatch: layer {layer} {suffix}")
    weight = initializers["layers.8.weight"]
    bias = initializers["layers.8.bias"]
    if weight.shape != (23, 512) or bias.shape != (23,):
        raise ValueError("rank256 final affine shape mismatch")
    state["actor_module.decoders.g1_dyn.module.16.weight"] = torch.from_numpy(weight.copy())
    state["actor_module.decoders.g1_dyn.module.16.bias"] = torch.from_numpy(bias.copy())
    state["std"] = torch.full((23,), EXPLORATION_STD, dtype=torch.float32)
    policy_sha = inspect_true23_policy_state(
        {"policy_state_dict": state},
        reference_profile=fs.REFERENCE_PROFILE,
    )
    return state, {
        "source_checkpoint_sha256": inputs["source_checkpoint"]["sha256"],
        "candidate_decoder_sha256": inputs["candidate_decoder"]["sha256"],
        "policy_state_sha256": policy_sha,
        "exploration_std": EXPLORATION_STD,
        "decoder_trunk_tensor_match_count": 16,
        "final_affine_overlay_tensor_count": 2,
    }


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        state, overlay = _rank256_state(root, contract)
        base_contract = task_space.load_task_space_ppo_contract(root)
        topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
            strict=True
        )
        motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
        source_paths = (
            Path("gear_sonic/scripts/train_g1_true23_sonic_rank256_disturbance_survival_score.py"),
            *EXTRA_SOURCE_RELATIVE_PATHS,
        )
        sources = {CONTRACT_RELATIVE_PATH.as_posix(): sha256_file(root / CONTRACT_RELATIVE_PATH)}
        sources.update(
            {relative.as_posix(): sha256_file((root / relative).resolve(strict=True)) for relative in source_paths}
        )
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        del state
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_disturbance_survival_score_preflight_v1",
            "ready": True,
            "contract": contract,
            "overlay": overlay,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_disturbance_survival_score_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "training_transitions": 0,
            "gradient_computations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _materials_unchanged(root: Path, audit: Mapping[str, Any], phase: str) -> None:
    current = preflight(root)
    if current.get("ready") is not True or current.get("material_manifest_sha256") != audit.get(
        "material_manifest_sha256"
    ):
        raise RuntimeError(f"rank256 disturbance-survival materials changed: {phase}")


def _impulse_vectors(num_envs: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(20260812)
    maxima = np.asarray([0.5, 0.5, 0.2, 0.52, 0.52, 0.78], dtype=np.float32)
    values = rng.uniform(-maxima, maxima, size=(num_envs, 6)).astype(np.float32)
    values[0] = FAILED_VECTOR
    return torch.from_numpy(values).to(device=device)


def _apply_impulse(wrapped: Any, vectors: torch.Tensor) -> None:
    raw = wrapped.unwrapped
    robot = raw.scene["robot"]
    before = robot.data.root_link_vel_w
    if before.shape != vectors.shape or before.dtype != torch.float32 or not bool(torch.isfinite(before).all()):
        raise RuntimeError("rank256 disturbance root velocity ABI mismatch")
    target = before.detach().clone() + vectors
    env_ids = torch.arange(int(raw.num_envs), dtype=torch.long, device=before.device)
    robot.write_root_link_velocity_to_sim(target, env_ids=env_ids)


class _ImpulseProxy:
    def __init__(self, wrapped: Any, vectors: torch.Tensor) -> None:
        self._wrapped = wrapped
        self._vectors = vectors
        self._transition = 0

    @property
    def unwrapped(self) -> Any:
        return self._wrapped.unwrapped

    def get_observations(self) -> Any:
        return self._wrapped.get_observations()

    def __getattr__(self, name: str) -> Any:
        """Preserve complete wrapper ABI while intercepting only ``step``."""

        return getattr(self._wrapped, name)

    def step(self, actions: torch.Tensor) -> Any:
        if self._transition == IMPULSE_TRANSITION:
            _apply_impulse(self._wrapped, self._vectors)
        result = self._wrapped.step(actions)
        self._transition += 1
        return result


@contextmanager
def _collection_scope() -> Iterator[None]:
    names = ("NUM_ENVS", "COLLECTION_STEPS", "TOTAL_TRANSITIONS", "GRADIENT_BATCH_SIZE", "SCALES")
    saved = {name: getattr(parent, name) for name in names}
    try:
        parent.NUM_ENVS = NUM_ENVS
        parent.COLLECTION_STEPS = COLLECTION_STEPS
        parent.TOTAL_TRANSITIONS = TOTAL_TRANSITIONS
        parent.GRADIENT_BATCH_SIZE = 512
        parent.SCALES = SCALES
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def _actor(state: Mapping[str, torch.Tensor], topology: Path) -> Any:
    saved = parent.SOURCE_POLICY_SHA256
    policy_sha = inspect_true23_policy_state(
        {"policy_state_dict": state},
        reference_profile=fs.REFERENCE_PROFILE,
    )
    try:
        parent.SOURCE_POLICY_SHA256 = policy_sha
        return parent._actor(state, topology)  # noqa: SLF001
    finally:
        parent.SOURCE_POLICY_SHA256 = saved


def _scaled_direction(direction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.mul(DIRECTION_SCALE).contiguous() for name, value in direction.items()}


def _states(
    baseline: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
) -> dict[float, dict[str, torch.Tensor]]:
    result: dict[float, dict[str, torch.Tensor]] = {}
    for scale in SCALES:
        state = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
        for name, delta in direction.items():
            state[name] = torch.add(state[name], delta, alpha=scale).contiguous()
        result[scale] = state
    return result


def _evaluate(
    *,
    state: Mapping[str, torch.Tensor],
    scale: float,
    scenario: str,
    topology: Path,
    motion: Path,
    material_sha: str,
) -> dict[str, Any]:
    import mjlab.rl as mjlab_rl

    original_seed = fs.FIXED_SEED
    original_wrapper = mjlab_rl.RslRlVecEnvWrapper
    seed = NOMINAL_SEED if scenario == "nominal" else DISTURBANCE_SEED

    if scenario == "disturbance":

        class ImpulseWrapper(original_wrapper):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._rank256_transition = 0
                self._rank256_vectors = FAILED_VECTOR.reshape(1, 6)

            def step(self, actions: torch.Tensor) -> Any:
                if self._rank256_transition == IMPULSE_TRANSITION:
                    vector = torch.from_numpy(self._rank256_vectors).to(
                        device=self.unwrapped.device,
                        dtype=torch.float32,
                    )
                    _apply_impulse(self, vector)
                result = super().step(actions)
                self._rank256_transition += 1
                return result

        mjlab_rl.RslRlVecEnvWrapper = ImpulseWrapper
    try:
        fs.FIXED_SEED = seed
        record = parent._evaluate_state(  # noqa: SLF001
            state=state,
            scale=scale,
            topology=topology,
            motion=motion,
            material_manifest_sha256=material_sha,
        )
    finally:
        fs.FIXED_SEED = original_seed
        mjlab_rl.RslRlVecEnvWrapper = original_wrapper
    record["scenario"] = scenario
    record["evaluation_seed"] = seed
    record["impulse_applied"] = (
        scenario == "disturbance" and record.get("completed_transitions", 0) > IMPULSE_TRANSITION
    )
    return record


def _clean(record: Mapping[str, Any]) -> bool:
    return all(record.get(name) == 0 for name in ZERO_COUNTS)


def _assess(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(float(record["scale"]), str(record["scenario"])): record for record in records}
    baseline_nominal = by_key[(0.0, "nominal")]
    baseline_disturbance = by_key[(0.0, "disturbance")]
    eligible: list[Mapping[str, Any]] = []
    for scale in SCALES:
        if scale == 0.0:
            continue
        nominal = by_key[(scale, "nominal")]
        disturbance = by_key[(scale, "disturbance")]
        if (
            _clean(nominal)
            and _clean(disturbance)
            and int(nominal.get("completed_transitions", -1))
            >= int(baseline_nominal.get("completed_transitions", -1))
            and int(disturbance.get("completed_transitions", -1))
            > int(baseline_disturbance.get("completed_transitions", -1))
        ):
            eligible.append(
                {
                    "scale": scale,
                    "nominal": nominal,
                    "disturbance": disturbance,
                }
            )
    selected = max(
        eligible,
        key=lambda item: (
            min(
                int(item["nominal"]["completed_transitions"]),
                int(item["disturbance"]["completed_transitions"]),
            ),
            int(item["nominal"]["completed_transitions"]) + int(item["disturbance"]["completed_transitions"]),
            -abs(float(item["scale"])),
        ),
        default=None,
    )
    return {
        "baseline_nominal_completed_transitions": baseline_nominal.get("completed_transitions"),
        "baseline_nominal_terminal_q9": baseline_nominal.get("terminal_q9"),
        "baseline_disturbance_completed_transitions": baseline_disturbance.get("completed_transitions"),
        "baseline_disturbance_terminal_q9": baseline_disturbance.get("terminal_q9"),
        "candidate_selected": selected is not None,
        "selected_scale": None if selected is None else selected["scale"],
        "selected_policy_state_sha256": (None if selected is None else selected["nominal"]["policy_state_sha256"]),
        "selected_nominal_completed_transitions": (
            None if selected is None else selected["nominal"]["completed_transitions"]
        ),
        "selected_disturbance_completed_transitions": (
            None if selected is None else selected["disturbance"]["completed_transitions"]
        ),
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError("rank256 disturbance-survival checkpoint exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("rank256 disturbance-survival preflight failed")
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

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(fs.FIXED_SEED)
    np.random.seed(fs.FIXED_SEED % (2**32))
    torch.manual_seed(fs.FIXED_SEED)
    torch.cuda.manual_seed_all(fs.FIXED_SEED)
    contract = audit["contract"]
    baseline, overlay = _rank256_state(root, contract)
    base_contract = task_space.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    run_dir = full_support._create_run_dir_exclusive(requested_run_dir)
    full_support._write_json_exclusive(run_dir / "preflight.json", audit)
    full_support._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])
    cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=NUM_ENVS)
    cfg.seed = fs.FIXED_SEED
    task_audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=cfg, device=parent.DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        actor = _actor(baseline, topology)
        vectors = _impulse_vectors(NUM_ENVS, torch.device(parent.DEVICE))
        _materials_unchanged(root, audit, "before_collection")
        with _collection_scope():
            direction, evidence = COLLECT_AND_SCORE(
                actor=actor,
                wrapped=_ImpulseProxy(wrapped, vectors),
            )
    finally:
        env.close()
    direction = _scaled_direction(direction)
    evidence["unscaled_direction_l2_target"] = evidence.get("target_direction_l2")
    evidence["direction_scale"] = DIRECTION_SCALE
    evidence["scaled_direction_state_sha256"] = fs._state_sha256(direction)
    evidence["scaled_direction_l2_target"] = float(contract["gradient"]["direction_l2"])
    evidence["impulse_transition"] = IMPULSE_TRANSITION
    evidence["failed_vector_environment_index"] = 0
    full_support._write_json_exclusive(run_dir / "gradient_evidence.json", evidence)
    direction_path = run_dir / "checkpoints" / DIRECTION_FILENAME
    _write_checkpoint(
        direction_path,
        {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_survival_score_direction_v1",
            "contract_sha256": CONTRACT_SHA256,
            "direction_state_dict": direction,
            "direction_state_sha256": evidence["scaled_direction_state_sha256"],
            "direction_l2": evidence["scaled_direction_l2_target"],
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        },
    )
    direction_artifact = {
        "path": str(direction_path),
        "sha256": sha256_file(direction_path),
    }
    del actor
    torch.cuda.empty_cache()
    states = _states(baseline, direction)
    records: list[dict[str, Any]] = []
    evaluations_dir = run_dir / "evaluations"
    if not evaluations_dir.is_dir() or evaluations_dir.is_symlink():
        raise RuntimeError("rank256 disturbance-survival evaluations directory missing")
    for scale in SCALES:
        for scenario in ("nominal", "disturbance"):
            _materials_unchanged(root, audit, f"before_evaluation_{scale}_{scenario}")
            record = _evaluate(
                state=states[scale],
                scale=scale,
                scenario=scenario,
                topology=topology,
                motion=motion,
                material_sha=audit["material_manifest_sha256"],
            )
            full_support._write_json_exclusive(
                evaluations_dir / f"scale_{str(scale).replace('.', '_')}_{scenario}.json",
                record,
            )
            print(  # noqa: T201
                " ".join(
                    (
                        f"scale={scale}",
                        f"scenario={scenario}",
                        f"steps={record['completed_transitions']}",
                        f"q9={record['terminal_q9']}",
                    )
                ),
                flush=True,
            )
            records.append(record)
    assessment = _assess(records)
    candidate: dict[str, Any] | None = None
    if assessment["candidate_selected"] is True:
        scale = float(assessment["selected_scale"])
        checkpoint = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_disturbance_survival_score_candidate_v1",
            "contract_sha256": CONTRACT_SHA256,
            "source_overlay": overlay,
            "policy_state_dict": states[scale],
            "policy_state_sha256": assessment["selected_policy_state_sha256"],
            "selected_scale": scale,
            "gradient_evidence": evidence,
            "training_transitions": TOTAL_TRANSITIONS,
            "gradient_computations": 1,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "support_qualified": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        path = run_dir / "checkpoints" / CHECKPOINT_FILENAME
        _write_checkpoint(path, checkpoint)
        candidate = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema_version": 1,
        "kind": KIND,
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "task_audit": task_audit,
        "prime": prime,
        "source_overlay": overlay,
        "gradient_evidence": evidence,
        "direction_artifact": direction_artifact,
        "evaluations": records,
        "assessment": assessment,
        "candidate": candidate,
        "training_transitions": TOTAL_TRANSITIONS,
        "gradient_computations": 1,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _materials_unchanged(root, audit, "before_result")
    full_support._write_json_exclusive(run_dir / RESULT_FILENAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    train = sub.add_parser("train")
    train.add_argument("--repository-root", type=Path, default=ROOT)
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
        if run_dir.is_dir() and not run_dir.is_symlink() and not (run_dir / FAILURE_FILENAME).exists():
            full_support._write_json_exclusive(
                run_dir / FAILURE_FILENAME,
                {
                    "schema_version": 1,
                    "kind": "g1_true23_sonic_rank256_disturbance_survival_score_failure_v1",
                    "contract_sha256": CONTRACT_SHA256,
                    "error_type": type(error).__name__,
                    "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                    "candidate": None,
                    "support_qualified": False,
                    "deployment_ready": False,
                    "hardware_authorized": False,
                },
            )
        print(f"rank256 disturbance-survival failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True))  # noqa: T201
    return 0 if result.get("candidate") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
