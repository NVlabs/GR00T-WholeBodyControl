"""Nominal multi-seed teacher-actuated dataset collection for SONIC true23."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils import (
    g1_true23_native124_21204_bootstrap_mjlab as bootstrap,
    g1_true23_sonic_recovery_qualification_campaign as base_campaign,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_selected_teacher_nominal_multiseed_collection_contract_v1"
MANIFEST_KIND = "g1_true23_selected_teacher_nominal_multiseed_manifest_v1"
FAILURE_KIND = "g1_true23_selected_teacher_nominal_multiseed_failure_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_selected_teacher_nominal_multiseed_collection_v1.json"
)
CONTRACT_SHA256 = "5d693bfab86c063b0943fba75fd39427b90192c0981f2b77fb33ab397e0ac121"
EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_selected_teacher_nominal_multiseed_collection.py"),
    Path("gear_sonic/scripts/collect_g1_true23_selected_teacher_nominal_multiseed.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_bootstrap_mjlab.py"),
    Path("gear_sonic/utils/g1_true23_sonic_recovery_qualification_campaign.py"),
)

NOMINAL_SEEDS = (
    611723381,
    519990690,
    1277059621,
    677255333,
    1807192292,
    874120590,
    446813377,
    801834496,
    835868017,
    921108064,
)
TRAINING_RUN_INDICES = tuple(range(8))
HELDOUT_RUN_INDICES = (8, 9)
ROWS_PER_RUN = bootstrap.TOTAL_ROWS
TRAINING_ROWS = len(TRAINING_RUN_INDICES) * ROWS_PER_RUN


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"unsupported nominal collection manifest value: {type(value).__qualname__}")


@dataclass(frozen=True)
class CollectionRequest:
    repository_root: Path
    output_prefix: Path

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("nominal collection repository root invalid")
        return value

    @property
    def prefix(self) -> Path:
        evidence_root = (self.root / "artifacts" / "g1_true23").resolve(strict=True)
        raw = self.output_prefix.expanduser()
        if raw.suffix:
            raise ValueError("nominal collection output prefix cannot have suffix")
        candidate = raw if raw.is_absolute() else self.root / raw
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(evidence_root):
            raise ValueError("nominal collection output must stay under artifacts/g1_true23")
        if value.parent.is_symlink() or not value.parent.is_dir():
            raise ValueError("nominal collection output parent invalid")
        return value

    @property
    def npz_path(self) -> Path:
        return Path(f"{self.prefix}.npz")

    @property
    def manifest_path(self) -> Path:
        return Path(f"{self.prefix}.manifest.json")

    @property
    def failure_path(self) -> Path:
        return Path(f"{self.prefix}.failure.json")


def _source_binding(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"nominal collection source invalid: {relative.as_posix()}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "binding_sha256": _sha256_bytes(_canonical_bytes(files))}


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("nominal collection contract mismatch")
    contract = base_campaign._strict_json(path, "nominal collection contract")  # noqa: SLF001
    cohort = _mapping(contract.get("cohort"), "nominal collection cohort")
    runtime = _mapping(contract.get("runtime"), "nominal collection runtime")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "teacher_actuated_nominal_multiseed_behavior_cloning_collection_only"
        or cohort.get("nominal_seeds") != list(NOMINAL_SEEDS)
        or cohort.get("training_run_indices") != list(TRAINING_RUN_INDICES)
        or cohort.get("heldout_run_indices") != list(HELDOUT_RUN_INDICES)
        or cohort.get("rows_per_run") != ROWS_PER_RUN
        or cohort.get("published_training_row_count") != TRAINING_ROWS
        or cohort.get("heldout_teacher_label_rows_published") != 0
        or cohort.get("fail_fast") is not True
        or runtime.get("actions_per_run") != ROWS_PER_RUN
        or runtime.get("model_output_validation")
        != "hash_bound_live_per_row_outputs_plus_full_nonmodel_reassessment_without_duplicate_cpu_replay"
        or runtime.get("teacher_controlled_from_q9") != 9
        or runtime.get("post_collection_q9") != 519
        or runtime.get("autoreset_permitted") is not False
    ):
        raise ValueError("nominal collection contract semantics mismatch")
    return contract


def _verify_prerequisites(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    prerequisite = _mapping(contract["prerequisites"], "nominal collection prerequisites")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in (
        "teacher_q9_pass_report",
        "teacher_campaign_failure",
        "bootstrap_contract",
        "bootstrap_npz",
        "bootstrap_manifest",
    ):
        path = (root / str(prerequisite[f"{name}_relative_path"])).resolve(strict=True)
        expected = str(prerequisite[f"{name}_sha256"])
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"nominal collection prerequisite mismatch: {name}")
        artifacts[name] = {"path": str(path), "sha256": expected, "size_bytes": path.stat().st_size}
    q9 = base_campaign._strict_json(  # noqa: SLF001
        Path(artifacts["teacher_q9_pass_report"]["path"]), "teacher q9 pass"
    )
    campaign = base_campaign._strict_json(  # noqa: SLF001
        Path(artifacts["teacher_campaign_failure"]["path"]), "teacher campaign failure"
    )
    if (
        q9.get("qualification", {}).get("passed") is not True
        or q9.get("mode") != "q9"
        or q9.get("controller_counts") != {"student": 0, "teacher": 510}
        or q9.get("first_done", {}).get("termination_names") != ["time_out"]
        or campaign.get("campaign_qualified") is not False
        or campaign.get("nominal_passed_count") != 1
        or campaign.get("disturbance_passed_count") != 0
        or campaign.get("published_teacher_label_count") != 0
    ):
        raise ValueError("nominal collection prerequisite semantics mismatch")
    bootstrap.load_bootstrap_contract(root)
    bootstrap.load_bootstrap_training_candidate(
        artifacts["bootstrap_npz"]["path"],
        artifacts["bootstrap_manifest"]["path"],
        repository_root=root,
    )
    nominal_specs = tuple(spec for spec in base_campaign.campaign_run_specs(root) if spec.scenario == "nominal")
    if tuple(spec.seed for spec in nominal_specs) != NOMINAL_SEEDS:
        raise ValueError("nominal collection seed cohort drift")
    return {"artifacts": artifacts, "nominal_specs": nominal_specs}


def _inner_request(root: Path) -> bootstrap.BootstrapCollectionRequest:
    return bootstrap.BootstrapCollectionRequest(
        root,
        root / "artifacts/g1_true23/.nominal-multiseed-inner-unused",
    )


@contextmanager
def _runtime_seed_scope(seed: int):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("nominal collection runtime seed invalid")
    original = bootstrap.FIXED_SEED
    bootstrap.FIXED_SEED = seed
    try:
        yield
    finally:
        bootstrap.FIXED_SEED = original


@contextmanager
def _capture_effective_env_seed(expected_seed: int, observed: list[int]):
    import mjlab.envs as mjlab_envs

    original = mjlab_envs.ManagerBasedRlEnv

    class InstrumentedManagerBasedRlEnv(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            cfg = kwargs.get("cfg") if "cfg" in kwargs else args[0]
            value = getattr(cfg, "seed", None)
            if value != expected_seed:
                raise RuntimeError("nominal collection effective env seed mismatch")
            observed.append(value)
            super().__init__(*args, **kwargs)

    mjlab_envs.ManagerBasedRlEnv = InstrumentedManagerBasedRlEnv
    try:
        yield
    finally:
        mjlab_envs.ManagerBasedRlEnv = original


@contextmanager
def _defer_duplicate_model_replay_assessment():
    original = bootstrap.assess_bootstrap_arrays

    def runtime_checked_assessment(
        arrays: Mapping[str, np.ndarray], *, repository_root: str | Path | None = None
    ) -> bootstrap.BootstrapAssessment:
        del repository_root
        bootstrap._validated_arrays(arrays)  # noqa: SLF001
        return bootstrap.BootstrapAssessment(())

    bootstrap.assess_bootstrap_arrays = runtime_checked_assessment
    try:
        yield
    finally:
        bootstrap.assess_bootstrap_arrays = original


def _assess_recorded_model_outputs(arrays: Mapping[str, np.ndarray], root: Path) -> bootstrap.BootstrapAssessment:
    values = bootstrap._validated_arrays(arrays)  # noqa: SLF001
    key = (
        str(root),
        bootstrap._array_sha256(values["encoder267"]),  # noqa: SLF001
        bootstrap._array_sha256(values["selected_observation124"]),  # noqa: SLF001
    )
    previous = bootstrap._MODEL_OUTPUT_CACHE.get(key)  # noqa: SLF001
    bootstrap._MODEL_OUTPUT_CACHE[key] = (  # noqa: SLF001
        values["token64"].copy(),
        values["teacher_action_onnx_hardware23"].copy(),
    )
    try:
        return bootstrap.assess_bootstrap_arrays(values, repository_root=root)
    finally:
        if previous is None:
            bootstrap._MODEL_OUTPUT_CACHE.pop(key, None)  # noqa: SLF001
        else:
            bootstrap._MODEL_OUTPUT_CACHE[key] = previous  # noqa: SLF001


def _preflight_internal(request: CollectionRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root)
    prerequisites = _verify_prerequisites(root, contract)
    sources = _source_binding(root)
    inner_request = _inner_request(root)
    inner_preflight = bootstrap.preflight_bootstrap_collection(inner_request)
    if inner_preflight.get("ready") is not True:
        raise RuntimeError("nominal collection inner bootstrap preflight not ready")
    return {
        "ready": True,
        "root": root,
        "contract": contract,
        "prerequisites": prerequisites,
        "sources": sources,
        "inner_request": inner_request,
        "inner_preflight": inner_preflight,
    }


def preflight(request: CollectionRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_selected_teacher_nominal_multiseed_preflight_v1",
            "ready": True,
            "contract_sha256": CONTRACT_SHA256,
            "nominal_seed_count": len(NOMINAL_SEEDS),
            "training_run_count": len(TRAINING_RUN_INDICES),
            "heldout_run_count": len(HELDOUT_RUN_INDICES),
            "planned_training_rows": TRAINING_ROWS,
            "executed_source_binding_sha256": value["sources"]["binding_sha256"],
            "inner_bootstrap_contract_sha256": value["inner_preflight"]["contract_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "teacher_labels_admitted": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_selected_teacher_nominal_multiseed_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "teacher_labels_admitted": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }


def _array_sha(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _validate_dataset_arrays(arrays: Mapping[str, np.ndarray], root: Path) -> dict[str, np.ndarray]:
    expected = set(bootstrap.ARRAY_SPECS) | {"collection_run_index", "runtime_seed"}
    if set(arrays) != expected:
        raise ValueError("nominal collection dataset array keys mismatch")
    values: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.shape[0] != TRAINING_ROWS:
            raise ValueError(f"nominal collection array shape mismatch: {name}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nominal collection array nonfinite: {name}")
        values[name] = np.ascontiguousarray(value).copy()
    expected_run = np.repeat(np.arange(len(TRAINING_RUN_INDICES), dtype=np.int64), ROWS_PER_RUN)
    expected_seed = np.concatenate(
        [np.full(ROWS_PER_RUN, NOMINAL_SEEDS[index], dtype=np.int64) for index in TRAINING_RUN_INDICES]
    )
    if (
        values["collection_run_index"].dtype != np.int64
        or values["runtime_seed"].dtype != np.int64
        or not np.array_equal(values["collection_run_index"], expected_run)
        or not np.array_equal(values["runtime_seed"], expected_seed)
    ):
        raise ValueError("nominal collection row lineage mismatch")
    for published_index, _ in enumerate(TRAINING_RUN_INDICES):
        start = published_index * ROWS_PER_RUN
        stop = start + ROWS_PER_RUN
        segment = {name: values[name][start:stop].copy() for name in bootstrap.ARRAY_SPECS}
        assessment = _assess_recorded_model_outputs(segment, root)
        if assessment.quarantined or assessment.bootstrap_bc_eligible_rows != ROWS_PER_RUN:
            raise ValueError("nominal collection segment failed bootstrap reassessment")
    return values


def collect(
    request: CollectionRequest,
    *,
    progress: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    value = _preflight_internal(request)
    root = request.root
    inner_request = value["inner_request"]
    inner_preflight = value["inner_preflight"]
    training: dict[str, list[np.ndarray]] = {name: [] for name in bootstrap.ARRAY_SPECS}
    records: list[dict[str, Any]] = []
    for run_index, seed in enumerate(NOMINAL_SEEDS):
        observed: list[int] = []
        try:
            with (
                _runtime_seed_scope(seed),
                _capture_effective_env_seed(seed, observed),
                _defer_duplicate_model_replay_assessment(),
            ):
                arrays, materials = bootstrap.run_teacher_bootstrap_collection(
                    inner_request,
                    preflight=inner_preflight,
                )
            assessment = _assess_recorded_model_outputs(arrays, root)
            runtime = _mapping(materials["teacher_runtime"], "nominal teacher runtime")
            training_state = _mapping(materials["training_state"], "nominal training state")
            if (
                observed != [seed]
                or assessment.quarantined
                or assessment.bootstrap_bc_eligible_rows != ROWS_PER_RUN
                or runtime.get("parity_violation_count") != 0
                or runtime.get("action_semantics_mismatch_count") != 0
                or training_state.get("training_performed") is not False
                or training_state.get("optimizer_steps") != 0
            ):
                raise RuntimeError("nominal teacher collection run gate failed")
            published = run_index in TRAINING_RUN_INDICES
            if published:
                for name in bootstrap.ARRAY_SPECS:
                    training[name].append(np.ascontiguousarray(arrays[name]).copy())
            record = {
                "run_index": run_index,
                "seed": seed,
                "effective_env_seed": observed[0],
                "passed": True,
                "published_training_rows": ROWS_PER_RUN if published else 0,
                "heldout": run_index in HELDOUT_RUN_INDICES,
                "decoder994_sha256": _array_sha(arrays["decoder994"]),
                "teacher_label_sha256": _array_sha(arrays["teacher_label_raw_native23"]),
                "whole_run_issue_count": len(assessment.issues),
                "teacher_parity_violation_count": runtime["parity_violation_count"],
                "action_semantics_mismatch_count": runtime["action_semantics_mismatch_count"],
                "training_performed": False,
            }
        except Exception as error:
            record = {
                "run_index": run_index,
                "seed": seed,
                "effective_env_seed": observed[0] if observed else None,
                "passed": False,
                "published_training_rows": 0,
                "heldout": run_index in HELDOUT_RUN_INDICES,
                "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
                "training_performed": False,
            }
        records.append(record)
        if progress is not None:
            progress(record)
        if record["passed"] is not True:
            raise RuntimeError(f"nominal collection failed at run {run_index}")
    if len(records) != len(NOMINAL_SEEDS) or not all(record["passed"] for record in records):
        raise RuntimeError("nominal collection cohort incomplete")
    concatenated = {name: np.concatenate(items, axis=0) for name, items in training.items()}
    concatenated["collection_run_index"] = np.repeat(
        np.arange(len(TRAINING_RUN_INDICES), dtype=np.int64), ROWS_PER_RUN
    )
    concatenated["runtime_seed"] = np.concatenate(
        [np.full(ROWS_PER_RUN, NOMINAL_SEEDS[index], dtype=np.int64) for index in TRAINING_RUN_INDICES]
    )
    arrays = _validate_dataset_arrays(concatenated, root)
    sources_after = _source_binding(root)
    if sources_after != value["sources"]:
        raise RuntimeError("nominal collection source bytes changed during collection")
    materials = {
        "contract_sha256": CONTRACT_SHA256,
        "inner_bootstrap_contract_sha256": inner_preflight["contract_sha256"],
        "executed_source_binding": value["sources"],
        "prerequisites": value["prerequisites"]["artifacts"],
        "runs": records,
        "training_run_indices": list(TRAINING_RUN_INDICES),
        "heldout_run_indices": list(HELDOUT_RUN_INDICES),
        "training_rows": TRAINING_ROWS,
        "heldout_teacher_label_rows_published": 0,
    }
    return arrays, _json_safe(materials)


def publish_new(
    request: CollectionRequest,
    arrays: Mapping[str, np.ndarray],
    materials: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    values = _validate_dataset_arrays(arrays, request.root)
    npz = request.npz_path
    manifest = request.manifest_path
    for path in (npz, manifest, request.failure_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite nominal collection artifact: {path}")
    temporary_npz: Path | None = None
    temporary_manifest: Path | None = None
    npz_published = False
    manifest_published = False
    try:
        temporary_npz = bootstrap._write_temporary_npz(npz.parent, values)  # noqa: SLF001
        npz_sha = sha256_file(temporary_npz)
        schema = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _array_sha(value)}
            for name, value in sorted(values.items())
        }
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
            "artifact": {
                "npz_filename": npz.name,
                "npz_sha256": npz_sha,
                "npz_size_bytes": temporary_npz.stat().st_size,
                "array_count": len(values),
                "array_schema": schema,
            },
            "rows": {
                "total": TRAINING_ROWS,
                "runs": len(TRAINING_RUN_INDICES),
                "rows_per_run": ROWS_PER_RUN,
                "teacher_label_rows": TRAINING_ROWS,
                "heldout_teacher_label_rows_published": 0,
            },
            "materials": _json_safe(materials),
            "boundaries": {
                "classification": "teacher_actuated_nominal_multiseed_behavior_cloning_only",
                "support_qualified": False,
                "disturbance_robustness_claimed": False,
                "on_policy_data": False,
                "dagger_data": False,
                "training_performed": False,
                "promotion_eligible": False,
                "deployment_ready": False,
                "hardware_authorized": False,
            },
            "publication": {
                "protocol": "npz_hardlink_then_manifest_hardlink_commit",
                "manifest_required_for_loader": True,
                "overwrite_permitted": False,
            },
        }
        body["manifest_payload_sha256"] = _sha256_bytes(_canonical_bytes(body))
        temporary_manifest = bootstrap._write_temporary_bytes(  # noqa: SLF001
            manifest.parent, _canonical_bytes(body)
        )
        bootstrap._publish_link_new(temporary_npz, npz, "nominal collection NPZ")  # noqa: SLF001
        npz_published = True
        bootstrap._publish_link_new(  # noqa: SLF001
            temporary_manifest, manifest, "nominal collection manifest"
        )
        manifest_published = True
        return npz, manifest, body
    except BaseException:
        if manifest_published and os.path.lexists(manifest):
            manifest.unlink()
        if npz_published and os.path.lexists(npz):
            npz.unlink()
        raise
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


def load_training_candidate(request: CollectionRequest) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    npz = request.npz_path
    manifest = request.manifest_path
    body = base_campaign._strict_json(manifest, "nominal collection manifest")  # noqa: SLF001
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != MANIFEST_KIND
        or body.get("contract") != {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        or body.get("artifact", {}).get("npz_filename") != npz.name
        or body.get("artifact", {}).get("npz_sha256") != sha256_file(npz)
        or body.get("rows", {}).get("total") != TRAINING_ROWS
        or body.get("rows", {}).get("heldout_teacher_label_rows_published") != 0
        or body.get("boundaries", {}).get("classification")
        != "teacher_actuated_nominal_multiseed_behavior_cloning_only"
        or body.get("boundaries", {}).get("support_qualified") is not False
        or body.get("boundaries", {}).get("training_performed") is not False
    ):
        raise ValueError("nominal collection manifest semantics mismatch")
    expected_payload = dict(body)
    observed_payload_sha = expected_payload.pop("manifest_payload_sha256", None)
    if observed_payload_sha != _sha256_bytes(_canonical_bytes(expected_payload)):
        raise ValueError("nominal collection manifest payload hash mismatch")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]).copy() for name in archive.files}
    values = _validate_dataset_arrays(arrays, request.root)
    schema = body["artifact"]["array_schema"]
    for name, value in values.items():
        if schema.get(name) != {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha(value),
        }:
            raise ValueError(f"nominal collection array manifest mismatch: {name}")
    return values, body


def write_failure_new(request: CollectionRequest, error: Exception) -> Path:
    if os.path.lexists(request.npz_path) or os.path.lexists(request.manifest_path):
        raise RuntimeError("nominal collection failure cannot coexist with dataset")
    path = request.failure_path
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite nominal collection failure: {path}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_KIND,
        "contract_sha256": CONTRACT_SHA256,
        "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
        "npz_published": False,
        "manifest_published": False,
        "teacher_labels_admitted": 0,
        "training_performed": False,
        "support_qualified": False,
        "hardware_authorized": False,
    }
    temporary = bootstrap._write_temporary_bytes(path.parent, _canonical_bytes(body))  # noqa: SLF001
    try:
        bootstrap._publish_link_new(temporary, path, "nominal collection failure")  # noqa: SLF001
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "CollectionRequest",
    "HELDOUT_RUN_INDICES",
    "NOMINAL_SEEDS",
    "TRAINING_ROWS",
    "TRAINING_RUN_INDICES",
    "collect",
    "load_contract",
    "load_training_candidate",
    "preflight",
    "publish_new",
    "write_failure_new",
]
