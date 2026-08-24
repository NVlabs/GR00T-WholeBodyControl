"""Collect exact cutoff50 student-shadow/teacher-recovery BC rows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePath
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_multiseed_student_qualification as adapter,
    g1_true23_sonic_student_teacher_recovery as recovery,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    compose_checkpoint21204_teacher_action,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_cutoff50_collection_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_cutoff50_collection_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_cutoff50_collection_v1.json"
)
RECOVERY_CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_student_teacher_recovery_v4.json"
)
RECOVERY_CONTRACT_SHA256 = "2b70f24f07619ad586a93f0e9c465e5d9647143d198f124c2568e4dd7e357581"
CANDIDATE_DECODER_SHA256 = "91014f0cc37899ae795cc09e6a5a3c653ff6c587cf5d10cf22f41cbad280d544"
TOTAL_ROWS = 510
STUDENT_ROWS = 50
TEACHER_ROWS = 460


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise TypeError("cutoff50 collection contract must be object")
    runtime = _mapping(body.get("runtime"), "runtime")
    rows = _mapping(body.get("rows"), "rows")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != CONTRACT_KIND
        or runtime.get("seed") != 20260805
        or runtime.get("mode") != "cutoff50"
        or runtime.get("student_controlled_transitions") != STUDENT_ROWS
        or runtime.get("teacher_controlled_transitions") != TEACHER_ROWS
        or rows.get("published_total") != TOTAL_ROWS
        or rows.get("student_on_policy_shadow_label_rows") != STUDENT_ROWS
        or rows.get("teacher_actuated_recovery_rows") != TEACHER_ROWS
        or rows.get("whole_rollout_must_pass_before_any_rows_publish") is not True
        or boundaries.get("support_qualified") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("cutoff50 collection contract semantic drift")
    return body


def _regular_file(root: Path, entry: Mapping[str, Any], context: str) -> Path:
    relative = entry.get("path")
    expected = entry.get("sha256")
    if type(relative) is not str or PurePath(relative).is_absolute() or type(expected) is not str:
        raise ValueError(f"{context} entry invalid")
    path = (root / relative).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"{context} file invalid")
    if sha256_file(path) != expected:
        raise ValueError(f"{context} hash mismatch")
    return path


@dataclass(frozen=True)
class CollectionRequest:
    repository_root: Path
    output_prefix: Path

    @property
    def root(self) -> Path:
        root = self.repository_root.expanduser().resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("repository root invalid")
        return root

    @property
    def prefix(self) -> Path:
        evidence = (self.root / "artifacts" / "g1_true23").resolve(strict=True)
        raw = self.output_prefix.expanduser()
        if raw.suffix:
            raise ValueError("output prefix cannot have suffix")
        value = (raw if raw.is_absolute() else self.root / raw).resolve(strict=False)
        if not value.is_relative_to(evidence) or value.parent.is_symlink() or not value.parent.is_dir():
            raise ValueError("output prefix invalid")
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


@contextmanager
def _recovery_scope(root: Path) -> Iterator[None]:
    original = {
        "RECOVERY_CONTRACT_RELATIVE_PATH": recovery.RECOVERY_CONTRACT_RELATIVE_PATH,
        "RECOVERY_CONTRACT_SHA256": recovery.RECOVERY_CONTRACT_SHA256,
        "CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH": recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        "CURRENT_CANDIDATE_MANIFEST_SHA256": recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        "CURRENT_CANDIDATE_DECODER_SHA256": recovery.CURRENT_CANDIDATE_DECODER_SHA256,
        "RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS": recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS,
    }
    try:
        recovery.RECOVERY_CONTRACT_RELATIVE_PATH = RECOVERY_CONTRACT_RELATIVE_PATH
        recovery.RECOVERY_CONTRACT_SHA256 = RECOVERY_CONTRACT_SHA256
        recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH = adapter.CANDIDATE_MANIFEST_RELATIVE_PATH
        recovery.CURRENT_CANDIDATE_MANIFEST_SHA256 = adapter.CANDIDATE_MANIFEST_SHA256
        recovery.CURRENT_CANDIDATE_DECODER_SHA256 = CANDIDATE_DECODER_SHA256
        recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (
            *original["RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS"],
            RECOVERY_CONTRACT_RELATIVE_PATH,
            CONTRACT_RELATIVE_PATH,
            Path("gear_sonic/utils/g1_true23_sonic_nominal_dagger_cutoff50_collection.py"),
            Path("gear_sonic/scripts/collect_g1_true23_sonic_nominal_dagger_cutoff50.py"),
        )
        yield
    finally:
        for name, value in original.items():
            setattr(recovery, name, value)


def _recovery_request(root: Path) -> recovery.RecoveryRequest:
    return recovery.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=adapter.CANDIDATE_MANIFEST_SHA256,
        output=Path("artifacts/g1_true23/.cutoff50-collector-inner-unused.json"),
        mode="cutoff50",
    )


def preflight(request: CollectionRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    inputs = _mapping(contract["inputs"], "inputs")
    paths = {name: _regular_file(root, _mapping(entry, name), name) for name, entry in inputs.items()}
    failure = json.loads(paths["failed_student_report"].read_text(encoding="utf-8"))
    first_done = _mapping(failure.get("first_done"), "failed first_done")
    if (
        failure.get("verdict") != "student_qualification_failed"
        or failure.get("attempted_transitions") != 113
        or first_done.get("transition") != 112
        or first_done.get("termination_names") != ["ee_body_pos"]
    ):
        raise ValueError("failed student parent evidence drift")
    with adapter._adapter_scope(root, 20260805), _recovery_scope(root):  # noqa: SLF001
        inner = recovery._preflight_internal(_recovery_request(root))  # noqa: SLF001
    return {
        "ready": inner.get("ready") is True,
        "issues": list(inner.get("issues", ())),
        "contract_sha256": sha256_file(root / CONTRACT_RELATIVE_PATH),
        "recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
        "candidate_manifest_sha256": adapter.CANDIDATE_MANIFEST_SHA256,
        "candidate_decoder_sha256": CANDIDATE_DECODER_SHA256,
        "failed_student_report_sha256": inputs["failed_student_report"]["sha256"],
        "simulator_constructed": False,
        "simulator_steps": 0,
        "teacher_labels_admitted": 0,
        "hardware_authorized": False,
    }


def _stack(rows: Mapping[int, Mapping[str, np.ndarray]], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    if tuple(sorted(rows)) != tuple(range(TOTAL_ROWS)):
        raise ValueError("captured row index continuity mismatch")
    return {
        name: np.ascontiguousarray(np.stack([rows[index][name] for index in range(TOTAL_ROWS)])) for name in names
    }


def collect(request: CollectionRequest) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    root = request.root
    pre = preflight(request)
    if pre.get("ready") is not True:
        raise RuntimeError("cutoff50 collector preflight not ready")
    inference_rows: dict[int, dict[str, np.ndarray]] = {}
    action_rows: dict[int, dict[str, np.ndarray]] = {}
    controllers: dict[int, str] = {}
    q9_values: dict[int, int] = {}
    original_add = recovery._ArrayDigestAccumulator.add  # noqa: SLF001

    def capturing_add(
        self: Any,
        *,
        transition: int,
        q9: int,
        controller: str,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        original_add(self, transition=transition, q9=q9, controller=controller, arrays=arrays)
        copied = {name: np.ascontiguousarray(value).copy() for name, value in arrays.items()}
        if "student_decoder994" in copied:
            inference_rows[transition] = copied
            controllers[transition] = controller
            q9_values[transition] = q9
        elif "executed_plain_raw_native23" in copied:
            action_rows[transition] = copied

    with adapter._adapter_scope(root, 20260805), _recovery_scope(root):  # noqa: SLF001
        recovery._ArrayDigestAccumulator.add = capturing_add  # noqa: SLF001
        try:
            report = recovery.run_recovery_diagnostic(_recovery_request(root))
        finally:
            recovery._ArrayDigestAccumulator.add = original_add  # noqa: SLF001
    if (
        report.get("verdict") != "mixed_controller_recovery_passed"
        or report.get("recovered_to_original_q9_518_boundary") is not True
        or report.get("attempted_transitions") != TOTAL_ROWS
        or report.get("simulator_step_calls_started") != TOTAL_ROWS
        or report.get("controller_counts") != {"student": STUDENT_ROWS, "teacher": TEACHER_ROWS}
        or report.get("partial_failure") is not None
        or len(inference_rows) != TOTAL_ROWS
        or len(action_rows) != TOTAL_ROWS
    ):
        raise RuntimeError("cutoff50 recovery did not pass whole-rollout collection gate")
    inference = _stack(
        inference_rows,
        (
            "student_encoder267",
            "student_token64",
            "student_policy930",
            "student_decoder994",
            "student_raw_native23",
            "selected_teacher_observation124",
            "selected_teacher_onnx_raw_hardware23",
        ),
    )
    actions = _stack(
        action_rows,
        (
            "executed_plain_raw_native23",
            "executed_safe_native23",
            "executed_final_target_hardware23",
        ),
    )
    labels = np.stack(
        [
            compose_checkpoint21204_teacher_action(
                inference["selected_teacher_onnx_raw_hardware23"][index], repository_root=root
            ).teacher_action_native
            for index in range(TOTAL_ROWS)
        ]
    ).astype(np.float32)
    arrays = {
        "encoder267": inference["student_encoder267"],
        "token64": inference["student_token64"],
        "policy930": inference["student_policy930"],
        "decoder994": inference["student_decoder994"],
        "student_raw_native23": inference["student_raw_native23"],
        "selected_observation124": inference["selected_teacher_observation124"],
        "teacher_raw_hardware23": inference["selected_teacher_onnx_raw_hardware23"],
        "teacher_label_raw_native23": labels,
        "executed_raw_native23": actions["executed_plain_raw_native23"],
        "executed_safe_native23": actions["executed_safe_native23"],
        "executed_final_target_hardware23": actions["executed_final_target_hardware23"],
        "controller_code": np.asarray(
            [0 if controllers[index] == "student" else 1 for index in range(TOTAL_ROWS)],
            dtype=np.int8,
        ),
        "q9": np.asarray([q9_values[index] for index in range(TOTAL_ROWS)], dtype=np.int64),
    }
    _validate_arrays(arrays)
    return arrays, {"preflight": pre, "recovery_report": report}


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "encoder267": (TOTAL_ROWS, 267),
        "token64": (TOTAL_ROWS, 64),
        "policy930": (TOTAL_ROWS, 930),
        "decoder994": (TOTAL_ROWS, 994),
        "student_raw_native23": (TOTAL_ROWS, 23),
        "selected_observation124": (TOTAL_ROWS, 124),
        "teacher_raw_hardware23": (TOTAL_ROWS, 23),
        "teacher_label_raw_native23": (TOTAL_ROWS, 23),
        "executed_raw_native23": (TOTAL_ROWS, 23),
        "executed_safe_native23": (TOTAL_ROWS, 23),
        "executed_final_target_hardware23": (TOTAL_ROWS, 23),
        "controller_code": (TOTAL_ROWS,),
        "q9": (TOTAL_ROWS,),
    }
    if set(arrays) != set(expected):
        raise ValueError("cutoff50 array key mismatch")
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or not bool(np.isfinite(value).all()):
            raise ValueError(f"cutoff50 array invalid: {name}")
    if not np.array_equal(arrays["q9"], np.arange(9, 519, dtype=np.int64)):
        raise ValueError("cutoff50 q9 sequence mismatch")
    expected_controller = np.concatenate(
        (np.zeros(STUDENT_ROWS, dtype=np.int8), np.ones(TEACHER_ROWS, dtype=np.int8))
    )
    if not np.array_equal(arrays["controller_code"], expected_controller):
        raise ValueError("cutoff50 controller sequence mismatch")
    if not np.array_equal(arrays["decoder994"], np.concatenate((arrays["token64"], arrays["policy930"]), axis=1)):
        raise ValueError("cutoff50 decoder concat mismatch")
    if float(np.max(np.abs(arrays["teacher_label_raw_native23"]))) >= 10.0:
        raise ValueError("cutoff50 labels require forbidden clipping")


def _array_sha(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _write_npz(parent: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=".cutoff50-dagger-", suffix=".npz.tmp", dir=parent)
    os.close(descriptor)
    path = Path(raw)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _write_bytes(parent: Path, payload: bytes) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=".cutoff50-dagger-", suffix=".json.tmp", dir=parent)
    path = Path(raw)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def publish(
    request: CollectionRequest,
    arrays: Mapping[str, np.ndarray],
    materials: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    _validate_arrays(arrays)
    if any(os.path.lexists(path) for path in (request.npz_path, request.manifest_path, request.failure_path)):
        raise FileExistsError("cutoff50 collection output exists")
    temporary_npz = _write_npz(request.npz_path.parent, arrays)
    temporary_manifest: Path | None = None
    npz_published = False
    manifest_published = False
    try:
        body = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "contract": {
                "path": CONTRACT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(request.root / CONTRACT_RELATIVE_PATH),
            },
            "artifact": {
                "npz_filename": request.npz_path.name,
                "npz_sha256": sha256_file(temporary_npz),
                "npz_size_bytes": temporary_npz.stat().st_size,
                "array_schema": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_sha(value)}
                    for name, value in arrays.items()
                },
            },
            "rows": {
                "total": TOTAL_ROWS,
                "student_on_policy_shadow_label_rows": STUDENT_ROWS,
                "teacher_actuated_recovery_rows": TEACHER_ROWS,
            },
            "materials": materials,
            "boundaries": dict(load_contract(request.root)["boundaries"]),
        }
        body["manifest_payload_sha256"] = _sha256_bytes(_canonical_bytes(body))
        temporary_manifest = _write_bytes(request.manifest_path.parent, _canonical_bytes(body))
        os.link(temporary_npz, request.npz_path)
        npz_published = True
        os.link(temporary_manifest, request.manifest_path)
        manifest_published = True
        return request.npz_path, request.manifest_path, body
    except Exception:
        if manifest_published:
            request.manifest_path.unlink(missing_ok=True)
        if npz_published:
            request.npz_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_npz.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


def write_failure(request: CollectionRequest, error: BaseException) -> Path:
    if os.path.lexists(request.npz_path) or os.path.lexists(request.manifest_path):
        raise FileExistsError("cannot publish failure beside dataset")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_sonic_nominal_dagger_cutoff50_collection_failure_v1",
        "contract_sha256": sha256_file(request.root / CONTRACT_RELATIVE_PATH),
        "error_type": type(error).__name__,
        "error_detail_sha256": _sha256_bytes(str(error).encode()),
        "npz_published": False,
        "teacher_labels_admitted": 0,
        "hardware_authorized": False,
    }
    temporary = _write_bytes(request.failure_path.parent, _canonical_bytes(body))
    try:
        os.link(temporary, request.failure_path)
    finally:
        temporary.unlink(missing_ok=True)
    return request.failure_path


__all__ = ["CollectionRequest", "collect", "load_contract", "preflight", "publish", "write_failure"]
