"""Collect exact teacher counterexamples on seed835 student failure prefix."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_hybrid_cutoff100_qualification as cutoff100,
    g1_true23_sonic_nominal_dagger_cutoff50_collection as base,
    g1_true23_sonic_student_teacher_recovery as recovery,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    compose_checkpoint21204_teacher_action,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_seed835_failure_prefix_dagger_v1.json"
)
CONTRACT_SHA256 = "ea9940447a62f510be25e358c0a70210830c674aa7bdbbce0602e2b85f4f0d7e"
CONTRACT_KIND = "g1_true23_sonic_seed835_failure_prefix_dagger_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_seed835_failure_prefix_dagger_manifest_v1"
RUNTIME_SEED = 835868017
ROWS = 89
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_seed835_failure_prefix_dagger.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_seed835_failure_prefix_dagger.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if recovery.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("failure-prefix DAgger contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    rows = _mapping(body.get("rows"), "rows")
    admission = _mapping(body.get("admission"), "admission")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or rows.get("total") != ROWS
        or rows.get("student_on_policy") != ROWS
        or rows.get("teacher_actuated") != 0
        or rows.get("q9_first") != 9
        or rows.get("q9_last") != 97
        or rows.get("terminal_causing_pre_action_row_included") is not True
        or admission.get("expected_terminal_transition") != 88
        or admission.get("expected_controller_counts") != {"student": 89, "teacher": 0}
        or boundaries.get("classification") != "failure_prefix_counterexample_dagger_bc_only"
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("failure-prefix DAgger contract semantic drift")
    inputs = _mapping(body.get("inputs"), "inputs")
    for name, raw_entry in inputs.items():
        entry = _mapping(raw_entry, name)
        candidate = (root / str(entry["path"])).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_file() or recovery.sha256_file(candidate) != entry["sha256"]:
            raise ValueError(f"failure-prefix DAgger input mismatch: {name}")
    parent = _mapping(
        json.loads((root / str(inputs["cutoff100_failure"]["path"])).read_text(encoding="utf-8")),
        "parent failure",
    )
    first_done = _mapping(parent.get("first_done"), "parent first_done")
    if (
        parent.get("verdict") != "mixed_controller_recovery_failed"
        or parent.get("attempted_transitions") != ROWS
        or parent.get("controller_counts") != {"student": 89, "teacher": 0}
        or first_done.get("transition") != 88
        or first_done.get("q9_before") != 97
        or first_done.get("termination_names") != ["ee_body_pos"]
    ):
        raise ValueError("failure-prefix parent evidence drift")
    return body


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    with cutoff100._qualification_scope(root, RUNTIME_SEED):  # noqa: SLF001
        before = recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS
        try:
            recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (*before, *SOURCE_PATHS)
            yield
        finally:
            recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = before


def _recovery_request(root: Path) -> recovery.RecoveryRequest:
    return cutoff100._request(  # noqa: SLF001
        root,
        Path("artifacts/g1_true23/.seed835-failure-prefix-inner-unused.json"),
    )


def preflight(request: base.CollectionRequest) -> Mapping[str, Any]:
    root = request.root
    load_contract(root)
    with _scope(root):
        inner = recovery._preflight_internal(_recovery_request(root))  # noqa: SLF001
    return {
        "ready": inner.get("ready") is True,
        "issues": list(inner.get("issues", ())),
        "contract_sha256": CONTRACT_SHA256,
        "runtime_seed": RUNTIME_SEED,
        "rows": ROWS,
        "simulator_constructed": False,
        "simulator_steps": 0,
        "teacher_labels_admitted": 0,
        "hardware_authorized": False,
    }


def _stack(rows: Mapping[int, Mapping[str, np.ndarray]], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    if tuple(sorted(rows)) != tuple(range(ROWS)):
        raise ValueError("failure-prefix row continuity mismatch")
    return {name: np.ascontiguousarray(np.stack([rows[index][name] for index in range(ROWS)])) for name in names}


def collect(request: base.CollectionRequest) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    root = request.root
    receipt = preflight(request)
    if receipt.get("ready") is not True:
        raise RuntimeError("failure-prefix collector preflight not ready")
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

    with _scope(root):
        recovery._ArrayDigestAccumulator.add = capturing_add  # noqa: SLF001
        try:
            report = recovery.run_recovery_diagnostic(_recovery_request(root))
        finally:
            recovery._ArrayDigestAccumulator.add = original_add  # noqa: SLF001
    first_done = _mapping(report.get("first_done"), "reproduced first_done")
    support = _mapping(report.get("support_summary"), "support summary")
    if (
        report.get("verdict") != "mixed_controller_recovery_failed"
        or report.get("attempted_transitions") != ROWS
        or report.get("simulator_step_calls_started") != ROWS
        or report.get("controller_counts") != {"student": ROWS, "teacher": 0}
        or report.get("partial_failure") is not None
        or first_done.get("transition") != 88
        or first_done.get("q9_before") != 97
        or first_done.get("termination_names") != ["ee_body_pos"]
        or support.get("teacher_parity_violation_count") != 0
        or support.get("action_semantics_mismatch_count") != 0
        or len(inference_rows) != ROWS
        or len(action_rows) != ROWS
    ):
        raise RuntimeError("failure-prefix rollout did not reproduce exact admitted boundary")
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
        ("executed_plain_raw_native23", "executed_safe_native23", "executed_final_target_hardware23"),
    )
    labels = np.stack(
        [
            compose_checkpoint21204_teacher_action(
                inference["selected_teacher_onnx_raw_hardware23"][index], repository_root=root
            ).teacher_action_native
            for index in range(ROWS)
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
        "q9": np.asarray([q9_values[index] for index in range(ROWS)], dtype=np.int64),
    }
    if any(controllers[index] != "student" for index in range(ROWS)):
        raise RuntimeError("failure-prefix contains nonstudent behavior")
    _validate_arrays(arrays)
    return arrays, {"preflight": receipt, "recovery_report": report}


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    widths = {
        "encoder267": 267,
        "token64": 64,
        "policy930": 930,
        "decoder994": 994,
        "student_raw_native23": 23,
        "selected_observation124": 124,
        "teacher_raw_hardware23": 23,
        "teacher_label_raw_native23": 23,
        "executed_raw_native23": 23,
        "executed_safe_native23": 23,
        "executed_final_target_hardware23": 23,
    }
    if set(arrays) != {*widths, "q9"}:
        raise ValueError("failure-prefix array key mismatch")
    for name, width in widths.items():
        value = arrays[name]
        if value.shape != (ROWS, width) or value.dtype != np.float32 or not bool(np.isfinite(value).all()):
            raise ValueError(f"failure-prefix array invalid: {name}")
    if arrays["q9"].dtype != np.int64 or not np.array_equal(arrays["q9"], np.arange(9, 98, dtype=np.int64)):
        raise ValueError("failure-prefix q9 mismatch")
    if not np.array_equal(arrays["decoder994"], np.concatenate((arrays["token64"], arrays["policy930"]), axis=1)):
        raise ValueError("failure-prefix decoder concat mismatch")
    if float(np.max(np.abs(arrays["teacher_label_raw_native23"]))) >= 10.0:
        raise ValueError("failure-prefix teacher label requires clipping")


def _write_temp(parent: Path, suffix: str, writer: Any) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=".seed835-failure-prefix-", suffix=suffix, dir=parent)
    os.close(descriptor)
    path = Path(raw)
    writer(path)
    return path


def publish(
    request: base.CollectionRequest,
    arrays: Mapping[str, np.ndarray],
    materials: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    _validate_arrays(arrays)
    npz = request.npz_path
    manifest = request.manifest_path
    if any(os.path.lexists(path) for path in (npz, manifest)):
        raise FileExistsError("failure-prefix output exists")

    def write_npz(path: Path) -> None:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())

    temp_npz = _write_temp(npz.parent, ".npz.tmp", write_npz)
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "artifact": {
            "npz_filename": npz.name,
            "npz_sha256": recovery.sha256_file(temp_npz),
            "array_schema": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": base._array_sha(value)}  # noqa: SLF001
                for name, value in sorted(arrays.items())
            },
        },
        "rows": {
            "total": ROWS,
            "student_on_policy": ROWS,
            "teacher_actuated": 0,
            "q9_first": 9,
            "q9_last": 97,
        },
        "materials": materials,
        "boundaries": dict(load_contract(request.root)["boundaries"]),
    }
    body["manifest_payload_sha256"] = base._sha256_bytes(base._canonical_bytes(body))  # noqa: SLF001

    def write_manifest(path: Path) -> None:
        with path.open("wb") as stream:
            stream.write(base._canonical_bytes(body))  # noqa: SLF001
            stream.flush()
            os.fsync(stream.fileno())

    temp_manifest = _write_temp(manifest.parent, ".json.tmp", write_manifest)
    npz_linked = False
    manifest_linked = False
    try:
        os.link(temp_npz, npz)
        npz_linked = True
        os.link(temp_manifest, manifest)
        manifest_linked = True
        return npz, manifest, body
    except Exception:
        if manifest_linked:
            manifest.unlink(missing_ok=True)
        if npz_linked:
            npz.unlink(missing_ok=True)
        raise
    finally:
        temp_npz.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)


__all__ = ["MANIFEST_KIND", "ROWS", "collect", "load_contract", "preflight", "publish"]
