"""Deterministic final-affine SONIC BC over nominal multi-seed teacher data."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePath
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from gear_sonic.utils import g1_true23_native124_21204_bc_last_affine_ridge as base_bc
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    load_bootstrap_training_candidate,
)
from gear_sonic.utils.g1_true23_selected_teacher_nominal_multiseed_collection import (
    CollectionRequest,
    load_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge_v1.json"
)
CONTRACT_SHA256 = "b1b1d1c85a7965f734bf54d91f9314770f8d90c1ba120aefe53c9abcb7df7a5e"
RUN_COUNT = 8
ROWS_PER_RUN = 510
TOTAL_ROWS = RUN_COUNT * ROWS_PER_RUN
RESET_PREFIX = 10
ACTION_DIM = 23
DECODER_DIM = 994
HIDDEN_DIM = 512
FINAL_WEIGHT_NAME = "layers.8.weight"
FINAL_BIAS_NAME = "layers.8.bias"
HIDDEN_OUTPUT_NAME = "/Mul_7_output_0"


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


def _regular_file(root: Path, entry: Mapping[str, Any], context: str) -> Path:
    relative = entry.get("path")
    expected = entry.get("sha256")
    if type(relative) is not str or PurePath(relative).is_absolute():
        raise ValueError(f"{context} path invalid")
    if type(expected) is not str or len(expected) != 64:
        raise ValueError(f"{context} hash invalid")
    path = (root / relative).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"{context} file invalid")
    if sha256_file(path) != expected:
        raise ValueError(f"{context} hash mismatch")
    return path


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("multi-seed BC contract hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise TypeError("multi-seed BC contract must be an object")
    _validate_contract(body)
    return body


def contract_sha256(root: Path) -> str:
    return sha256_file((root / CONTRACT_RELATIVE_PATH).resolve(strict=True))


def _validate_contract(contract: Mapping[str, Any]) -> None:
    rows = _mapping(contract.get("rows"), "rows")
    fit = _mapping(contract.get("fit"), "fit")
    model = _mapping(contract.get("model"), "model")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "deterministic_nominal_multiseed_teacher_bc_simulator_candidate_only"
        or rows.get("run_count") != RUN_COUNT
        or rows.get("rows_per_run") != ROWS_PER_RUN
        or rows.get("total") != TOTAL_ROWS
        or rows.get("heldout_collection_runs_used_for_fit") != 0
        or model.get("only_final_affine_may_change") is not True
        or model.get("v2_transform_external_application_count") != 1
        or fit.get("reset_prefix_row_weight") != 4.0
        or fit.get("real_h10_row_weight") != 1.0
        or fit.get("weights_normalized_to_mean_one") is not True
        or fit.get("bias_regularized") is not True
        or boundaries.get("heldout_collection_labels_used") is not False
        or boundaries.get("hardware_authorized") is not False
        or boundaries.get("robot_or_network_commands_permitted") is not False
    ):
        raise ValueError("multi-seed BC contract semantic drift")
    lambdas = fit.get("lambda_grid")
    if (
        not isinstance(lambdas, list)
        or len(lambdas) != 15
        or any(type(value) not in (int, float) or isinstance(value, bool) or value <= 0 for value in lambdas)
        or list(map(float, lambdas)) != sorted(map(float, lambdas))
    ):
        raise ValueError("multi-seed BC lambda grid drift")


@dataclass(frozen=True)
class FitRequest:
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
            raise ValueError("output prefix outside artifacts/g1_true23")
        return value

    @property
    def decoder_path(self) -> Path:
        return Path(f"{self.prefix}.decoder.onnx")

    @property
    def manifest_path(self) -> Path:
        return Path(f"{self.prefix}.manifest.json")


def row_weights(contract: Mapping[str, Any]) -> np.ndarray:
    fit = _mapping(contract["fit"], "fit")
    pattern = np.concatenate(
        (
            np.full(RESET_PREFIX, float(fit["reset_prefix_row_weight"]), dtype=np.float64),
            np.full(ROWS_PER_RUN - RESET_PREFIX, float(fit["real_h10_row_weight"]), dtype=np.float64),
        )
    )
    weights = np.tile(pattern, RUN_COUNT)
    weights *= TOTAL_ROWS / float(weights.sum())
    if weights.shape != (TOTAL_ROWS,) or not math.isclose(float(weights.mean()), 1.0, abs_tol=1e-14):
        raise RuntimeError("row weight normalization failed")
    return weights


def leave_one_run_out_folds() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    rows = np.arange(TOTAL_ROWS, dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for run in range(RUN_COUNT):
        start = run * ROWS_PER_RUN
        stop = start + ROWS_PER_RUN
        validation = rows[start:stop]
        training = np.concatenate((rows[:start], rows[stop:]))
        if training.size != TOTAL_ROWS - ROWS_PER_RUN or validation.size != ROWS_PER_RUN:
            raise RuntimeError("leave-one-run-out fold construction failed")
        folds.append((training, validation))
    return tuple(folds)


def _weighted_ridge(
    scores: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim != 2 or r.shape != (z.shape[0], ACTION_DIM) or w.shape != (z.shape[0],):
        raise ValueError("weighted ridge input shape mismatch")
    if not all(bool(np.isfinite(value).all()) for value in (z, r, w)) or np.any(w <= 0):
        raise ValueError("weighted ridge input invalid")
    design = np.concatenate((z, np.ones((z.shape[0], 1), dtype=np.float64)), axis=1)
    weight_sum = float(w.sum())
    gram = (design.T @ (design * w[:, None])) / weight_sum
    system = gram + ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    rhs = (design.T @ (r * w[:, None])) / weight_sum
    solution = np.linalg.solve(system, rhs)
    return solution[:-1].T, solution[-1], float(np.linalg.cond(system))


def _weighted_sse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sum(error * error * np.asarray(weights)[:, None], dtype=np.float64))


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    row = np.sqrt(np.mean(error * error, axis=1, dtype=np.float64))
    return {
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "max_abs": float(np.max(np.abs(error))),
        "p95_row_rmse": float(np.quantile(row, 0.95)),
        "per_dof_rmse": np.sqrt(np.mean(error * error, axis=0, dtype=np.float64)).tolist(),
    }


def _extract_hidden_variable(
    model: Any, decoder994: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    import copy

    from onnx import TensorProto, helper
    import onnxruntime as ort

    values = np.asarray(decoder994, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != DECODER_DIM or not bool(np.isfinite(values).all()):
        raise ValueError("decoder input matrix invalid")
    probe = copy.deepcopy(model)
    if HIDDEN_OUTPUT_NAME in {value.name for value in probe.graph.output}:
        raise ValueError("decoder unexpectedly exports hidden output")
    probe.graph.output.extend(
        [helper.make_tensor_value_info(HIDDEN_OUTPUT_NAME, TensorProto.FLOAT, [1, HIDDEN_DIM])]
    )
    session = base_bc._cpu_ort_session(probe.SerializeToString(deterministic=True))  # noqa: SLF001
    predictions: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    for row in values:
        action, feature = session.run(["action", HIDDEN_OUTPUT_NAME], {"obs_dict": row.reshape(1, DECODER_DIM)})
        predictions.append(action[0])
        hidden.append(feature[0])
    output = np.asarray(predictions, dtype=np.float32)
    features = np.asarray(hidden, dtype=np.float32)
    if output.shape != (values.shape[0], ACTION_DIM) or features.shape != (values.shape[0], HIDDEN_DIM):
        raise RuntimeError("decoder probe output shape mismatch")
    return (
        features.astype(np.float64),
        output.astype(np.float64),
        {
            "provider": session.get_providers()[0],
            "onnxruntime_version": ort.__version__,
            "row_count": values.shape[0],
            "host_reads_inside_simulator_step": False,
        },
    )


def _initializer_arrays(model: Any) -> tuple[np.ndarray, np.ndarray]:
    from onnx import numpy_helper

    values = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    weight = np.asarray(values[FINAL_WEIGHT_NAME], dtype=np.float32)
    bias = np.asarray(values[FINAL_BIAS_NAME], dtype=np.float32)
    if weight.shape != (ACTION_DIM, HIDDEN_DIM) or bias.shape != (ACTION_DIM,):
        raise ValueError("decoder final affine shape drift")
    return weight, bias


def _fit(
    hidden: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    fit = _mapping(contract["fit"], "fit")
    gates = _mapping(contract["gates"], "gates")
    h = np.asarray(hidden, dtype=np.float64)
    y0 = np.asarray(base_prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if h.shape != (TOTAL_ROWS, HIDDEN_DIM) or y0.shape != (TOTAL_ROWS, ACTION_DIM) or y.shape != y0.shape:
        raise ValueError("multi-seed fit input shape mismatch")
    weights = row_weights(contract)
    residual = y - y0
    lambdas = [float(value) for value in fit["lambda_grid"]]
    predictions = {value: np.empty_like(y) for value in lambdas}
    fold_records: list[dict[str, Any]] = []
    for run, (training, validation) in enumerate(leave_one_run_out_folds()):
        projection = base_bc.fit_projection(
            h[training],
            variance_fraction=float(fit["pca_variance_fraction"]),
            minimum_rank=int(fit["pca_min_rank"]),
            maximum_rank=int(fit["pca_max_rank"]),
        )
        train_scores = projection.transform(h[training])
        validation_scores = projection.transform(h[validation])
        conditions: dict[str, float] = {}
        for ridge_lambda in lambdas:
            coefficient, intercept, condition = _weighted_ridge(
                train_scores, residual[training], weights[training], ridge_lambda
            )
            predictions[ridge_lambda][validation] = y0[validation] + (
                validation_scores @ coefficient.T + intercept
            )
            conditions[format(ridge_lambda, ".17g")] = condition
        fold_records.append(
            {
                "heldout_training_run_index": run,
                "pca_rank": projection.rank,
                "pca_explained_fraction": projection.explained_fraction,
                "condition_by_lambda": conditions,
            }
        )
    pooled = {value: _weighted_sse(predictions[value], y, weights) for value in lambdas}
    minimum = min(pooled.values())
    selected = max(value for value in lambdas if pooled[value] <= minimum * 1.01)
    oof = predictions[selected]
    oof_rmse = math.sqrt(pooled[selected] / (float(weights.sum()) * ACTION_DIM))
    base_oof_rmse = math.sqrt(_weighted_sse(y0, y, weights) / (float(weights.sum()) * ACTION_DIM))
    folds_beating = 0
    for record, (_, validation) in zip(fold_records, leave_one_run_out_folds(), strict=True):
        adapted = _metrics(oof[validation], y[validation])
        frozen = _metrics(y0[validation], y[validation])
        beats = adapted["rmse"] < frozen["rmse"]
        folds_beating += int(beats)
        record.update(
            {
                "selected_condition": record["condition_by_lambda"][format(selected, ".17g")],
                "adapted": adapted,
                "base": frozen,
                "beats_base": beats,
            }
        )

    projection = base_bc.fit_projection(
        h,
        variance_fraction=float(fit["pca_variance_fraction"]),
        minimum_rank=int(fit["pca_min_rank"]),
        maximum_rank=int(fit["pca_max_rank"]),
    )
    coefficient, intercept, condition = _weighted_ridge(projection.transform(h), residual, weights, selected)
    delta_weight = coefficient @ projection.basis
    delta_bias = intercept
    prediction = y0 + h @ delta_weight.T + delta_bias
    all_metrics = _metrics(prediction, y)
    prefix_indices = np.concatenate(
        [np.arange(run * ROWS_PER_RUN, run * ROWS_PER_RUN + RESET_PREFIX) for run in range(RUN_COUNT)]
    )
    h10_indices = np.setdiff1d(np.arange(TOTAL_ROWS), prefix_indices, assume_unique=True)
    prefix_metrics = _metrics(prediction[prefix_indices], y[prefix_indices])
    h10_metrics = _metrics(prediction[h10_indices], y[h10_indices])
    run_metrics = [
        _metrics(
            prediction[run * ROWS_PER_RUN : (run + 1) * ROWS_PER_RUN],
            y[run * ROWS_PER_RUN : (run + 1) * ROWS_PER_RUN],
        )
        for run in range(RUN_COUNT)
    ]
    issues: list[str] = []
    if oof_rmse > float(gates["maximum_oof_rmse_ratio_to_base"]) * base_oof_rmse:
        issues.append("leave_one_run_out_rmse_gate_failed")
    if folds_beating < int(gates["minimum_leave_one_run_out_folds_beating_base"]):
        issues.append("leave_one_run_out_fold_count_gate_failed")
    if condition > float(gates["maximum_ridge_condition_number"]):
        issues.append("ridge_condition_gate_failed")
    checks = (
        (all_metrics["rmse"], "maximum_training_all_rmse", "training_all_rmse_gate_failed"),
        (
            max(value["rmse"] for value in run_metrics),
            "maximum_training_each_run_rmse",
            "training_run_rmse_gate_failed",
        ),
        (prefix_metrics["rmse"], "maximum_training_reset_prefix_rmse", "training_prefix_rmse_gate_failed"),
        (h10_metrics["rmse"], "maximum_training_real_h10_rmse", "training_h10_rmse_gate_failed"),
        (all_metrics["p95_row_rmse"], "maximum_training_p95_row_rmse", "training_p95_gate_failed"),
        (all_metrics["max_abs"], "maximum_training_absolute_error", "training_max_abs_gate_failed"),
    )
    for observed, threshold, issue in checks:
        if observed > float(gates[threshold]):
            issues.append(issue)
    if float(np.max(np.abs(prediction))) >= float(gates["plain_raw_absolute_strict_max"]):
        issues.append("plain_raw_absolute_gate_failed")
    report = {
        "selected_lambda": selected,
        "pca_rank": projection.rank,
        "pca_explained_fraction": projection.explained_fraction,
        "final_condition": condition,
        "delta_weight_frobenius": float(np.linalg.norm(delta_weight)),
        "delta_bias_l2": float(np.linalg.norm(delta_bias)),
        "leave_one_run_out": {
            "weighted_rmse": oof_rmse,
            "base_weighted_rmse": base_oof_rmse,
            "ratio_to_base": oof_rmse / base_oof_rmse,
            "folds_beating_base": folds_beating,
            "pooled_weighted_sse_by_lambda": [{"lambda": value, "sse": pooled[value]} for value in lambdas],
            "folds": fold_records,
        },
        "training_resubstitution": {
            "all": all_metrics,
            "reset_prefix_80": prefix_metrics,
            "real_h10_4000": h10_metrics,
            "per_run": run_metrics,
        },
        "gate_issues": sorted(set(issues)),
    }
    return delta_weight, delta_bias, report


def _canonical_manifest(path: Path, expected_sha: str, context: str) -> Mapping[str, Any]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{context} hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise TypeError(f"{context} must be object")
    return body


def validate_candidate_manifest_fields(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    claimed_payload = payload.pop("manifest_payload_sha256", None)
    if claimed_payload != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("multi-seed BC manifest payload hash mismatch")
    contract_entry = _mapping(manifest.get("contract"), "candidate contract")
    artifact = _mapping(manifest.get("artifact"), "candidate artifact")
    lineage = _mapping(manifest.get("lineage"), "candidate lineage")
    inputs = _mapping(contract.get("inputs"), "contract inputs")
    required_lineage = {
        "dataset_npz_sha256": inputs["dataset_npz"]["sha256"],
        "dataset_manifest_sha256": inputs["dataset_manifest"]["sha256"],
        "base_decoder_sha256": inputs["base_decoder"]["sha256"],
        "base_manifest_sha256": inputs["base_manifest"]["sha256"],
        "bootstrap_npz_sha256": inputs["bootstrap_npz"]["sha256"],
        "bootstrap_manifest_sha256": inputs["bootstrap_manifest"]["sha256"],
        "source_decoder_sha256": inputs["source_decoder"]["sha256"],
        "teacher_controlled_training_rows": TOTAL_ROWS,
        "heldout_collection_labels_used": False,
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("classification") != "offline_bc_eligible_for_closed_loop_simulator_experiment"
        or manifest.get("eligible_for_closed_loop_simulator_experiment") is not True
        or manifest.get("gate_issues") != []
        or contract_entry != {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        or any(lineage.get(name) != value for name, value in required_lineage.items())
        or artifact.get("publishable_decoder") is not True
        or artifact.get("publication_protocol") != "decoder_hardlink_then_manifest_hardlink_commit"
        or artifact.get("decoder_sha256") != manifest.get("export", {}).get("candidate_decoder_sha256")
        or manifest.get("boundaries") != contract.get("boundaries")
    ):
        raise ValueError("multi-seed BC candidate manifest semantic drift")


def preflight(request: FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    inputs = _mapping(contract["inputs"], "inputs")
    paths = {name: _regular_file(root, _mapping(entry, name), name) for name, entry in inputs.items()}
    dataset_prefix = Path(str(paths["dataset_npz"])[: -len(".npz")])
    arrays, dataset_manifest = load_training_candidate(CollectionRequest(root, dataset_prefix))
    if arrays["decoder994"].shape != (TOTAL_ROWS, DECODER_DIM):
        raise ValueError("dataset decoder input shape mismatch")
    base_manifest = _canonical_manifest(paths["base_manifest"], inputs["base_manifest"]["sha256"], "base manifest")
    artifact = _mapping(base_manifest.get("artifact"), "base artifact")
    if (
        artifact.get("decoder_sha256") != inputs["base_decoder"]["sha256"]
        or artifact.get("publishable_decoder") is not True
        or base_manifest.get("eligible_for_closed_loop_simulator_experiment") is not True
    ):
        raise ValueError("base decoder manifest is not admitted v2 candidate")
    bootstrap_arrays, bootstrap_manifest = load_bootstrap_training_candidate(
        paths["bootstrap_npz"], paths["bootstrap_manifest"], repository_root=root
    )
    if dataset_manifest.get("rows", {}).get("heldout_teacher_label_rows_published") != 0:
        raise ValueError("heldout labels leaked into dataset")
    return {
        "ready": True,
        "contract_sha256": contract_sha256(root),
        "dataset_npz_sha256": inputs["dataset_npz"]["sha256"],
        "dataset_manifest_sha256": inputs["dataset_manifest"]["sha256"],
        "base_decoder_sha256": inputs["base_decoder"]["sha256"],
        "bootstrap_npz_sha256": inputs["bootstrap_npz"]["sha256"],
        "dataset_rows": int(arrays["decoder994"].shape[0]),
        "bootstrap_rows": int(bootstrap_arrays["decoder994"].shape[0]),
        "heldout_labels_used": False,
        "simulator_constructed": False,
        "training_updates": 0,
        "hardware_authorized": False,
        "bootstrap_manifest_kind": bootstrap_manifest.get("kind"),
    }


@contextmanager
def _numeric_runtime() -> Iterator[None]:
    import scipy

    controls: list[tuple[Any, int]] = []
    specifications = (
        (
            np,
            (
                ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                ("scipy_openblas_set_num_threads_64_", "scipy_openblas_get_num_threads_64_"),
                ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
            ),
        ),
        (
            scipy,
            (
                ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
                ("scipy_openblas_set_num_threads_", "scipy_openblas_get_num_threads_"),
            ),
        ),
    )
    try:
        for package, symbol_pairs in specifications:
            package_root = Path(package.__file__).resolve().parent
            candidates = sorted(
                {
                    *package_root.glob(".libs/*openblas*"),
                    *package_root.parent.glob(f"{package.__name__}.libs/*openblas*"),
                }
            )
            libraries = [path for path in candidates if path.is_file()]
            if len(libraries) != 1:
                raise RuntimeError(f"expected one {package.__name__} OpenBLAS library")
            library = ctypes.CDLL(str(libraries[0]))
            resolved: tuple[Any, Any] | None = None
            for set_name, get_name in symbol_pairs:
                try:
                    resolved = getattr(library, set_name), getattr(library, get_name)
                except AttributeError:
                    continue
                break
            if resolved is None:
                raise RuntimeError(f"current {package.__name__} OpenBLAS controls unavailable")
            setter, getter = resolved
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            getter.argtypes = []
            getter.restype = ctypes.c_int
            previous = int(getter())
            setter(1)
            if int(getter()) != 1:
                raise RuntimeError(f"{package.__name__} OpenBLAS refused one thread")
            controls.append((setter, previous))
        yield
    finally:
        for setter, previous in reversed(controls):
            setter(previous)


@dataclass(frozen=True)
class FitOutcome:
    passed: bool
    decoder_bytes: bytes | None
    report: Mapping[str, Any]


def run_fit(request: FitRequest) -> FitOutcome:
    import onnx

    root = request.root
    contract = load_contract(root)
    inputs = _mapping(contract["inputs"], "inputs")
    pre = preflight(request)
    paths = {name: _regular_file(root, _mapping(entry, name), name) for name, entry in inputs.items()}
    dataset_prefix = Path(str(paths["dataset_npz"])[: -len(".npz")])
    arrays, _ = load_training_candidate(CollectionRequest(root, dataset_prefix))
    bootstrap, _ = load_bootstrap_training_candidate(
        paths["bootstrap_npz"], paths["bootstrap_manifest"], repository_root=root
    )
    base_model = onnx.load(paths["base_decoder"], load_external_data=False)
    source_model = onnx.load(paths["source_decoder"], load_external_data=False)
    onnx.checker.check_model(base_model, full_check=True)
    base_bc.assert_only_final_affine_changed(source_model, base_model)
    with _numeric_runtime():
        hidden, base_prediction, runtime = _extract_hidden_variable(base_model, arrays["decoder994"])
        delta_weight, delta_bias, fit_report = _fit(
            hidden, base_prediction, arrays["teacher_label_raw_native23"], contract
        )
        base_weight, base_bias = _initializer_arrays(base_model)
        candidate_model, candidate_bytes = base_bc.export_final_affine_model(
            base_model,
            base_weight.astype(np.float64) + delta_weight,
            base_bias.astype(np.float64) + delta_bias,
        )
        session = base_bc._cpu_ort_session(candidate_bytes)  # noqa: SLF001
        exported = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in arrays["decoder994"]
            ],
            axis=0,
        )
        formula = fit_report["training_resubstitution"]["all"]
        export_metrics = _metrics(exported, arrays["teacher_label_raw_native23"])
        parity = float(np.max(np.abs(exported - (base_prediction + hidden @ delta_weight.T + delta_bias))))

        bootstrap_hidden, bootstrap_base, bootstrap_runtime = _extract_hidden_variable(
            base_model, bootstrap["decoder994"]
        )
        bootstrap_candidate = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in bootstrap["decoder994"]
            ],
            axis=0,
        )
        bootstrap_base_metrics = _metrics(bootstrap_base, bootstrap["teacher_label_raw_native23"])
        bootstrap_candidate_metrics = _metrics(bootstrap_candidate, bootstrap["teacher_label_raw_native23"])
        base_prefix = _metrics(
            bootstrap_base[:RESET_PREFIX], bootstrap["teacher_label_raw_native23"][:RESET_PREFIX]
        )
        candidate_prefix = _metrics(
            bootstrap_candidate[:RESET_PREFIX], bootstrap["teacher_label_raw_native23"][:RESET_PREFIX]
        )
    gates = _mapping(contract["gates"], "gates")
    issues = list(fit_report["gate_issues"])
    if parity > float(gates["maximum_export_reference_absolute_error"]):
        issues.append("export_reference_parity_gate_failed")
    if export_metrics["rmse"] > float(formula["rmse"]) + 2e-6:
        issues.append("export_training_rmse_drift")
    if bootstrap_candidate_metrics["rmse"] > bootstrap_base_metrics["rmse"] + float(
        gates["maximum_bootstrap_all_rmse_regression"]
    ):
        issues.append("bootstrap_all_preservation_gate_failed")
    if candidate_prefix["rmse"] > base_prefix["rmse"] + float(
        gates["maximum_bootstrap_reset_prefix_rmse_regression"]
    ):
        issues.append("bootstrap_prefix_preservation_gate_failed")
    if bootstrap_candidate_metrics["max_abs"] > float(gates["maximum_bootstrap_absolute_error"]):
        issues.append("bootstrap_max_abs_gate_failed")
    if float(np.max(np.abs(bootstrap_candidate))) >= float(gates["plain_raw_absolute_strict_max"]):
        issues.append("bootstrap_plain_raw_gate_failed")
    candidate_sha = _sha256_bytes(candidate_bytes)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "classification": (
            "offline_bc_eligible_for_closed_loop_simulator_experiment" if not issues else "offline_bc_rejected"
        ),
        "eligible_for_closed_loop_simulator_experiment": not issues,
        "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": contract_sha256(root)},
        "lineage": {
            "dataset_npz_sha256": inputs["dataset_npz"]["sha256"],
            "dataset_manifest_sha256": inputs["dataset_manifest"]["sha256"],
            "base_decoder_sha256": inputs["base_decoder"]["sha256"],
            "base_manifest_sha256": inputs["base_manifest"]["sha256"],
            "bootstrap_npz_sha256": inputs["bootstrap_npz"]["sha256"],
            "bootstrap_manifest_sha256": inputs["bootstrap_manifest"]["sha256"],
            "source_decoder_sha256": inputs["source_decoder"]["sha256"],
            "teacher_controlled_training_rows": TOTAL_ROWS,
            "heldout_collection_labels_used": False,
        },
        "fit": fit_report,
        "export": {
            "candidate_decoder_sha256": candidate_sha,
            "candidate_decoder_size_bytes": len(candidate_bytes),
            "training_metrics": export_metrics,
            "numpy_ort_max_abs": parity,
            "only_final_affine_changed_from_base": True,
        },
        "bootstrap_preservation": {
            "base_all": bootstrap_base_metrics,
            "candidate_all": bootstrap_candidate_metrics,
            "base_reset_prefix": base_prefix,
            "candidate_reset_prefix": candidate_prefix,
            "runtime": bootstrap_runtime,
        },
        "runtime": runtime,
        "preflight": pre,
        "gate_issues": sorted(set(issues)),
        "boundaries": dict(contract["boundaries"]),
    }
    return FitOutcome(passed=not issues, decoder_bytes=candidate_bytes if not issues else None, report=report)


def _write_temp(parent: Path, payload: bytes, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=".nominal-multiseed-bc-", suffix=suffix, dir=parent, delete=False)
    path = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    return path


def _publish_new(temp: Path, final: Path) -> None:
    os.link(temp, final)


def publish(request: FitRequest, outcome: FitOutcome) -> tuple[Path | None, Path, Mapping[str, Any]]:
    decoder = request.decoder_path
    manifest = request.manifest_path
    if any(os.path.lexists(path) for path in (decoder, manifest)):
        raise FileExistsError("multi-seed BC output already exists")
    if outcome.passed != (outcome.decoder_bytes is not None):
        raise ValueError("fit outcome pass/decoder mismatch")
    body = dict(outcome.report)
    body["artifact"] = {
        "decoder_filename": decoder.name if outcome.passed else None,
        "decoder_sha256": _sha256_bytes(outcome.decoder_bytes) if outcome.decoder_bytes else None,
        "decoder_size_bytes": len(outcome.decoder_bytes) if outcome.decoder_bytes else None,
        "manifest_filename": manifest.name,
        "publishable_decoder": outcome.passed,
        "publication_protocol": "decoder_hardlink_then_manifest_hardlink_commit",
        "overwrite_permitted": False,
    }
    body["manifest_payload_sha256"] = _sha256_bytes(_canonical_bytes(body))
    encoded = _canonical_bytes(body)
    temporary_decoder: Path | None = None
    temporary_manifest: Path | None = None
    decoder_published = False
    manifest_published = False
    try:
        if outcome.decoder_bytes is not None:
            temporary_decoder = _write_temp(decoder.parent, outcome.decoder_bytes, ".onnx.tmp")
        temporary_manifest = _write_temp(manifest.parent, encoded, ".json.tmp")
        if temporary_decoder is not None:
            _publish_new(temporary_decoder, decoder)
            decoder_published = True
        _publish_new(temporary_manifest, manifest)
        manifest_published = True
        return decoder if outcome.passed else None, manifest, body
    except Exception:
        if manifest_published:
            manifest.unlink(missing_ok=True)
        if decoder_published:
            decoder.unlink(missing_ok=True)
        raise
    finally:
        if temporary_decoder is not None:
            temporary_decoder.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


__all__ = [
    "FitOutcome",
    "FitRequest",
    "contract_sha256",
    "leave_one_run_out_folds",
    "load_contract",
    "preflight",
    "publish",
    "row_weights",
    "run_fit",
]
