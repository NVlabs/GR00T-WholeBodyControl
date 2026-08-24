"""Second targeted final-affine BC fit on seed835 student counterexample states."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_bc_last_affine_ridge as prior_fit,
    g1_true23_sonic_nominal_dagger_cutoff50_collection as cutoff50,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
    g1_true23_sonic_seed835_counterexample_bc_last_affine as round1_fit,
    g1_true23_sonic_seed835_failure_prefix_dagger as round1_counterexample,
    g1_true23_sonic_seed835_failure_prefix_dagger_round2 as round2_counterexample,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    load_bootstrap_training_candidate,
)
from gear_sonic.utils.g1_true23_selected_teacher_nominal_multiseed_collection import (
    CollectionRequest,
    load_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_seed835_counterexample_round2_bc_last_affine_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_seed835_counterexample_round2_bc_last_affine_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_seed835_counterexample_round2_bc_last_affine_v1.json"
)
CONTRACT_SHA256 = "9656e6b4c55c247b2eaf3f6a85efab5a9bbad9f06136fe16f935400f47cb31d3"
NOMINAL_ROWS = 4080
SEGMENT_ROWS = 510
ROUND1_ROWS = 89
ROUND2_ROWS = 168
TOTAL_ROWS = 5357
DECODER_DIM = 994


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if nominal_fit.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("round2 counterexample BC contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    rows = _mapping(body.get("rows"), "rows")
    fit = _mapping(body.get("fit"), "fit")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != CONTRACT_KIND
        or rows
        != {
            "nominal_train8": NOMINAL_ROWS,
            "prior_intervention": SEGMENT_ROWS,
            "seed835_teacher_reset": SEGMENT_ROWS,
            "round1_student_counterexample": ROUND1_ROWS,
            "round2_student_counterexample": ROUND2_ROWS,
            "total": TOTAL_ROWS,
        }
        or fit.get("only_final_affine_may_change") is not True
        or fit.get("round2_first118_weight") != 16.0
        or fit.get("round2_last50_weight") != 64.0
        or boundaries.get("decisive_supervised_retry") is not True
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("round2 counterexample BC contract semantic drift")
    return body


def _paths(root: Path, contract: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: nominal_fit._regular_file(root, _mapping(entry, name), name)  # noqa: SLF001
        for name, entry in _mapping(contract["inputs"], "inputs").items()
    }


def validate_candidate_manifest_fields(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    claimed = payload.pop("manifest_payload_sha256", None)
    if claimed != nominal_fit._sha256_bytes(nominal_fit._canonical_bytes(payload)):  # noqa: SLF001
        raise ValueError("round2 counterexample candidate payload hash mismatch")
    inputs = _mapping(contract["inputs"], "inputs")
    artifact = _mapping(manifest.get("artifact"), "artifact")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("eligible_for_closed_loop_simulator_experiment") is not True
        or manifest.get("gate_issues") != []
        or manifest.get("contract") != {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        or manifest.get("lineage") != {name: entry["sha256"] for name, entry in inputs.items()}
        or artifact.get("publishable_decoder") is not True
        or artifact.get("decoder_sha256") != manifest.get("export", {}).get("candidate_decoder_sha256")
        or manifest.get("boundaries") != contract.get("boundaries")
    ):
        raise ValueError("round2 counterexample candidate manifest semantic drift")


def _load_counterexample(
    npz: Path,
    manifest: Path,
    *,
    kind: str,
    rows: Mapping[str, int],
    validator: Any,
) -> dict[str, np.ndarray]:
    body = _mapping(json.loads(manifest.read_text(encoding="utf-8")), "counterexample manifest")
    if (
        body.get("kind") != kind
        or body.get("artifact", {}).get("npz_sha256") != nominal_fit.sha256_file(npz)
        or body.get("rows") != dict(rows)
    ):
        raise ValueError("counterexample manifest semantic drift")
    payload = dict(body)
    claimed = payload.pop("manifest_payload_sha256", None)
    if claimed != nominal_fit._sha256_bytes(nominal_fit._canonical_bytes(payload)):  # noqa: SLF001
        raise ValueError("counterexample manifest payload mismatch")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]).copy() for name in archive.files}
    validator(arrays)
    for name, value in arrays.items():
        expected = body["artifact"]["array_schema"][name]
        if expected != {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": cutoff50._array_sha(value),  # noqa: SLF001
        }:
            raise ValueError(f"counterexample array hash drift: {name}")
    return arrays


def _load_inputs(root: Path, paths: Mapping[str, Path]) -> tuple[dict[str, np.ndarray], ...]:
    nominal, _ = load_training_candidate(CollectionRequest(root, Path(str(paths["nominal_npz"])[: -len(".npz")])))
    intervention, _ = prior_fit._load_intervention(  # noqa: SLF001
        paths["intervention_npz"], paths["intervention_manifest"]
    )
    seed835, _ = load_bootstrap_training_candidate(
        paths["seed835_teacher_npz"], paths["seed835_teacher_manifest"], repository_root=root
    )
    round1 = _load_counterexample(
        paths["round1_counterexample_npz"],
        paths["round1_counterexample_manifest"],
        kind=round1_counterexample.MANIFEST_KIND,
        rows={"total": 89, "student_on_policy": 89, "teacher_actuated": 0, "q9_first": 9, "q9_last": 97},
        validator=round1_counterexample._validate_arrays,  # noqa: SLF001
    )
    round2 = _load_counterexample(
        paths["round2_counterexample_npz"],
        paths["round2_counterexample_manifest"],
        kind=round2_counterexample.MANIFEST_KIND,
        rows={
            "total": 168,
            "student_on_policy": 168,
            "teacher_actuated": 0,
            "q9_first": 9,
            "q9_last": 176,
        },
        validator=round2_counterexample._validate_arrays,  # noqa: SLF001
    )
    return nominal, intervention, seed835, round1, round2


def _weights(contract: Mapping[str, Any]) -> np.ndarray:
    fit = _mapping(contract["fit"], "fit")
    nominal_pattern = np.concatenate(
        (
            np.full(10, float(fit["nominal_reset_prefix_weight"]), dtype=np.float64),
            np.full(500, float(fit["nominal_real_h10_weight"]), dtype=np.float64),
        )
    )
    value = np.concatenate(
        (
            np.tile(nominal_pattern, 8),
            np.full(50, float(fit["prior_intervention_shadow_weight"]), dtype=np.float64),
            np.full(460, float(fit["prior_intervention_recovery_weight"]), dtype=np.float64),
            np.full(100, float(fit["seed835_teacher_first100_weight"]), dtype=np.float64),
            np.full(410, float(fit["seed835_teacher_tail410_weight"]), dtype=np.float64),
            np.full(50, float(fit["round1_first50_weight"]), dtype=np.float64),
            np.full(39, float(fit["round1_last39_weight"]), dtype=np.float64),
            np.full(118, float(fit["round2_first118_weight"]), dtype=np.float64),
            np.full(50, float(fit["round2_last50_weight"]), dtype=np.float64),
        )
    )
    value *= TOTAL_ROWS / float(value.sum())
    if value.shape != (TOTAL_ROWS,) or not math.isclose(float(value.mean()), 1.0, abs_tol=1e-14):
        raise RuntimeError("round2 counterexample fit weights invalid")
    return value


def preflight(request: nominal_fit.FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    datasets = _load_inputs(root, paths)
    base_manifest = _mapping(json.loads(paths["base_manifest"].read_text(encoding="utf-8")), "base manifest")
    round1_fit.validate_candidate_manifest_fields(base_manifest, round1_fit.load_contract(root))
    expected_rows = (NOMINAL_ROWS, SEGMENT_ROWS, SEGMENT_ROWS, ROUND1_ROWS, ROUND2_ROWS)
    if any(
        data["decoder994"].shape != (rows, DECODER_DIM) for data, rows in zip(datasets, expected_rows, strict=True)
    ):
        raise ValueError("round2 counterexample fit input shape drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "total_rows": TOTAL_ROWS,
        "round1_counterexample_rows": ROUND1_ROWS,
        "round2_counterexample_rows": ROUND2_ROWS,
        "base_decoder_sha256": contract["inputs"]["base_decoder"]["sha256"],
        "simulator_constructed": False,
        "training_updates": 0,
        "hardware_authorized": False,
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, Any]:
    return nominal_fit._metrics(prediction, target)  # noqa: SLF001


def _ort_prediction(session: Any, values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
            for row in values
        ],
        axis=0,
    )


def run_fit(request: nominal_fit.FitRequest) -> nominal_fit.FitOutcome:
    import onnx

    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    pre = preflight(request)
    datasets = _load_inputs(root, paths)
    decoder994 = np.concatenate([value["decoder994"] for value in datasets], axis=0)
    labels = np.concatenate([value["teacher_label_raw_native23"] for value in datasets], axis=0)
    base_model = onnx.load(paths["base_decoder"], load_external_data=False)
    onnx.checker.check_model(base_model, full_check=True)
    with nominal_fit._numeric_runtime():  # noqa: SLF001
        hidden, base_prediction, runtime = nominal_fit._extract_hidden_variable(  # noqa: SLF001
            base_model, decoder994
        )
        fit_cfg = _mapping(contract["fit"], "fit")
        projection = nominal_fit.base_bc.fit_projection(
            hidden,
            variance_fraction=float(fit_cfg["pca_variance_fraction"]),
            minimum_rank=int(fit_cfg["pca_min_rank"]),
            maximum_rank=int(fit_cfg["pca_max_rank"]),
        )
        coefficient, intercept, condition = nominal_fit._weighted_ridge(  # noqa: SLF001
            projection.transform(hidden),
            labels.astype(np.float64) - base_prediction,
            _weights(contract),
            float(fit_cfg["fixed_ridge_lambda"]),
        )
        delta_weight = coefficient @ projection.basis
        delta_bias = intercept
        base_weight, base_bias = nominal_fit._initializer_arrays(base_model)  # noqa: SLF001
        candidate_model, candidate_bytes = nominal_fit.base_bc.export_final_affine_model(
            base_model,
            base_weight.astype(np.float64) + delta_weight,
            base_bias.astype(np.float64) + delta_bias,
        )
        nominal_fit.base_bc.assert_only_final_affine_changed(base_model, candidate_model)
        session = nominal_fit.base_bc._cpu_ort_session(candidate_bytes)  # noqa: SLF001
        prediction = _ort_prediction(session, decoder994)
        formula = base_prediction + hidden @ delta_weight.T + delta_bias
        parity = float(np.max(np.abs(prediction - formula)))
        bootstrap, _ = load_bootstrap_training_candidate(
            paths["bootstrap_npz"], paths["bootstrap_manifest"], repository_root=root
        )
        _, bootstrap_base, bootstrap_runtime = nominal_fit._extract_hidden_variable(  # noqa: SLF001
            base_model, bootstrap["decoder994"]
        )
        bootstrap_candidate = _ort_prediction(session, bootstrap["decoder994"])

    slices = {
        "nominal": slice(0, 4080),
        "intervention": slice(4080, 4590),
        "seed835_teacher": slice(4590, 5100),
        "round1_counterexample": slice(5100, 5189),
        "round2_counterexample": slice(5189, 5357),
        "round2_last50": slice(5307, 5357),
    }
    metrics = {
        name: {
            "base": _metrics(base_prediction[index], labels[index]),
            "candidate": _metrics(prediction[index], labels[index]),
        }
        for name, index in slices.items()
    }
    bootstrap_metrics = {
        "base": _metrics(bootstrap_base, bootstrap["teacher_label_raw_native23"]),
        "candidate": _metrics(bootstrap_candidate, bootstrap["teacher_label_raw_native23"]),
    }
    all_metrics = _metrics(prediction, labels)
    gates = _mapping(contract["gates"], "gates")
    issues: list[str] = []
    if condition > float(gates["maximum_ridge_condition_number"]):
        issues.append("ridge_condition_gate_failed")
    for name, key in (
        ("round2_counterexample", "maximum_round2_all_rmse_ratio_to_base"),
        ("round2_last50", "maximum_round2_last50_rmse_ratio_to_base"),
    ):
        if metrics[name]["candidate"]["rmse"] > float(gates[key]) * metrics[name]["base"]["rmse"]:
            issues.append(f"{name}_improvement_gate_failed")
    for name, key in (
        ("round1_counterexample", "maximum_round1_rmse_regression"),
        ("seed835_teacher", "maximum_seed835_teacher_rmse_regression"),
        ("nominal", "maximum_nominal_rmse_regression"),
        ("intervention", "maximum_prior_intervention_rmse_regression"),
    ):
        if metrics[name]["candidate"]["rmse"] > metrics[name]["base"]["rmse"] + float(gates[key]):
            issues.append(f"{name}_preservation_gate_failed")
    if bootstrap_metrics["candidate"]["rmse"] > bootstrap_metrics["base"]["rmse"] + float(
        gates["maximum_bootstrap_rmse_regression"]
    ):
        issues.append("bootstrap_preservation_gate_failed")
    if all_metrics["max_abs"] > float(gates["maximum_training_absolute_error"]):
        issues.append("maximum_absolute_error_gate_failed")
    if float(np.max(np.abs(prediction))) >= float(gates["plain_raw_absolute_strict_max"]):
        issues.append("plain_raw_gate_failed")
    if parity > float(gates["maximum_export_reference_absolute_error"]):
        issues.append("export_parity_gate_failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "classification": (
            "offline_bc_eligible_for_closed_loop_simulator_experiment" if not issues else "offline_bc_rejected"
        ),
        "eligible_for_closed_loop_simulator_experiment": not issues,
        "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "lineage": {name: entry["sha256"] for name, entry in _mapping(contract["inputs"], "inputs").items()},
        "fit": {
            "fixed_ridge_lambda": fit_cfg["fixed_ridge_lambda"],
            "pca_rank": projection.rank,
            "pca_explained_fraction": projection.explained_fraction,
            "condition": condition,
            "delta_weight_frobenius": float(np.linalg.norm(delta_weight)),
            "delta_bias_l2": float(np.linalg.norm(delta_bias)),
        },
        "metrics": {**metrics, "bootstrap": bootstrap_metrics, "all_candidate": all_metrics},
        "export": {
            "candidate_decoder_sha256": nominal_fit._sha256_bytes(candidate_bytes),  # noqa: SLF001
            "candidate_decoder_size_bytes": len(candidate_bytes),
            "numpy_ort_max_abs": parity,
            "only_final_affine_changed_from_base": True,
        },
        "runtime": {"union": runtime, "bootstrap": bootstrap_runtime},
        "preflight": pre,
        "gate_issues": sorted(set(issues)),
        "boundaries": dict(contract["boundaries"]),
    }
    return nominal_fit.FitOutcome(
        passed=not issues,
        decoder_bytes=candidate_bytes if not issues else None,
        report=report,
    )


__all__ = [
    "CONTRACT_SHA256",
    "MANIFEST_KIND",
    "load_contract",
    "preflight",
    "run_fit",
    "validate_candidate_manifest_fields",
]
