"""Fixed-rank state-separated final-affine BC fit for seed835 round3 data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
    g1_true23_sonic_seed835_counterexample_round2_bc_last_affine as round2_fit,
    g1_true23_sonic_seed835_counterexample_round3_bc_last_affine as round3_fit,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    load_bootstrap_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_seed835_counterexample_round3_rank256_bc_last_affine_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_seed835_counterexample_round3_rank256_bc_last_affine_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_seed835_counterexample_round3_rank256_bc_last_affine_v1.json"
)
CONTRACT_SHA256 = "8ea44f1d05ae06c7e1405a4514f2346c92505f99e3ab99ce11108ad38a74f308"
TOTAL_ROWS = 5823
DECODER_DIM = 994
EXACT_RANK = 256


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if nominal_fit.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("rank256 BC contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    fit = _mapping(body.get("fit"), "fit")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != CONTRACT_KIND
        or body.get("rows", {}).get("total") != TOTAL_ROWS
        or fit.get("only_final_affine_may_change") is not True
        or fit.get("pca_exact_rank") != EXACT_RANK
        or fit.get("fixed_ridge_lambda") != 0.0001
        or boundaries.get("rank256_selected_by_read_only_state_separation_diagnostic") is not True
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("rank256 BC contract semantic drift")
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
        raise ValueError("rank256 candidate payload hash mismatch")
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
        raise ValueError("rank256 candidate manifest semantic drift")


def _load_inputs(root: Path, paths: Mapping[str, Path]) -> tuple[dict[str, np.ndarray], ...]:
    return round3_fit._load_inputs(root, paths)  # noqa: SLF001


def _weights(contract: Mapping[str, Any]) -> np.ndarray:
    return round3_fit._weights(contract)  # noqa: SLF001


def _fixed_projection(hidden: np.ndarray) -> nominal_fit.base_bc.Projection:
    values = np.asarray(hidden, dtype=np.float64)
    mean = values.mean(axis=0, dtype=np.float64)
    _, singular, vt = nominal_fit.base_bc.canonical_svd(values - mean)
    energy = singular * singular
    cumulative = np.cumsum(energy, dtype=np.float64) / float(np.sum(energy, dtype=np.float64))
    total = float(np.sum(energy, dtype=np.float64))
    centered_score_rms = math.sqrt(total / (values.shape[0] * EXACT_RANK))
    return nominal_fit.base_bc.Projection(
        mean=mean,
        basis=vt[:EXACT_RANK].copy(),
        centered_score_rms=centered_score_rms,
        rank=EXACT_RANK,
        explained_fraction=float(cumulative[EXACT_RANK - 1]),
        singular_values=singular[:EXACT_RANK].copy(),
    )


def preflight(request: nominal_fit.FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    datasets = _load_inputs(root, paths)
    base_manifest = _mapping(json.loads(paths["base_manifest"].read_text(encoding="utf-8")), "base manifest")
    round2_fit.validate_candidate_manifest_fields(base_manifest, round2_fit.load_contract(root))
    if sum(data["decoder994"].shape[0] for data in datasets) != TOTAL_ROWS:
        raise ValueError("rank256 fit input row drift")
    rejected = _mapping(
        json.loads(paths["rejected_rank42_manifest"].read_text(encoding="utf-8")), "rejected rank42 manifest"
    )
    if rejected.get("eligible_for_closed_loop_simulator_experiment") is not False or not rejected.get(
        "gate_issues"
    ):
        raise ValueError("rank42 rejection evidence drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "total_rows": TOTAL_ROWS,
        "pca_exact_rank": EXACT_RANK,
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
        projection = _fixed_projection(hidden)
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
        "round3_counterexample": slice(5357, 5823),
        "round3_last50": slice(5773, 5823),
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
        ("round3_counterexample", "maximum_round3_all_rmse_ratio_to_base"),
        ("round3_last50", "maximum_round3_last50_rmse_ratio_to_base"),
    ):
        if metrics[name]["candidate"]["rmse"] > float(gates[key]) * metrics[name]["base"]["rmse"]:
            issues.append(f"{name}_improvement_gate_failed")
    for name, key in (
        ("round2_counterexample", "maximum_round2_rmse_regression"),
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
