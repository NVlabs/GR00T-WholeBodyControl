"""Targeted final-affine refit using newly admitted seed835/921 teacher-reset rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_bc_last_affine_ridge as prior_fit,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    load_bootstrap_training_candidate,
)
from gear_sonic.utils.g1_true23_selected_teacher_nominal_multiseed_collection import (
    CollectionRequest,
    load_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_multiseed_round3_bc_last_affine_ridge_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_multiseed_round3_bc_last_affine_ridge_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_multiseed_round3_bc_last_affine_ridge_v1.json"
)
CONTRACT_SHA256 = "703bf79daf9ca1ec23a97ae7e5972ea42d44cd948f0aa1b9a1e2255e7fa07e6e"
NOMINAL_ROWS = 4080
SEGMENT_ROWS = 510
TOTAL_ROWS = 5610
ACTION_DIM = 23
DECODER_DIM = 994


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if nominal_fit.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("round3 multiseed BC contract hash mismatch")
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
            "seed921_teacher_reset": SEGMENT_ROWS,
            "total": TOTAL_ROWS,
        }
        or fit.get("only_final_affine_may_change") is not True
        or fit.get("fixed_ridge_lambda") != 0.0001
        or fit.get("seed835_first100_weight") != 12.0
        or fit.get("seed921_first100_weight") != 6.0
        or boundaries.get("final_affine_last_attempt_before_supervised_branch_retirement") is not True
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("round3 multiseed BC contract semantic drift")
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
        raise ValueError("round3 candidate payload hash mismatch")
    inputs = _mapping(contract["inputs"], "contract inputs")
    artifact = _mapping(manifest.get("artifact"), "candidate artifact")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("classification") != "offline_bc_eligible_for_closed_loop_simulator_experiment"
        or manifest.get("eligible_for_closed_loop_simulator_experiment") is not True
        or manifest.get("gate_issues") != []
        or manifest.get("contract") != {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        or manifest.get("lineage") != {name: entry["sha256"] for name, entry in inputs.items()}
        or artifact.get("publishable_decoder") is not True
        or artifact.get("decoder_sha256") != manifest.get("export", {}).get("candidate_decoder_sha256")
        or manifest.get("boundaries") != contract.get("boundaries")
    ):
        raise ValueError("round3 candidate manifest semantic drift")


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
            np.full(100, float(fit["seed835_first100_weight"]), dtype=np.float64),
            np.full(410, float(fit["seed835_tail410_weight"]), dtype=np.float64),
            np.full(100, float(fit["seed921_first100_weight"]), dtype=np.float64),
            np.full(410, float(fit["seed921_tail410_weight"]), dtype=np.float64),
        )
    )
    value *= TOTAL_ROWS / float(value.sum())
    if value.shape != (TOTAL_ROWS,) or not math.isclose(float(value.mean()), 1.0, abs_tol=1e-14):
        raise RuntimeError("round3 fit weights invalid")
    return value


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, Any]:
    return nominal_fit._metrics(prediction, target)  # noqa: SLF001


def _load_inputs(
    root: Path, paths: Mapping[str, Path]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    nominal_prefix = Path(str(paths["nominal_npz"])[: -len(".npz")])
    nominal, _ = load_training_candidate(CollectionRequest(root, nominal_prefix))
    intervention, _ = prior_fit._load_intervention(  # noqa: SLF001
        paths["intervention_npz"], paths["intervention_manifest"]
    )
    seed835, _ = load_bootstrap_training_candidate(
        paths["seed835_npz"], paths["seed835_manifest"], repository_root=root
    )
    seed921, _ = load_bootstrap_training_candidate(
        paths["seed921_npz"], paths["seed921_manifest"], repository_root=root
    )
    return nominal, intervention, seed835, seed921


def preflight(request: nominal_fit.FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    nominal, intervention, seed835, seed921 = _load_inputs(root, paths)
    base_manifest = _mapping(json.loads(paths["base_manifest"].read_text(encoding="utf-8")), "base manifest")
    prior_fit.validate_candidate_manifest_fields(base_manifest, prior_fit.load_contract(root))
    expected = (
        (nominal["decoder994"], (NOMINAL_ROWS, DECODER_DIM)),
        (intervention["decoder994"], (SEGMENT_ROWS, DECODER_DIM)),
        (seed835["decoder994"], (SEGMENT_ROWS, DECODER_DIM)),
        (seed921["decoder994"], (SEGMENT_ROWS, DECODER_DIM)),
    )
    if any(value.shape != shape for value, shape in expected):
        raise ValueError("round3 input shape drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "total_rows": TOTAL_ROWS,
        "seed835_rows": SEGMENT_ROWS,
        "seed921_rows": SEGMENT_ROWS,
        "base_decoder_sha256": contract["inputs"]["base_decoder"]["sha256"],
        "simulator_constructed": False,
        "training_updates": 0,
        "hardware_authorized": False,
    }


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
    nominal, intervention, seed835, seed921 = _load_inputs(root, paths)
    datasets = (nominal, intervention, seed835, seed921)
    decoder994 = np.concatenate([value["decoder994"] for value in datasets], axis=0)
    labels = np.concatenate([value["teacher_label_raw_native23"] for value in datasets], axis=0)
    base_model = onnx.load(paths["base_decoder"], load_external_data=False)
    onnx.checker.check_model(base_model, full_check=True)
    with nominal_fit._numeric_runtime():  # noqa: SLF001
        hidden, base_prediction, runtime = nominal_fit._extract_hidden_variable(base_model, decoder994)  # noqa: SLF001
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
        "seed835": slice(4590, 5100),
        "seed835_first100": slice(4590, 4690),
        "seed921": slice(5100, 5610),
        "seed921_first100": slice(5100, 5200),
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
        ("seed835", "maximum_seed835_all_rmse_ratio_to_base"),
        ("seed835_first100", "maximum_seed835_first100_rmse_ratio_to_base"),
        ("seed921", "maximum_seed921_all_rmse_ratio_to_base"),
        ("seed921_first100", "maximum_seed921_first100_rmse_ratio_to_base"),
    ):
        if metrics[name]["candidate"]["rmse"] > float(gates[key]) * metrics[name]["base"]["rmse"]:
            issues.append(f"{name}_improvement_gate_failed")
    for name, key in (
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
