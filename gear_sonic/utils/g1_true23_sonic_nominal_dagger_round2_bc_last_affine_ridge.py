"""Final-affine SONIC refit on nominal plus cutoff50 and cutoff140 intervention rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_bc_last_affine_ridge as prior_fit,
    g1_true23_sonic_nominal_dagger_cutoff50_collection as cutoff50,
    g1_true23_sonic_nominal_dagger_cutoff140_collection as cutoff140,
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
CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_v1.json"
)
CONTRACT_SHA256 = "67244c6933d261c0c649f0488d427b2ceb7a79ef676298881d04f332f3d79d75"
NOMINAL_ROWS = 4080
CUTOFF50_ROWS = 510
CUTOFF140_ROWS = 510
TOTAL_ROWS = 5100
ACTION_DIM = 23
DECODER_DIM = 994


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if nominal_fit.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("round2 DAgger BC contract hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    rows = _mapping(body.get("rows"), "rows")
    fit = _mapping(body.get("fit"), "fit")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != CONTRACT_KIND
        or rows
        != {
            "nominal": 4080,
            "cutoff50_student_shadow": 50,
            "cutoff50_teacher_recovery": 460,
            "cutoff140_student_shadow": 140,
            "cutoff140_teacher_recovery": 370,
            "total": TOTAL_ROWS,
        }
        or fit.get("only_final_affine_may_change") is not True
        or fit.get("fixed_ridge_lambda") != 0.0001
        or fit.get("cutoff50_student_shadow_weight") != 8.0
        or fit.get("cutoff140_student_shadow_weight") != 16.0
        or boundaries.get("support_qualified") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("round2 DAgger BC contract semantic drift")
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
        raise ValueError("round2 candidate payload hash mismatch")
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
        or artifact.get("publication_protocol") != "decoder_hardlink_then_manifest_hardlink_commit"
        or artifact.get("decoder_sha256") != manifest.get("export", {}).get("candidate_decoder_sha256")
        or manifest.get("boundaries") != contract.get("boundaries")
    ):
        raise ValueError("round2 candidate manifest semantic drift")


def _load_intervention(
    root: Path,
    npz: Path,
    manifest: Path,
    *,
    kind: str,
    student_rows: int,
    teacher_rows: int,
    use_cutoff140_schema: bool,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    body = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        body.get("kind") != kind
        or body.get("artifact", {}).get("npz_sha256") != nominal_fit.sha256_file(npz)
        or body.get("rows")
        != {
            "total": 510,
            "student_on_policy_shadow_label_rows": student_rows,
            "teacher_actuated_recovery_rows": teacher_rows,
        }
        or body.get("boundaries", {}).get("support_qualified") is not False
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("intervention manifest semantic drift")
    payload = dict(body)
    claimed = payload.pop("manifest_payload_sha256", None)
    if claimed != cutoff50._sha256_bytes(cutoff50._canonical_bytes(payload)):  # noqa: SLF001
        raise ValueError("intervention manifest payload mismatch")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]).copy() for name in archive.files}
    if use_cutoff140_schema:
        with cutoff140._base_patch(root):  # noqa: SLF001
            cutoff50._validate_arrays(arrays)  # noqa: SLF001
    else:
        cutoff50._validate_arrays(arrays)  # noqa: SLF001
    for name, value in arrays.items():
        expected = body["artifact"]["array_schema"][name]
        if expected != {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": cutoff50._array_sha(value),  # noqa: SLF001
        }:
            raise ValueError(f"intervention array hash drift: {name}")
    return arrays, body


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
            np.full(50, float(fit["cutoff50_student_shadow_weight"]), dtype=np.float64),
            np.full(460, float(fit["cutoff50_teacher_recovery_weight"]), dtype=np.float64),
            np.full(140, float(fit["cutoff140_student_shadow_weight"]), dtype=np.float64),
            np.full(370, float(fit["cutoff140_teacher_recovery_weight"]), dtype=np.float64),
        )
    )
    value *= TOTAL_ROWS / float(value.sum())
    if value.shape != (TOTAL_ROWS,) or not math.isclose(float(value.mean()), 1.0, abs_tol=1e-14):
        raise RuntimeError("round2 fit weights invalid")
    return value


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, Any]:
    return nominal_fit._metrics(prediction, target)  # noqa: SLF001


def _load_inputs(
    root: Path, paths: Mapping[str, Path]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    nominal_prefix = Path(str(paths["nominal_npz"])[: -len(".npz")])
    nominal, _ = load_training_candidate(CollectionRequest(root, nominal_prefix))
    old, _ = _load_intervention(
        root,
        paths["cutoff50_npz"],
        paths["cutoff50_manifest"],
        kind=cutoff50.MANIFEST_KIND,
        student_rows=50,
        teacher_rows=460,
        use_cutoff140_schema=False,
    )
    new, _ = _load_intervention(
        root,
        paths["cutoff140_npz"],
        paths["cutoff140_manifest"],
        kind=cutoff140.MANIFEST_KIND,
        student_rows=140,
        teacher_rows=370,
        use_cutoff140_schema=True,
    )
    return nominal, old, new


def preflight(request: nominal_fit.FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    nominal, old, new = _load_inputs(root, paths)
    base_manifest = json.loads(paths["base_manifest"].read_text(encoding="utf-8"))
    prior_fit.validate_candidate_manifest_fields(base_manifest, prior_fit.load_contract(root))
    if nominal["decoder994"].shape != (NOMINAL_ROWS, DECODER_DIM):
        raise ValueError("nominal input shape drift")
    if old["decoder994"].shape != (CUTOFF50_ROWS, DECODER_DIM):
        raise ValueError("cutoff50 input shape drift")
    if new["decoder994"].shape != (CUTOFF140_ROWS, DECODER_DIM):
        raise ValueError("cutoff140 input shape drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "nominal_rows": NOMINAL_ROWS,
        "cutoff50_rows": CUTOFF50_ROWS,
        "cutoff140_rows": CUTOFF140_ROWS,
        "student_on_policy_shadow_rows": 190,
        "teacher_actuated_recovery_rows": 830,
        "base_decoder_sha256": contract["inputs"]["base_decoder"]["sha256"],
        "simulator_constructed": False,
        "training_updates": 0,
        "hardware_authorized": False,
    }


def run_fit(request: nominal_fit.FitRequest) -> nominal_fit.FitOutcome:
    import onnx

    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    pre = preflight(request)
    nominal, old, new = _load_inputs(root, paths)
    decoder994 = np.concatenate((nominal["decoder994"], old["decoder994"], new["decoder994"]), axis=0)
    labels = np.concatenate(
        (
            nominal["teacher_label_raw_native23"],
            old["teacher_label_raw_native23"],
            new["teacher_label_raw_native23"],
        ),
        axis=0,
    )
    base_model = onnx.load(paths["base_decoder"], load_external_data=False)
    parent_model = onnx.load(paths["parent_base_decoder"], load_external_data=False)
    onnx.checker.check_model(base_model, full_check=True)
    nominal_fit.base_bc.assert_only_final_affine_changed(parent_model, base_model)
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
        prediction = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in decoder994
            ],
            axis=0,
        )
        formula = base_prediction + hidden @ delta_weight.T + delta_bias
        parity = float(np.max(np.abs(prediction - formula)))
        bootstrap, _ = load_bootstrap_training_candidate(
            paths["bootstrap_npz"], paths["bootstrap_manifest"], repository_root=root
        )
        _, bootstrap_base, bootstrap_runtime = nominal_fit._extract_hidden_variable(  # noqa: SLF001
            base_model, bootstrap["decoder994"]
        )
        bootstrap_candidate = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in bootstrap["decoder994"]
            ],
            axis=0,
        )

    slices = {
        "nominal": slice(0, 4080),
        "cutoff50": slice(4080, 4590),
        "cutoff50_shadow": slice(4080, 4130),
        "cutoff140": slice(4590, 5100),
        "cutoff140_shadow": slice(4590, 4730),
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
    if metrics["cutoff140"]["candidate"]["rmse"] > (
        float(gates["maximum_cutoff140_intervention_rmse_ratio_to_base"]) * metrics["cutoff140"]["base"]["rmse"]
    ):
        issues.append("cutoff140_intervention_gate_failed")
    if metrics["cutoff140_shadow"]["candidate"]["rmse"] > (
        float(gates["maximum_cutoff140_shadow_rmse_ratio_to_base"]) * metrics["cutoff140_shadow"]["base"]["rmse"]
    ):
        issues.append("cutoff140_shadow_ratio_gate_failed")
    if metrics["cutoff140_shadow"]["candidate"]["rmse"] > float(gates["maximum_cutoff140_shadow_rmse"]):
        issues.append("cutoff140_shadow_absolute_gate_failed")
    if metrics["cutoff50"]["candidate"]["rmse"] > metrics["cutoff50"]["base"]["rmse"] + float(
        gates["maximum_cutoff50_intervention_rmse_regression"]
    ):
        issues.append("cutoff50_intervention_preservation_gate_failed")
    if metrics["cutoff50_shadow"]["candidate"]["rmse"] > metrics["cutoff50_shadow"]["base"]["rmse"] + float(
        gates["maximum_cutoff50_shadow_rmse_regression"]
    ):
        issues.append("cutoff50_shadow_preservation_gate_failed")
    if metrics["nominal"]["candidate"]["rmse"] > metrics["nominal"]["base"]["rmse"] + float(
        gates["maximum_nominal_rmse_regression"]
    ):
        issues.append("nominal_preservation_gate_failed")
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
            "weights": {
                "nominal_reset": 4.0,
                "nominal_h10": 1.0,
                "cutoff50_shadow": 8.0,
                "cutoff50_recovery": 2.0,
                "cutoff140_shadow": 16.0,
                "cutoff140_recovery": 4.0,
            },
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
