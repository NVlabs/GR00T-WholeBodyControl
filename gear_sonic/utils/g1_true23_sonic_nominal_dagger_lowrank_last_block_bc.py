"""Low-rank penultimate-block plus final-head BC after final-affine DAgger saturation."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_bc_last_affine_ridge as base_fit,
    g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge as round2,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    load_bootstrap_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_manifest_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_v1.json"
)
CONTRACT_SHA256 = "12f1e7e3f7695eb173f2d056b5fcf9e5d58bb9c5e2f902c85a23c955adaf058d"
TOTAL_ROWS = 5100
OPTIMIZATION_ROWS = 4080
DECODER_DIM = 994
FEATURE_DIM = 512
ACTION_DIM = 23
PENULTIMATE_INPUT_NAME = "/Mul_6_output_0"
PENULTIMATE_PREACT_NAME = "/layers.7/Gemm_output_0"
TRAINABLE_INITIALIZERS = (
    "layers.7.weight",
    "layers.7.bias",
    "layers.8.weight",
    "layers.8.bias",
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if nominal_fit.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("low-rank BC contract hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    rows = _mapping(body.get("rows"), "rows")
    fit = _mapping(body.get("fit"), "fit")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != CONTRACT_KIND
        or rows
        != {
            "nominal_training_runs": 6,
            "nominal_heldout_runs": 2,
            "nominal_training": 3060,
            "nominal_heldout": 1020,
            "cutoff50": 510,
            "cutoff140": 510,
            "optimization_total": OPTIMIZATION_ROWS,
            "evaluation_total": TOTAL_ROWS,
        }
        or fit.get("frozen_decoder_layers") != [0, 1, 2, 3, 4, 5, 6]
        or fit.get("penultimate_delta_rank") != 8
        or fit.get("optimizer_steps") != 800
        or fit.get("batch_size") != 512
        or fit.get("device") != "cuda:0"
        or boundaries.get("support_qualified") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("low-rank BC contract semantic drift")
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
        raise ValueError("low-rank candidate payload hash mismatch")
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
        raise ValueError("low-rank candidate manifest semantic drift")


def optimization_indices() -> np.ndarray:
    value = np.concatenate((np.arange(0, 3060), np.arange(4080, 5100))).astype(np.int64)
    if value.shape != (OPTIMIZATION_ROWS,) or np.unique(value).size != OPTIMIZATION_ROWS:
        raise RuntimeError("optimization split invalid")
    return value


def heldout_indices() -> np.ndarray:
    return np.arange(3060, 4080, dtype=np.int64)


def optimization_weights(contract: Mapping[str, Any]) -> np.ndarray:
    fit = _mapping(contract["fit"], "fit")
    nominal_pattern = np.concatenate(
        (
            np.full(10, float(fit["nominal_reset_prefix_weight"]), dtype=np.float64),
            np.full(500, float(fit["nominal_real_h10_weight"]), dtype=np.float64),
        )
    )
    value = np.concatenate(
        (
            np.tile(nominal_pattern, 6),
            np.full(50, float(fit["cutoff50_student_shadow_weight"]), dtype=np.float64),
            np.full(460, float(fit["cutoff50_teacher_recovery_weight"]), dtype=np.float64),
            np.full(140, float(fit["cutoff140_student_shadow_weight"]), dtype=np.float64),
            np.full(370, float(fit["cutoff140_teacher_recovery_weight"]), dtype=np.float64),
        )
    )
    value *= OPTIMIZATION_ROWS / float(value.sum())
    if value.shape != (OPTIMIZATION_ROWS,) or not math.isclose(float(value.mean()), 1.0, abs_tol=1e-14):
        raise RuntimeError("optimization weights invalid")
    return value


def _initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {
        item.name: np.ascontiguousarray(numpy_helper.to_array(item)).copy() for item in model.graph.initializer
    }


def _extract_block_features(
    model: Any, decoder994: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    from onnx import TensorProto, helper
    import onnxruntime as ort

    values = np.asarray(decoder994, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != DECODER_DIM or not bool(np.isfinite(values).all()):
        raise ValueError("decoder input matrix invalid")
    probe = copy.deepcopy(model)
    outputs = {value.name for value in probe.graph.output}
    for name in (PENULTIMATE_INPUT_NAME, PENULTIMATE_PREACT_NAME):
        if name in outputs:
            raise ValueError("decoder unexpectedly exports training probe")
        probe.graph.output.extend([helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, FEATURE_DIM])])
    session = nominal_fit.base_bc._cpu_ort_session(probe.SerializeToString(deterministic=True))  # noqa: SLF001
    x_rows: list[np.ndarray] = []
    pre_rows: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for row in values:
        action, x_value, pre_value = session.run(
            ["action", PENULTIMATE_INPUT_NAME, PENULTIMATE_PREACT_NAME],
            {"obs_dict": row.reshape(1, DECODER_DIM)},
        )
        actions.append(action[0])
        x_rows.append(x_value[0])
        pre_rows.append(pre_value[0])
    x = np.asarray(x_rows, dtype=np.float32)
    pre = np.asarray(pre_rows, dtype=np.float32)
    output = np.asarray(actions, dtype=np.float32)
    expected = (values.shape[0], FEATURE_DIM)
    if x.shape != expected or pre.shape != expected or output.shape != (values.shape[0], ACTION_DIM):
        raise RuntimeError("block probe output shape mismatch")
    return (
        x,
        pre,
        output,
        {
            "provider": session.get_providers()[0],
            "onnxruntime_version": ort.__version__,
            "row_count": values.shape[0],
            "host_reads_inside_simulator_step": False,
        },
    )


def _export_model(
    base_model: Any,
    *,
    layer7_weight: np.ndarray,
    layer7_bias: np.ndarray,
    layer8_weight: np.ndarray,
    layer8_bias: np.ndarray,
) -> tuple[Any, bytes]:
    from onnx import numpy_helper

    candidate = copy.deepcopy(base_model)
    replacements = {
        "layers.7.weight": np.asarray(layer7_weight, dtype=np.float32),
        "layers.7.bias": np.asarray(layer7_bias, dtype=np.float32),
        "layers.8.weight": np.asarray(layer8_weight, dtype=np.float32),
        "layers.8.bias": np.asarray(layer8_bias, dtype=np.float32),
    }
    found: set[str] = set()
    for index, initializer in enumerate(candidate.graph.initializer):
        if initializer.name in replacements:
            candidate.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(replacements[initializer.name]), initializer.name)
            )
            found.add(initializer.name)
    if found != set(TRAINABLE_INITIALIZERS):
        raise ValueError("trainable initializer inventory drift")
    import onnx

    onnx.checker.check_model(candidate, full_check=True)
    return candidate, candidate.SerializeToString(deterministic=True)


def _assert_only_last_block_changed(base_model: Any, candidate_model: Any) -> None:
    base_values = _initializer_arrays(base_model)
    candidate_values = _initializer_arrays(candidate_model)
    if tuple(base_values) != tuple(candidate_values):
        raise ValueError("initializer order drift")
    changed = {
        name
        for name in base_values
        if not np.array_equal(base_values[name], candidate_values[name], equal_nan=True)
    }
    if changed != set(TRAINABLE_INITIALIZERS):
        raise ValueError("candidate changed outside exact last block")
    if [node.SerializeToString() for node in base_model.graph.node] != [
        node.SerializeToString() for node in candidate_model.graph.node
    ]:
        raise ValueError("candidate decoder graph changed")


def _formula_prediction(
    preactivation: np.ndarray,
    scores: np.ndarray,
    adapter_b: np.ndarray,
    delta_b7: np.ndarray,
    weight8: np.ndarray,
    bias8: np.ndarray,
) -> np.ndarray:
    import torch

    with torch.no_grad():
        pre = torch.from_numpy(np.asarray(preactivation, dtype=np.float32))
        score = torch.from_numpy(np.asarray(scores, dtype=np.float32))
        b = torch.from_numpy(np.asarray(adapter_b, dtype=np.float32))
        db = torch.from_numpy(np.asarray(delta_b7, dtype=np.float32))
        w8 = torch.from_numpy(np.asarray(weight8, dtype=np.float32))
        b8 = torch.from_numpy(np.asarray(bias8, dtype=np.float32))
        return (torch.nn.functional.silu(pre + score @ b.T + db) @ w8.T + b8).numpy()


def _train_adapter(
    *,
    preactivation: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    base_weight8: np.ndarray,
    base_bias8: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    fit = _mapping(contract["fit"], "fit")
    if not torch.cuda.is_available():
        raise RuntimeError("contract requires CUDA for low-rank fit")
    torch.manual_seed(int(fit["seed"]))
    torch.cuda.manual_seed_all(int(fit["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(str(fit["device"]))
    pre = torch.from_numpy(np.asarray(preactivation, dtype=np.float32)).to(device)
    score = torch.from_numpy(np.asarray(scores, dtype=np.float32)).to(device)
    target = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device)
    row_weight = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(device)
    base_w8 = torch.from_numpy(np.asarray(base_weight8, dtype=np.float32)).to(device)
    base_b8 = torch.from_numpy(np.asarray(base_bias8, dtype=np.float32)).to(device)
    rank = int(fit["penultimate_delta_rank"])
    adapter_b = torch.nn.Parameter(torch.zeros((FEATURE_DIM, rank), dtype=torch.float32, device=device))
    delta_b7 = torch.nn.Parameter(torch.zeros(FEATURE_DIM, dtype=torch.float32, device=device))
    delta_w8 = torch.nn.Parameter(torch.zeros((ACTION_DIM, FEATURE_DIM), dtype=torch.float32, device=device))
    delta_b8 = torch.nn.Parameter(torch.zeros(ACTION_DIM, dtype=torch.float32, device=device))
    parameters = (adapter_b, delta_b7, delta_w8, delta_b8)
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(fit["learning_rate"]),
        betas=tuple(map(float, fit["betas"])),
        eps=float(fit["epsilon"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(fit["seed"]))
    batch_size = int(fit["batch_size"])
    permutation = torch.randperm(pre.shape[0], generator=generator)
    cursor = 0
    final_loss = math.inf
    final_data_loss = math.inf
    maximum_gradient_norm = 0.0
    for _step in range(int(fit["optimizer_steps"])):
        if cursor + batch_size > pre.shape[0]:
            permutation = torch.randperm(pre.shape[0], generator=generator)
            cursor = 0
        index = permutation[cursor : cursor + batch_size].to(device)
        cursor += batch_size
        z = pre[index] + score[index] @ adapter_b.T + delta_b7
        prediction = torch.nn.functional.silu(z) @ (base_w8 + delta_w8).T + base_b8 + delta_b8
        error = torch.mean((prediction - target[index]) ** 2, dim=1)
        data_loss = torch.sum(error * row_weight[index]) / torch.sum(row_weight[index])
        trust = float(fit["parameter_trust_l2"]) * sum(torch.mean(value * value) for value in parameters)
        loss = data_loss + trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(fit["gradient_clip_norm"]))
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("low-rank optimizer became nonfinite")
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_data_loss = float(data_loss.detach().cpu())
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.detach().cpu()))
    result = {
        "adapter_b": adapter_b.detach().cpu().numpy().copy(),
        "delta_b7": delta_b7.detach().cpu().numpy().copy(),
        "delta_w8": delta_w8.detach().cpu().numpy().copy(),
        "delta_b8": delta_b8.detach().cpu().numpy().copy(),
    }
    torch.cuda.synchronize(device)
    return result, {
        "optimizer": "adam",
        "optimizer_steps": int(fit["optimizer_steps"]),
        "batch_size": batch_size,
        "final_loss": final_loss,
        "final_data_loss": final_data_loss,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "device": str(device),
        "deterministic_algorithms": True,
        "tf32": False,
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, Any]:
    return nominal_fit._metrics(prediction, target)  # noqa: SLF001


def _fixed_rank_projection(hidden: np.ndarray, rank: int) -> Any:
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= rank or rank <= 0:
        raise ValueError("fixed-rank projection input invalid")
    mean = values.mean(axis=0, dtype=np.float64)
    _, singular, vt = nominal_fit.base_bc.canonical_svd(values - mean)
    energy = singular * singular
    total = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("fixed-rank projection variance invalid")
    explained = float(np.sum(energy[:rank], dtype=np.float64) / total)
    return nominal_fit.base_bc.Projection(
        mean=mean,
        basis=vt[:rank].copy(),
        centered_score_rms=math.sqrt(total / (values.shape[0] * rank)),
        rank=rank,
        explained_fraction=explained,
        singular_values=singular[:rank].copy(),
    )


def preflight(request: nominal_fit.FitRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = _paths(root, contract)
    nominal, old, new = round2._load_inputs(root, paths)  # noqa: SLF001
    base_manifest = json.loads(paths["base_manifest"].read_text(encoding="utf-8"))
    base_fit.validate_candidate_manifest_fields(base_manifest, base_fit.load_contract(root))
    failure = json.loads(paths["round2_failed_student_report"].read_text(encoding="utf-8"))
    first_done = _mapping(failure.get("first_done"), "round2 first done")
    if (
        failure.get("verdict") != "student_qualification_failed"
        or failure.get("attempted_transitions") != 220
        or first_done.get("transition") != 219
        or first_done.get("q9_before") != 228
        or first_done.get("termination_names") != ["anchor_pos", "ee_body_pos"]
        or failure.get("safety", {}).get("teacher_queries") != 0
    ):
        raise ValueError("round2 closed-loop failure evidence drift")
    if nominal["decoder994"].shape != (4080, DECODER_DIM):
        raise ValueError("nominal input shape drift")
    if old["decoder994"].shape != (510, DECODER_DIM) or new["decoder994"].shape != (510, DECODER_DIM):
        raise ValueError("intervention input shape drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "optimization_rows": OPTIMIZATION_ROWS,
        "heldout_rows": 1020,
        "evaluation_rows": TOTAL_ROWS,
        "penultimate_delta_rank": 8,
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
    preflight_receipt = preflight(request)
    nominal, old, new = round2._load_inputs(root, paths)  # noqa: SLF001
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
    base_initializers = _initializer_arrays(base_model)
    with nominal_fit._numeric_runtime():  # noqa: SLF001
        x, preactivation, base_prediction, runtime = _extract_block_features(base_model, decoder994)
        train_index = optimization_indices()
        fit_cfg = _mapping(contract["fit"], "fit")
        projection = _fixed_rank_projection(
            x[train_index].astype(np.float64), int(fit_cfg["penultimate_delta_rank"])
        )
        basis = projection.basis.astype(np.float32)
        scores = x @ basis.T
        trained, optimizer_report = _train_adapter(
            preactivation=preactivation[train_index],
            scores=scores[train_index],
            labels=labels[train_index],
            weights=optimization_weights(contract),
            base_weight8=base_initializers["layers.8.weight"],
            base_bias8=base_initializers["layers.8.bias"],
            contract=contract,
        )
        delta_w7 = trained["adapter_b"] @ basis
        weight7 = base_initializers["layers.7.weight"] + delta_w7
        bias7 = base_initializers["layers.7.bias"] + trained["delta_b7"]
        weight8 = base_initializers["layers.8.weight"] + trained["delta_w8"]
        bias8 = base_initializers["layers.8.bias"] + trained["delta_b8"]
        candidate_model, candidate_bytes = _export_model(
            base_model,
            layer7_weight=weight7,
            layer7_bias=bias7,
            layer8_weight=weight8,
            layer8_bias=bias8,
        )
        _assert_only_last_block_changed(base_model, candidate_model)
        session = nominal_fit.base_bc._cpu_ort_session(candidate_bytes)  # noqa: SLF001
        exported = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in decoder994
            ],
            axis=0,
        )
        formula = _formula_prediction(
            preactivation,
            scores,
            trained["adapter_b"],
            trained["delta_b7"],
            weight8,
            bias8,
        )
        union_parity = float(np.max(np.abs(exported - formula)))
        bootstrap, _ = load_bootstrap_training_candidate(
            paths["bootstrap_npz"], paths["bootstrap_manifest"], repository_root=root
        )
        bootstrap_x, bootstrap_pre, bootstrap_base, bootstrap_runtime = _extract_block_features(
            base_model, bootstrap["decoder994"]
        )
        bootstrap_scores = bootstrap_x @ basis.T
        bootstrap_exported = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM).astype(np.float32)})[0]
                for row in bootstrap["decoder994"]
            ],
            axis=0,
        )
        bootstrap_formula = _formula_prediction(
            bootstrap_pre,
            bootstrap_scores,
            trained["adapter_b"],
            trained["delta_b7"],
            weight8,
            bias8,
        )
        bootstrap_parity = float(np.max(np.abs(bootstrap_exported - bootstrap_formula)))

    slices = {
        "nominal_training": slice(0, 3060),
        "nominal_heldout": slice(3060, 4080),
        "cutoff50": slice(4080, 4590),
        "cutoff50_shadow": slice(4080, 4130),
        "cutoff140": slice(4590, 5100),
        "cutoff140_shadow": slice(4590, 4730),
    }
    metrics = {
        name: {
            "base": _metrics(base_prediction[index], labels[index]),
            "candidate": _metrics(exported[index], labels[index]),
        }
        for name, index in slices.items()
    }
    bootstrap_metrics = {
        "base": _metrics(bootstrap_base, bootstrap["teacher_label_raw_native23"]),
        "candidate": _metrics(bootstrap_exported, bootstrap["teacher_label_raw_native23"]),
    }
    all_metrics = _metrics(exported, labels)
    gates = _mapping(contract["gates"], "gates")
    delta7_relative = float(np.linalg.norm(delta_w7) / np.linalg.norm(base_initializers["layers.7.weight"]))
    delta8_relative = float(
        np.linalg.norm(trained["delta_w8"]) / np.linalg.norm(base_initializers["layers.8.weight"])
    )
    parity = max(union_parity, bootstrap_parity)
    issues: list[str] = []
    if metrics["cutoff140_shadow"]["candidate"]["rmse"] > (
        float(gates["maximum_cutoff140_shadow_rmse_ratio_to_base"]) * metrics["cutoff140_shadow"]["base"]["rmse"]
    ):
        issues.append("cutoff140_shadow_ratio_gate_failed")
    if metrics["cutoff140_shadow"]["candidate"]["rmse"] > float(gates["maximum_cutoff140_shadow_rmse"]):
        issues.append("cutoff140_shadow_absolute_gate_failed")
    if metrics["cutoff140"]["candidate"]["rmse"] > (
        float(gates["maximum_cutoff140_all_rmse_ratio_to_base"]) * metrics["cutoff140"]["base"]["rmse"]
    ):
        issues.append("cutoff140_all_gate_failed")
    for name, gate in (
        ("cutoff50_shadow", "maximum_cutoff50_shadow_rmse_regression"),
        ("cutoff50", "maximum_cutoff50_all_rmse_regression"),
        ("nominal_training", "maximum_nominal_training_rmse_regression"),
        ("nominal_heldout", "maximum_nominal_heldout_rmse_regression"),
    ):
        if metrics[name]["candidate"]["rmse"] > metrics[name]["base"]["rmse"] + float(gates[gate]):
            issues.append(f"{name}_preservation_gate_failed")
    if bootstrap_metrics["candidate"]["rmse"] > bootstrap_metrics["base"]["rmse"] + float(
        gates["maximum_bootstrap_rmse_regression"]
    ):
        issues.append("bootstrap_preservation_gate_failed")
    if all_metrics["max_abs"] > float(gates["maximum_training_absolute_error"]):
        issues.append("maximum_absolute_error_gate_failed")
    if delta7_relative > float(gates["maximum_penultimate_weight_delta_relative_l2"]):
        issues.append("penultimate_delta_trust_gate_failed")
    if delta8_relative > float(gates["maximum_final_weight_delta_relative_l2"]):
        issues.append("final_delta_trust_gate_failed")
    if max(float(np.max(np.abs(exported))), float(np.max(np.abs(bootstrap_exported)))) >= float(
        gates["plain_raw_absolute_strict_max"]
    ):
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
            "rank": projection.rank,
            "basis_explained_fraction": projection.explained_fraction,
            "trainable_parameter_count": 16407,
            "delta_layer7_weight_relative_l2": delta7_relative,
            "delta_layer8_weight_relative_l2": delta8_relative,
            "delta_layer7_bias_l2": float(np.linalg.norm(trained["delta_b7"])),
            "delta_layer8_bias_l2": float(np.linalg.norm(trained["delta_b8"])),
            "optimizer": optimizer_report,
        },
        "metrics": {**metrics, "bootstrap": bootstrap_metrics, "all_candidate": all_metrics},
        "export": {
            "candidate_decoder_sha256": nominal_fit._sha256_bytes(candidate_bytes),  # noqa: SLF001
            "candidate_decoder_size_bytes": len(candidate_bytes),
            "numpy_ort_max_abs": parity,
            "only_layers7_and8_changed_from_base": True,
        },
        "runtime": {"union": runtime, "bootstrap": bootstrap_runtime},
        "preflight": preflight_receipt,
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
    "heldout_indices",
    "load_contract",
    "optimization_indices",
    "optimization_weights",
    "preflight",
    "run_fit",
    "validate_candidate_manifest_fields",
]
