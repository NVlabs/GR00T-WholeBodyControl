from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest

from gear_sonic.scripts.fit_g1_true23_native124_21204_bc_last_affine_ridge import (
    _parser,
)
from gear_sonic.utils import g1_true23_native124_21204_bc_last_affine_ridge as bc
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_final_affine_model() -> onnx.ModelProto:
    weight = np.linspace(-0.2, 0.2, bc.ACTION_DIM * bc.HIDDEN_DIM, dtype=np.float32).reshape(
        bc.ACTION_DIM,
        bc.HIDDEN_DIM,
    )
    bias = np.linspace(-0.1, 0.1, bc.ACTION_DIM, dtype=np.float32)
    trunk = np.asarray([1.0, 2.0], dtype=np.float32)
    node = helper.make_node(
        "Gemm",
        ["hidden", bc.FINAL_WEIGHT_NAME, bc.FINAL_BIAS_NAME],
        ["action"],
        transB=1,
    )
    graph = helper.make_graph(
        [node],
        "tiny-final-affine",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, bc.HIDDEN_DIM])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, bc.ACTION_DIM])],
        [
            numpy_helper.from_array(weight, name=bc.FINAL_WEIGHT_NAME),
            numpy_helper.from_array(bias, name=bc.FINAL_BIAS_NAME),
            numpy_helper.from_array(trunk, name="protected.trunk"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def _published_arrays() -> dict[str, np.ndarray]:
    path = REPO_ROOT / "artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_contract_and_preflight_bind_exact_lineage_and_absent_source_pt() -> None:
    contract = bc.load_offline_bc_contract(REPO_ROOT)
    assert contract["rows"] == {
        "total": 510,
        "reset_prefix": 10,
        "real_h10": 500,
        "q9_first": 9,
        "q9_last_action": 518,
        "fold_count": 5,
        "fold_size": 102,
        "purge_rows_each_side": 10,
    }
    assert contract["model"]["source_checkpoint_present"] is False
    assert contract["model"]["optimizer_state"] is None
    assert contract["model"]["resume_capable"] is False
    preflight = bc.preflight_offline_bc(REPO_ROOT)
    assert preflight["bootstrap_npz_sha256"] == (
        "136768fd1595265d9743d5a9e5f7ef38e431de9a57f9ff85246123a7d649f475"
    )
    assert preflight["source_checkpoint_present"] is False


def test_hash_manifest_and_array_schema_mutations_fail_closed(tmp_path: Path) -> None:
    payload = b"bound"
    path = tmp_path / "bound.bin"
    path.write_bytes(payload)
    entry = {"path": "bound.bin", "sha256": hashlib.sha256(payload).hexdigest()}
    assert bc._regular_file(tmp_path, entry, "test") == path.resolve()  # noqa: SLF001
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bc._regular_file(tmp_path, entry, "test")  # noqa: SLF001

    contract = bc.load_offline_bc_contract(REPO_ROOT)
    manifest_path = REPO_ROOT / contract["inputs"]["bootstrap_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boundaries"]["support_admitted"] = True
    with pytest.raises(ValueError, match="boundary drift"):
        bc._validate_bootstrap_manifest_binding(manifest, contract)  # noqa: SLF001

    arrays = _published_arrays()
    arrays.pop("encoder267")
    with pytest.raises(ValueError, match="array schema mismatch"):
        bc._validated_fit_arrays(arrays)  # noqa: SLF001
    arrays = _published_arrays()
    arrays["decoder994"][17, 80] += np.float32(0.01)
    with pytest.raises(ValueError, match=r"exact token64\+proprio930"):
        bc._validated_fit_arrays(arrays)  # noqa: SLF001


def test_source_decoder_abi_mutation_and_protected_trunk_drift_fail() -> None:
    contract = bc.load_offline_bc_contract(REPO_ROOT)
    source_path = REPO_ROOT / contract["inputs"]["source_decoder"]["path"]
    source = onnx.load(source_path, load_external_data=False)
    bad_abi = copy.deepcopy(source)
    bad_abi.graph.input[0].name = "wrong"
    with pytest.raises(ValueError, match="ABI drift"):
        bc._validate_source_decoder_abi(bad_abi, contract)  # noqa: SLF001

    tiny = _tiny_final_affine_model()
    initializers = {value.name: numpy_helper.to_array(value) for value in tiny.graph.initializer}
    adapted, _ = bc.export_final_affine_model(
        tiny,
        initializers[bc.FINAL_WEIGHT_NAME] + np.float32(0.01),
        initializers[bc.FINAL_BIAS_NAME] - np.float32(0.02),
    )
    bc.assert_only_final_affine_changed(tiny, adapted)
    for initializer in adapted.graph.initializer:
        if initializer.name == "protected.trunk":
            initializer.CopyFrom(
                numpy_helper.from_array(np.asarray([1.0, 3.0], dtype=np.float32), initializer.name)
            )
    with pytest.raises(ValueError, match="trunk initializer drift"):
        bc.assert_only_final_affine_changed(tiny, adapted)


def test_purged_folds_have_exact_blocks_and_no_h10_overlap() -> None:
    folds = bc.contiguous_purged_folds()
    assert len(folds) == 5
    for index, (training, validation) in enumerate(folds):
        assert np.array_equal(validation, np.arange(index * 102, (index + 1) * 102))
        assert not np.intersect1d(training, validation).size
        assert np.min(np.abs(training[:, None] - validation[None, :])) == 11
    assert folds[0][0][0] == 112
    assert folds[-1][0][-1] == 397


def test_canonical_svd_sign_and_bytes_are_repeatable() -> None:
    row = np.arange(96, dtype=np.float64)[:, None]
    column = np.arange(24, dtype=np.float64)[None, :]
    matrix = np.sin(row * 0.071 + column * 0.133) + np.cos(row * column * 0.003)
    first = bc.canonical_svd(matrix)
    second = bc.canonical_svd(matrix)
    for left, right in zip(first, second, strict=True):
        assert left.tobytes() == right.tobytes()
    vt = first[2]
    for vector in vt:
        assert vector[np.argmax(np.abs(vector))] >= 0.0
    assert np.allclose(first[0] @ np.diag(first[1]) @ first[2], matrix, atol=1.0e-12)


def test_numeric_runtime_binds_and_enforces_single_thread_same_runtime_only() -> None:
    with bc._single_thread_numeric_runtime() as binding:  # noqa: SLF001
        assert binding["determinism_scope"] == ("same_bound_runtime_only_cross_runtime_determinism_not_claimed")
        assert binding["numpy_version"] == np.__version__
        assert len(binding["openblas"]) == 2
        assert all(item["threads_during_fit"] == 1 for item in binding["openblas"])
        assert all(len(item["library_sha256"]) == 64 for item in binding["openblas"])


def test_uncentered_unscaled_scores_map_exactly_to_exported_head_for_every_fold() -> None:
    rows = np.arange(bc.TOTAL_ROWS, dtype=np.float64)[:, None]
    columns = np.arange(80, dtype=np.float64)[None, :]
    hidden = np.sin(rows * 0.021 + columns * 0.079) + 0.3 * np.cos(rows * columns * 0.0007)
    residual = np.stack(
        [
            np.sin(np.arange(bc.TOTAL_ROWS, dtype=np.float64) * 0.013 + action * 0.17)
            for action in range(bc.ACTION_DIM)
        ],
        axis=1,
    )
    for training, validation in bc.contiguous_purged_folds():
        projection = bc.fit_projection(hidden[training], minimum_rank=1)
        train_scores = projection.transform(hidden[training])
        validation_scores = projection.transform(hidden[validation])
        assert np.array_equal(train_scores, hidden[training] @ projection.basis.T)
        assert np.array_equal(validation_scores, hidden[validation] @ projection.basis.T)
        coefficient, intercept, _ = bc._ridge_residual(  # noqa: SLF001
            train_scores,
            residual[training],
            1.0,
        )
        delta_weight = coefficient @ projection.basis
        via_fit_coordinates = validation_scores @ coefficient.T + intercept
        via_exported_head = hidden[validation] @ delta_weight.T + intercept
        assert np.max(np.abs(via_fit_coordinates - via_exported_head)) <= 3.0e-14

        design = np.concatenate(
            (train_scores, np.ones((training.size, 1), dtype=np.float64)),
            axis=1,
        )
        expected = np.linalg.solve(
            design.T @ design / training.size + np.eye(design.shape[1]),
            design.T @ residual[training] / training.size,
        )
        assert np.max(np.abs(coefficient - expected[:-1].T)) <= 1.0e-13
        assert np.max(np.abs(intercept - expected[-1])) <= 1.0e-13


def test_final_affine_export_is_deterministic_and_matches_numpy() -> None:
    source = _tiny_final_affine_model()
    initializers = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    weight = initializers[bc.FINAL_WEIGHT_NAME] + np.float32(0.007)
    bias = initializers[bc.FINAL_BIAS_NAME] - np.float32(0.011)
    _, first = bc.export_final_affine_model(source, weight, bias)
    adapted, second = bc.export_final_affine_model(source, weight, bias)
    assert first == second
    full_state = bc._canonical_initializer_state_binding(adapted)  # noqa: SLF001
    head_state = bc._canonical_initializer_state_binding(  # noqa: SLF001
        adapted,
        {bc.FINAL_WEIGHT_NAME, bc.FINAL_BIAS_NAME},
    )
    assert len(full_state["state_sha256"]) == 64
    assert head_state["tensor_count"] == 2
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(first, sess_options=options, providers=["CPUExecutionProvider"])
    hidden = (np.sin(np.arange(bc.HIDDEN_DIM, dtype=np.float32) * np.float32(0.01)) * np.float32(0.1))[None]
    actual = session.run(["action"], {"hidden": hidden})[0]
    expected = hidden @ weight.T + bias
    assert np.max(np.abs(actual - expected)) <= 2.0e-6


def test_exclusive_publication_pass_and_failure_never_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "artifacts/g1_true23"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(bc, "_assert_outcome_lineage_current", lambda _request, _outcome: None)
    request = bc.OfflineBCRequest(tmp_path, Path("artifacts/g1_true23/candidate"))
    outcome = bc.OfflineBCOutcome(
        passed=True,
        report={
            "schema_version": 1,
            "kind": bc.MANIFEST_KIND,
            "classification": "offline_bc_eligible_for_closed_loop_simulator_experiment",
            "gate_issues": [],
            "eligible_for_closed_loop_simulator_experiment": True,
            "boundaries": {"deployment_ready": False},
        },
        decoder_bytes=b"deterministic-model",
    )
    decoder, manifest, body = bc.publish_offline_bc_outcome_new(request, outcome)
    assert decoder is not None and decoder.read_bytes() == b"deterministic-model"
    written = json.loads(manifest.read_text(encoding="utf-8"))
    unhashed = dict(written)
    claimed = unhashed.pop("manifest_payload_sha256")
    assert hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() == claimed
    assert body["artifact"]["publishable_decoder"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        bc.publish_offline_bc_outcome_new(request, outcome)

    failed_request = bc.OfflineBCRequest(tmp_path, Path("artifacts/g1_true23/failed"))
    failed = bc.failure_outcome(RuntimeError("gate"))
    failed_decoder, failed_manifest, failed_body = bc.publish_offline_bc_outcome_new(
        failed_request,
        failed,
    )
    assert failed_decoder is None
    assert failed_manifest.is_file()
    assert failed_body["artifact"]["publishable_decoder"] is False
    assert not failed_request.decoder_path.exists()


def test_prepublish_recheck_rejects_stale_parent_or_runtime_binding() -> None:
    lineage = bc.preflight_offline_bc(REPO_ROOT)
    with bc._single_thread_numeric_runtime() as numeric_runtime:  # noqa: SLF001
        numeric_runtime = copy.deepcopy(numeric_runtime)
    request = bc.OfflineBCRequest(
        REPO_ROOT,
        Path("artifacts/g1_true23/not_published_lineage_test"),
    )
    report = {
        "lineage": lineage,
        "export": {},
        "runtime": {
            "numeric_runtime": numeric_runtime,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
        },
        "classification": "offline_bc_diagnostic_failed_gates",
    }
    outcome = bc.OfflineBCOutcome(False, report, None)
    bc._assert_outcome_lineage_current(request, outcome)  # noqa: SLF001
    stale = copy.deepcopy(lineage)
    stale["support_contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="drift before publish"):
        bc._assert_outcome_lineage_current(  # noqa: SLF001
            request,
            bc.OfflineBCOutcome(False, {**report, "lineage": stale}, None),
        )


def test_cli_has_no_fit_seed_lambda_threshold_device_or_output_override() -> None:
    parser = _parser()
    args = parser.parse_args(["fit", "--output-prefix", "artifacts/g1_true23/candidate"])
    assert args.command == "fit"
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["fit"]._actions  # noqa: SLF001
        for option in action.option_strings
    }
    for forbidden in ("--seed", "--lambda", "--threshold", "--device", "--decoder"):
        assert forbidden not in option_strings
