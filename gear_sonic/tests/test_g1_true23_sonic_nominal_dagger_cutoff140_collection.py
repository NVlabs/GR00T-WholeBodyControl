from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_cutoff50_collection as base,
    g1_true23_sonic_nominal_dagger_cutoff140_collection as collection,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _arrays() -> dict[str, np.ndarray]:
    zero23 = np.zeros((510, 23), dtype=np.float32)
    token = np.zeros((510, 64), dtype=np.float32)
    policy = np.zeros((510, 930), dtype=np.float32)
    return {
        "encoder267": np.zeros((510, 267), dtype=np.float32),
        "token64": token,
        "policy930": policy,
        "decoder994": np.concatenate((token, policy), axis=1),
        "student_raw_native23": zero23.copy(),
        "selected_observation124": np.zeros((510, 124), dtype=np.float32),
        "teacher_raw_hardware23": zero23.copy(),
        "teacher_label_raw_native23": zero23.copy(),
        "executed_raw_native23": zero23.copy(),
        "executed_safe_native23": zero23.copy(),
        "executed_final_target_hardware23": zero23.copy(),
        "controller_code": np.concatenate(
            (
                np.zeros(collection.STUDENT_ROWS, dtype=np.int8),
                np.ones(collection.TEACHER_ROWS, dtype=np.int8),
            )
        ),
        "q9": np.arange(9, 519, dtype=np.int64),
    }


def test_contract_and_preflight_bind_failed_frontier() -> None:
    contract = collection.load_contract(REPO_ROOT)
    assert contract["runtime"]["mode"] == "cutoff140"
    assert contract["runtime"]["student_controlled_transitions"] == 140
    assert contract["runtime"]["teacher_controlled_transitions"] == 370
    request = base.CollectionRequest(REPO_ROOT, Path("artifacts/g1_true23/unused-cutoff140-test"))
    receipt = collection.preflight(request)
    assert receipt["ready"] is True
    assert receipt["simulator_constructed"] is False
    assert receipt["teacher_labels_admitted"] == 0


def test_scopes_patch_exact_values_and_restore_globals() -> None:
    base_before = (base.STUDENT_ROWS, base.TEACHER_ROWS, base.CONTRACT_RELATIVE_PATH, base.adapter)
    recovery_before = (
        recovery.CURRENT_CANDIDATE_DECODER_SHA256,
        recovery.RECOVERY_CONTRACT_SHA256,
        recovery.load_recovery_contract,
    )
    with collection._base_patch(REPO_ROOT):  # noqa: SLF001
        assert base.STUDENT_ROWS == 140
        assert base.TEACHER_ROWS == 370
        assert base.CONTRACT_RELATIVE_PATH == collection.CONTRACT_RELATIVE_PATH
        with base._recovery_scope(REPO_ROOT):  # noqa: SLF001
            current = recovery.load_recovery_contract(REPO_ROOT)
            assert current["candidate"]["decoder_sha256"] == collection.adapter.CANDIDATE_DECODER_SHA256
    assert (base.STUDENT_ROWS, base.TEACHER_ROWS, base.CONTRACT_RELATIVE_PATH, base.adapter) == base_before
    assert (
        recovery.CURRENT_CANDIDATE_DECODER_SHA256,
        recovery.RECOVERY_CONTRACT_SHA256,
        recovery.load_recovery_contract,
    ) == recovery_before


def test_array_partition_and_publication_fail_closed(tmp_path: Path) -> None:
    arrays = _arrays()
    with collection._base_patch(REPO_ROOT):  # noqa: SLF001
        base._validate_arrays(arrays)  # noqa: SLF001
        bad = {name: value.copy() for name, value in arrays.items()}
        bad["controller_code"][139] = 1
        with pytest.raises(ValueError):
            base._validate_arrays(bad)  # noqa: SLF001

    root = tmp_path / "repo"
    (root / "artifacts" / "g1_true23").mkdir(parents=True)
    config = root / collection.CONTRACT_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes((REPO_ROOT / collection.CONTRACT_RELATIVE_PATH).read_bytes())
    request = base.CollectionRequest(root, Path("artifacts/g1_true23/cutoff140"))
    with collection._base_patch(root):  # noqa: SLF001
        npz, manifest, body = base.publish(request, arrays, {"recovery_report": {"passed": True}})
    assert npz.is_file() and manifest.is_file()
    assert body["rows"] == {
        "total": 510,
        "student_on_policy_shadow_label_rows": 140,
        "teacher_actuated_recovery_rows": 370,
    }
    assert body["boundaries"]["support_qualified"] is False
    assert body["boundaries"]["hardware_authorized"] is False
    with collection._base_patch(root):  # noqa: SLF001
        with pytest.raises(FileExistsError):
            base.publish(request, arrays, {})
