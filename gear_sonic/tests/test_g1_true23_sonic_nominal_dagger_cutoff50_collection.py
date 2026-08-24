from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_cutoff50_collection as collection,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _arrays() -> dict[str, np.ndarray]:
    zero23 = np.zeros((collection.TOTAL_ROWS, 23), dtype=np.float32)
    token = np.zeros((collection.TOTAL_ROWS, 64), dtype=np.float32)
    policy = np.zeros((collection.TOTAL_ROWS, 930), dtype=np.float32)
    return {
        "encoder267": np.zeros((collection.TOTAL_ROWS, 267), dtype=np.float32),
        "token64": token,
        "policy930": policy,
        "decoder994": np.concatenate((token, policy), axis=1),
        "student_raw_native23": zero23.copy(),
        "selected_observation124": np.zeros((collection.TOTAL_ROWS, 124), dtype=np.float32),
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


def test_contract_and_recovery_scope_are_exact_and_restored() -> None:
    contract = collection.load_contract(REPO_ROOT)
    assert contract["runtime"]["mode"] == "cutoff50"
    old_path = recovery.RECOVERY_CONTRACT_RELATIVE_PATH
    old_candidate = recovery.CURRENT_CANDIDATE_DECODER_SHA256
    with collection._recovery_scope(REPO_ROOT):  # noqa: SLF001
        assert recovery.RECOVERY_CONTRACT_RELATIVE_PATH == collection.RECOVERY_CONTRACT_RELATIVE_PATH
        assert recovery.CURRENT_CANDIDATE_DECODER_SHA256 == collection.CANDIDATE_DECODER_SHA256
    assert recovery.RECOVERY_CONTRACT_RELATIVE_PATH == old_path
    assert recovery.CURRENT_CANDIDATE_DECODER_SHA256 == old_candidate


def test_array_schema_and_partition_fail_closed() -> None:
    arrays = _arrays()
    collection._validate_arrays(arrays)  # noqa: SLF001
    bad = {name: value.copy() for name, value in arrays.items()}
    bad["controller_code"][49] = 1
    with pytest.raises(ValueError):
        collection._validate_arrays(bad)  # noqa: SLF001
    bad = {name: value.copy() for name, value in arrays.items()}
    bad["decoder994"][0, 0] = 1.0
    with pytest.raises(ValueError):
        collection._validate_arrays(bad)  # noqa: SLF001


def test_publication_is_atomic_exclusive_and_has_no_support_claim(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    evidence = root / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    config = root / collection.CONTRACT_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes((REPO_ROOT / collection.CONTRACT_RELATIVE_PATH).read_bytes())
    request = collection.CollectionRequest(root, Path("artifacts/g1_true23/cutoff50"))
    npz, manifest, body = collection.publish(request, _arrays(), {"recovery_report": {"passed": True}})
    assert npz.is_file() and manifest.is_file()
    assert body["rows"] == {
        "total": 510,
        "student_on_policy_shadow_label_rows": 50,
        "teacher_actuated_recovery_rows": 460,
    }
    assert body["boundaries"]["support_qualified"] is False
    assert body["boundaries"]["hardware_authorized"] is False
    with pytest.raises(FileExistsError):
        collection.publish(request, _arrays(), {})
