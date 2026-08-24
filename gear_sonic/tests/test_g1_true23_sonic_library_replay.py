from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

ROOT = Path(__file__).resolve().parents[2]
MOTION = ROOT / "artifacts/g1_true23/sonic_library_true23_hand_v1/hand_crawling.true23.npz"


def _motion() -> dict[str, np.ndarray]:
    with np.load(MOTION, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def test_true23_library_motion_schema() -> None:
    motion = _motion()
    assert validate_library_motion(motion) == 2246
    assert motion["joint_pos"].shape[1] == 23
    assert motion["body_pos_w"].shape[1] == 24


def test_true23_library_motion_rejects_wrong_dof_and_nonfinite() -> None:
    motion = _motion()
    bad = copy.deepcopy(motion)
    bad["joint_pos"] = bad["joint_pos"][:, :22]
    with pytest.raises(ValueError, match="joint_pos shape"):
        validate_library_motion(bad)
    bad = copy.deepcopy(motion)
    bad["joint_pos"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_library_motion(bad)
