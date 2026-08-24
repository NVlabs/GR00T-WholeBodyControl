from pathlib import Path

import numpy as np

from gear_sonic.scripts.prepare_g1_moves_true23_csv import RETAINED_29_INDICES, convert


def test_convert_keeps_exact_native_joint_subset(tmp_path: Path) -> None:
    values = np.zeros((2, 36), dtype=np.float64)
    values[:, 6] = 1.0
    values[:, 7:] = np.arange(29, dtype=np.float64)
    source, destination = tmp_path / "source.csv", tmp_path / "true23.csv"
    np.savetxt(source, values, delimiter=",")
    convert(source, destination)
    result = np.loadtxt(destination, delimiter=",")
    assert result.shape == (2, 30)
    assert np.array_equal(result[0, 7:], np.asarray(RETAINED_29_INDICES))
