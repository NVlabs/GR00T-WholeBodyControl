from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_pico_library_projection import reachable_hardware_bounds


def test_reachable_hardware_bounds_match_exact_safe_transform() -> None:
    low, high = reachable_hardware_bounds(9.0)
    _, positive = safe_target_transform_numpy(np.full(23, np.float32(9.0)))
    _, negative = safe_target_transform_numpy(np.full(23, np.float32(-9.0)))
    np.testing.assert_array_equal(low, np.minimum(negative, positive).astype(np.float64))
    np.testing.assert_array_equal(high, np.maximum(negative, positive).astype(np.float64))
    assert low.shape == high.shape == (23,)
    assert np.all(low < high)


@pytest.mark.parametrize("value", [0.0, 10.0, -1.0, 9])
def test_reachable_hardware_bounds_rejects_invalid_bound(value: object) -> None:
    with pytest.raises(ValueError, match="raw_abs"):
        reachable_hardware_bounds(value)  # type: ignore[arg-type]
