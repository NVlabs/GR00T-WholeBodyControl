from pathlib import Path
import pickle

import numpy as np
import pytest

from decoupled_wbc.control.policy.interpolation_policy import (
    InterpolationPolicy,
)


def get_test_data_path(filename: str) -> str:
    """Get the absolute path to a test data file."""
    test_dir = Path(__file__).parent
    return str(test_dir / ".." / ".." / ".." / "replay_data" / filename)


@pytest.fixture
def logged_data():
    """Load the logged data from file."""
    data_path = get_test_data_path("interpolation_data.pkl")
    with open(data_path, "rb") as f:
        return pickle.load(f)


def test_replay_logged_data(logged_data):
    """Test that the wrapper produces the same pose commands as logged data."""
    init_args = logged_data["init_args"]
    interp = InterpolationPolicy(
        init_time=init_args["curr_t"],
        init_values={"target_pose": init_args["curr_pose"]},
        max_change_rate=np.inf,
    )

    # Test all data points including the first one
    for c in logged_data["calls"]:
        # Get the action from wrapper
        if c["type"] == "get_action":
            action = interp.get_action(**c["args"])
            expected_action = c["result"]
            np.testing.assert_allclose(
                action["target_pose"], expected_action["q"], rtol=1e-9, atol=1e-9
            )
            # print(action, expected_action)

        else:
            interp.set_goal(**c["args"])
