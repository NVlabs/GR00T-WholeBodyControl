import numpy as np
from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

def test_disabling_teleop_resets_command():
    policy = G1GearWbcPolicy.__new__(G1GearWbcPolicy)

    policy.config = {"cmd_init": np.zeros(3, dtype=np.float32)}

    policy.use_teleop_policy_cmd = True
    policy.cmd = np.array([0.3, 0.0, 0.0])

    policy.set_use_teleop_policy_cmd(False)

    assert not policy.use_teleop_policy_cmd

    np.testing.assert_array_equal(policy.cmd, policy.config["cmd_init"])
