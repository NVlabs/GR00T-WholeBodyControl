import numpy as np

from gear_sonic.g1_upper_body_telemetry.protocol import decode_sample, encode_sample
from gear_sonic.g1_upper_body_telemetry.sampler import TelemetrySampler


def _snapshot(dq_cmd: float = 2.0) -> dict:
    zeros = np.zeros(17, dtype=np.float32)
    return {
        "state": {
            "q": zeros + 2.0,
            "dq": zeros + 3.0,
            "ddq": zeros + 4.0,
            "tau_est": zeros + 20.0,
        },
        "command": {
            "q": zeros + 1.0,
            "dq": zeros + dq_cmd,
            "tau": zeros + 5.0,
            "kp": zeros + 2.0,
            "kd": zeros + 3.0,
        },
        "state_time_ns": 900_000_000,
        "command_time_ns": 950_000_000,
    }


def test_sampler_computes_ddq_pd_torque_and_residuals():
    sampler = TelemetrySampler(50.0)
    first = sampler.build(_snapshot(), now_ns=1_000_000_000)
    second = sampler.build(_snapshot(2.2), now_ns=1_020_000_000)

    assert not first["ddq_cmd_valid"]
    assert second["ddq_cmd_valid"]
    np.testing.assert_allclose(second["ddq_cmd"], 10.0, rtol=1e-5)
    np.testing.assert_allclose(second["tau_cmd_pd"], 0.6, rtol=1e-5)
    np.testing.assert_allclose(second["tau_residual"], 19.4, rtol=1e-5)
    np.testing.assert_allclose(second["ddq_residual"], -6.0, rtol=1e-5)


def test_msgpack_round_trip_preserves_arrays():
    sample = TelemetrySampler(50.0).build(_snapshot(), now_ns=1_000_000_000)
    restored = decode_sample(encode_sample(sample))

    assert restored["sequence_id"] == 0
    assert restored["q_cmd"].shape == (17,)
    np.testing.assert_array_equal(restored["tau_est"], sample["tau_est"])
