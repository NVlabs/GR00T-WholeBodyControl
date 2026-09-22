"""G1 upper-body DDS feedback transport."""

from copy import deepcopy
import threading
import time

import numpy as np

from .constants import G1_UPPER_BODY_JOINTS
from .runtime import (
    ChannelFactoryInitialize, ChannelSubscriber, G1LowState_,
    _G1_STATE_DDS_IMPORT_ERROR,
)

class G1UpperBodyStateSubscriber:
    """Keep a thread-safe, read-only snapshot of G1 upper-body DDS feedback."""

    def __init__(self):
        if _G1_STATE_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py with unitree_hg LowState_ is required "
                "to record G1 upper-body state"
            ) from _G1_STATE_DDS_IMPORT_ERROR

        self._lock = threading.Lock()
        self._lowstate = None
        self._lowstate_time = None
        self._subscribers = [ChannelSubscriber("rt/lowstate", G1LowState_)]
        self._subscribers[0].Init(self._lowstate_callback, 10)
        print("[G1State] DDS subscriber ready: rt/lowstate")

    def _lowstate_callback(self, msg) -> None:
        with self._lock:
            self._lowstate = deepcopy(msg)
            self._lowstate_time = time.monotonic()

    def snapshot_fields(self) -> dict[str, np.ndarray]:
        """Return arrays suitable for pack_pose_message/np.savez_compressed."""
        with self._lock:
            lowstate = self._lowstate
            lowstate_time = self._lowstate_time

            if lowstate is None or lowstate_time is None:
                return {"g1_upper_body_state": np.zeros(70, dtype=np.float32)}

            motors = [lowstate.motor_state[index] for index, _ in G1_UPPER_BODY_JOINTS]
            now = time.monotonic()
            # Layout: valid, age_ms, q[17], dq[17], ddq[17], tau_est[17].
            # The exporter expands this transport array into four named fields.
            float_values = [1.0, (now - lowstate_time) * 1000.0]
            for attribute in ("q", "dq", "ddq", "tau_est"):
                float_values.extend(float(getattr(motor, attribute)) for motor in motors)
            return {"g1_upper_body_state": np.asarray(float_values, dtype=np.float32)}

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass


def _create_g1_state_subscriber(
    domain_id: int,
    network_interface: str | None,
    dds_already_initialized: bool,
) -> G1UpperBodyStateSubscriber:
    if not dds_already_initialized:
        if ChannelFactoryInitialize is None:
            raise ImportError("unitree_sdk2py ChannelFactoryInitialize is unavailable")
        if network_interface:
            ChannelFactoryInitialize(domain_id, network_interface)
        else:
            ChannelFactoryInitialize(domain_id)
    return G1UpperBodyStateSubscriber()
