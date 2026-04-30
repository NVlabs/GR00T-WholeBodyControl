"""Brainco DDS controller, ported from xr_teleoperate.

Pulls (25, 3) hand-keypoint arrays out of two shared multiprocessing arrays
(populated by BraincoBridge from xrobotoolkit hand tracking), runs
dex_retargeting, normalises the resulting joint angles to [0, 1] and publishes
them on the brainco DDS topics.

The shared input arrays follow the same convention as
xr_teleoperate/teleop/robot_control/robot_hand_brainco.py: 75 doubles per hand,
reshaped to (25, 3) as a hand-keypoint cloud in the unitree-hand frame.
"""

import logging
import threading
import time
from enum import IntEnum
from multiprocessing import Array, Process

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from decoupled_wbc.control.teleop.device.pico.brainco.hand_retargeting import (
    BraincoHandRetargeting,
)

logger = logging.getLogger(__name__)

BRAINCO_NUM_MOTORS = 6
TOPIC_LEFT_CMD = "rt/brainco/left/cmd"
TOPIC_LEFT_STATE = "rt/brainco/left/state"
TOPIC_RIGHT_CMD = "rt/brainco/right/cmd"
TOPIC_RIGHT_STATE = "rt/brainco/right/state"


# Driver motor sequence per
# https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
class BraincoLeftJointIndex(IntEnum):
    THUMB = 0
    THUMB_AUX = 1
    INDEX = 2
    MIDDLE = 3
    RING = 4
    PINKY = 5


class BraincoRightJointIndex(IntEnum):
    THUMB = 0
    THUMB_AUX = 1
    INDEX = 2
    MIDDLE = 3
    RING = 4
    PINKY = 5


def _normalize(val, min_val, max_val):
    return 1.0 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)


class BraincoController:
    """Spawns a control process that retargets hand-keypoint clouds and
    publishes brainco DDS commands at a fixed rate.

    Parameters
    ----------
    left_hand_array, right_hand_array : multiprocessing.Array('d', 75)
        Flattened (25, 3) hand-keypoint clouds for left/right hand in the
        unitree-hand frame. Written by the bridge.
    fps : float
        Control loop rate.
    state_timeout_s : float
        How long to wait for the brainco DDS state topics to publish before
        raising. 0 disables (legacy "wait forever" behavior). Default 5 s.
    warmup_frames : int
        Number of consecutive valid input frames the control loop must see
        before it starts publishing retargeted joint angles. Until then it
        publishes the safe "open hand" pose (q=0 for all motors). This guards
        against junk readings during xrobotoolkit warmup or transient
        is_active=0 -> 1 transitions.
    """

    def __init__(
        self,
        left_hand_array: Array,
        right_hand_array: Array,
        dual_hand_state_array: Array | None = None,
        dual_hand_action_array: Array | None = None,
        dual_hand_data_lock=None,
        fps: float = 100.0,
        state_timeout_s: float = 5.0,
        warmup_frames: int = 10,
    ):
        logger.info("Initialising BraincoController")
        self.fps = fps
        self.warmup_frames = max(0, int(warmup_frames))
        self._sub_ready = False

        self._retargeting = BraincoHandRetargeting()

        self._left_pub = ChannelPublisher(TOPIC_LEFT_CMD, MotorCmds_)
        self._left_pub.Init()
        self._right_pub = ChannelPublisher(TOPIC_RIGHT_CMD, MotorCmds_)
        self._right_pub.Init()
        self._left_sub = ChannelSubscriber(TOPIC_LEFT_STATE, MotorStates_)
        self._left_sub.Init()
        self._right_sub = ChannelSubscriber(TOPIC_RIGHT_STATE, MotorStates_)
        self._right_sub.Init()

        self._left_state_array = Array("d", BRAINCO_NUM_MOTORS, lock=True)
        self._right_state_array = Array("d", BRAINCO_NUM_MOTORS, lock=True)

        self._state_thread = threading.Thread(target=self._subscribe_state, daemon=True)
        self._state_thread.start()

        deadline = time.time() + state_timeout_s if state_timeout_s > 0 else None
        last_warn = 0.0
        while not self._sub_ready:
            time.sleep(0.1)
            now = time.time()
            if deadline is not None and now > deadline:
                raise TimeoutError(
                    f"BraincoController: no DDS state on {TOPIC_LEFT_STATE!r} or "
                    f"{TOPIC_RIGHT_STATE!r} within {state_timeout_s:.1f}s. "
                    "Confirm brainco hands are powered, on the same DDS network, "
                    "and that ChannelFactoryInitialize was called with the right "
                    "interface."
                )
            if now - last_warn > 1.0:
                logger.warning("[BraincoController] waiting for DDS state...")
                last_warn = now
        logger.info("[BraincoController] DDS state ready")

        self._proc = Process(
            target=self._control_loop,
            args=(
                left_hand_array,
                right_hand_array,
                self._left_state_array,
                self._right_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
            daemon=True,
        )
        self._proc.start()
        logger.info("BraincoController control process started")

    def _subscribe_state(self):
        while True:
            left_msg = self._left_sub.Read()
            right_msg = self._right_sub.Read()
            self._sub_ready = True
            if left_msg is not None and right_msg is not None:
                for idx, jid in enumerate(BraincoLeftJointIndex):
                    self._left_state_array[idx] = left_msg.states[jid].q
                for idx, jid in enumerate(BraincoRightJointIndex):
                    self._right_state_array[idx] = right_msg.states[jid].q
            time.sleep(0.002)

    def _publish(self, left_msg, right_msg, left_q, right_q):
        for idx, jid in enumerate(BraincoLeftJointIndex):
            left_msg.cmds[jid].q = float(left_q[idx])
        for idx, jid in enumerate(BraincoRightJointIndex):
            right_msg.cmds[jid].q = float(right_q[idx])
        self._left_pub.Write(left_msg)
        self._right_pub.Write(right_msg)

    def _control_loop(
        self,
        left_hand_array,
        right_hand_array,
        left_state_array,
        right_state_array,
        lock,
        state_out,
        action_out,
    ):
        left_msg = MotorCmds_()
        left_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(BraincoLeftJointIndex))
        ]
        right_msg = MotorCmds_()
        right_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(BraincoRightJointIndex))
        ]
        for idx, jid in enumerate(BraincoLeftJointIndex):
            left_msg.cmds[jid].q = 0.0
            left_msg.cmds[jid].dq = 1.0
        for idx, jid in enumerate(BraincoRightJointIndex):
            right_msg.cmds[jid].q = 0.0
            right_msg.cmds[jid].dq = 1.0

        left_q_target = np.zeros(BRAINCO_NUM_MOTORS)
        right_q_target = np.zeros(BRAINCO_NUM_MOTORS)

        consecutive_valid = 0
        warmup_logged = False

        try:
            while True:
                start = time.time()

                with left_hand_array.get_lock():
                    left_pts = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_pts = np.array(right_hand_array[:]).reshape(25, 3).copy()

                state_data = np.concatenate(
                    (np.array(left_state_array[:]), np.array(right_state_array[:]))
                )

                # The bridge writes all-zeros into both arrays whenever
                # xrobotoolkit reports the hand inactive. Any non-zero entry
                # means we got real data this tick.
                left_valid = not np.all(left_pts == 0.0) and np.all(np.isfinite(left_pts))
                right_valid = not np.all(right_pts == 0.0) and np.all(np.isfinite(right_pts))

                if left_valid and right_valid:
                    consecutive_valid += 1
                else:
                    if warmup_logged:
                        logger.warning(
                            "[BraincoController] tracking lost; reverting to open hand"
                        )
                        warmup_logged = False
                    consecutive_valid = 0

                if consecutive_valid >= self.warmup_frames:
                    if not warmup_logged:
                        logger.info(
                            "[BraincoController] warmup complete after %d valid frames; "
                            "publishing retargeted commands",
                            consecutive_valid,
                        )
                        warmup_logged = True

                    left_indices = self._retargeting.left_indices
                    right_indices = self._retargeting.right_indices
                    ref_left = left_pts[left_indices[1, :]] - left_pts[left_indices[0, :]]
                    ref_right = right_pts[right_indices[1, :]] - right_pts[right_indices[0, :]]

                    left_q_target = self._retargeting.left_retargeting.retarget(ref_left)[
                        self._retargeting.left_dex_to_hardware
                    ]
                    right_q_target = self._retargeting.right_retargeting.retarget(ref_right)[
                        self._retargeting.right_dex_to_hardware
                    ]

                    # Brainco API expects [0, 1] (0 open, 1 closed). Joint
                    # ranges per the official docs: idx0 0~1.52, idx1 0~1.05,
                    # idx2..5 0~1.47.
                    for idx in range(BRAINCO_NUM_MOTORS):
                        if idx == 0:
                            left_q_target[idx] = _normalize(left_q_target[idx], 0.0, 1.52)
                            right_q_target[idx] = _normalize(right_q_target[idx], 0.0, 1.52)
                        elif idx == 1:
                            left_q_target[idx] = _normalize(left_q_target[idx], 0.0, 1.05)
                            right_q_target[idx] = _normalize(right_q_target[idx], 0.0, 1.05)
                        else:
                            left_q_target[idx] = _normalize(left_q_target[idx], 0.0, 1.47)
                            right_q_target[idx] = _normalize(right_q_target[idx], 0.0, 1.47)
                else:
                    # Warmup or tracking-loss: command the safe "open hand"
                    # pose (q = 0) regardless of what the previous tick set.
                    left_q_target = np.zeros(BRAINCO_NUM_MOTORS)
                    right_q_target = np.zeros(BRAINCO_NUM_MOTORS)

                action_data = np.concatenate((left_q_target, right_q_target))
                if state_out is not None and action_out is not None:
                    with lock:
                        state_out[:] = state_data
                        action_out[:] = action_data

                self._publish(left_msg, right_msg, left_q_target, right_q_target)

                elapsed = time.time() - start
                time.sleep(max(0.0, (1.0 / self.fps) - elapsed))
        finally:
            logger.info("BraincoController control loop exited")
