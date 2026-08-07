import time
import sys

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

MOTOR_NUM_HAND = 7

Kp = [0.5] * MOTOR_NUM_HAND
Kd = [0.1] * MOTOR_NUM_HAND

Kp[0] = 2.0

maxTorqueLimits_left = [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0]
minTorqueLimits_left = [-1.05, -0.72, 0.0, -1.57, -1.75, -1.57, -1.75]
maxTorqueLimits_right = [1.05, 0.74, 0.0, 1.57, 1.75, 1.57, 1.75]
minTorqueLimits_right = [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0]


def make_hand_mode(motor_index):
    status = 0x01
    timeout = 0x01
    mode = (motor_index & 0x0F)
    mode |= (status << 4)  # bits [4..6]
    mode |= (timeout << 7)  # bit 7
    return mode


class HandControl:
    def __init__(self, network_interface="enp36s0f1"):
        ChannelFactoryInitialize(0, network_interface)

        self.left_cmd_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.right_cmd_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)

        self.left_state_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.right_state_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)

        self.left_cmd_pub.Init()
        self.right_cmd_pub.Init()

        self.left_state_sub.Init(self.left_state_handler, 10)
        self.right_state_sub.Init(self.right_state_handler, 10)

        self.left_state = None
        self.right_state = None

        self.crc = CRC()

        # Control loop timing
        self.control_dt = 0.01  # 10 ms
        self.time_ = 0.0
        self.stage_time = 3.0  # 3s per stage

        # To let the code run once we start:
        self.run_flag = False

        self.counter = 0

    def left_state_handler(self, msg: HandState_):
        self.left_state = msg

        # self.counter +=1
        # if (self.counter % 1000 == 0) :
        #     self.counter = 0
        #     print('Left hand state:')
        #     for i in range(MOTOR_NUM_HAND):
        #         print(180/np.pi*self.left_state.motor_state[i].q)

    def right_state_handler(self, msg: HandState_):
        self.right_state = msg

        self.counter += 1
        if (self.counter % 1000 == 0):
            self.counter = 0
            print('Right hand state:')
            for i in range(MOTOR_NUM_HAND):
                print(180 / np.pi * self.right_state.motor_state[i].q)

    def start(self):
        """
        Kick off the main control loop thread.
        """
        self.run_flag = True
        self.control_thread = RecurrentThread(interval=self.control_dt,
                                              target=self.hand_control_loop,
                                              name="HandControlLoop")
        self.control_thread.Start()

    def hand_control_loop(self):
        """
        This gets called at a fixed rate (every self.control_dt seconds).
        We'll demonstrate 3 stages of motion:
          1) Move from current position to 'zero' (or some nominal) in 3s
          2) Sinusoidal motion in stage 2 (3s)
          3) Another motion in stage 3 (final)
        """
        if not self.run_flag:
            return

        self.time_ += self.control_dt
        t = self.time_
        cmd_left = unitree_hg_msg_dds__HandCmd_()
        cmd_right = unitree_hg_msg_dds__HandCmd_()

        # cmd_left.motor_cmd.resize(MOTOR_NUM_HAND)
        # cmd_right.motor_cmd.resize(MOTOR_NUM_HAND)

        # Prepare stage times
        stage1_end = self.stage_time
        stage2_end = self.stage_time * 2.0

        # We'll fetch current positions for left and right if available
        # so we can blend from actual state to zero. 
        # If we haven't gotten a state yet, default to 0.
        left_q_now = [0.0] * MOTOR_NUM_HAND
        right_q_now = [0.0] * MOTOR_NUM_HAND

        if self.left_state is not None:
            for i in range(MOTOR_NUM_HAND):
                left_q_now[i] = self.left_state.motor_state[i].q

        if self.right_state is not None:
            for i in range(MOTOR_NUM_HAND):
                right_q_now[i] = self.right_state.motor_state[i].q

        left_q_desired = np.deg2rad([50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        right_q_desired = np.deg2rad([50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Decide the desired position for each stage:
        if t < stage1_end:
            # Stage 1: Move from the current joint positions to zero in [0..3s]
            ratio = np.clip(t / stage1_end, 0.0, 1.0)
            # Simple linear blend: final = (1-ratio)*initial + ratio*0
            left_q_des = np.zeros(MOTOR_NUM_HAND)
            right_q_des = np.zeros(MOTOR_NUM_HAND)
            for i in range(MOTOR_NUM_HAND):
                left_q_des[i] = (1.0 - ratio) * (left_q_now[i] - left_q_desired[i]) + left_q_desired[i]
                right_q_des[i] = (1.0 - ratio) * (right_q_now[i] - right_q_desired[i]) + right_q_desired[i]

        else:
            # Stage 2: Some sinusoidal wave
            # We'll wave only the first 2 or 3 motors, just as a demo
            dt2 = t - stage1_end
            freq = 1.0  # 1 Hz
            amp = 0.3  # ~ 0.3 rad amplitude

            left_q_des = left_q_desired
            right_q_des = right_q_desired

            # E.g. wave motor 0 and 1:
            left_q_des[0] += amp * np.sin(2 * np.pi * freq * dt2)
            left_q_des[1] += amp * (1 - np.cos(2 * np.pi * freq * dt2))
            left_q_des[2] += amp * (1 - np.cos(2 * np.pi * freq * dt2))
            right_q_des[0] += amp * np.sin(2 * np.pi * freq * dt2)
            right_q_des[1] += amp * (np.cos(2 * np.pi * freq * dt2) - 1)
            right_q_des[2] += amp * (np.cos(2 * np.pi * freq * dt2) - 1)

            freqA = 2.0
            freqB = 2.0
            ampA = 0.2
            ampB = 0.4

            left_q_des[3] += ampA * (np.cos(2 * np.pi * freqA * dt2) - 1)
            left_q_des[4] += ampB * (np.cos(2 * np.pi * freqB * dt2) - 1)
            left_q_des[5] += ampA * (np.cos(2 * np.pi * freqA * dt2) - 1)
            left_q_des[6] += ampB * (np.cos(2 * np.pi * freqB * dt2) - 1)
            right_q_des[3] += ampA * (1 - np.cos(2 * np.pi * freqA * dt2))
            right_q_des[4] += ampB * (1 - np.cos(2 * np.pi * freqB * dt2))
            right_q_des[5] += ampA * (1 - np.cos(2 * np.pi * freqA * dt2))
            right_q_des[6] += ampB * (1 - np.cos(2 * np.pi * freqB * dt2))

        # Fill in the commands
        for i in range(MOTOR_NUM_HAND):
            # Build the bitfield mode (see your C++ example)
            mode_val = make_hand_mode(i)

            # Left
            cmd_left.motor_cmd[i].mode = mode_val
            cmd_left.motor_cmd[i].q = left_q_des[i]
            cmd_left.motor_cmd[i].dq = 0.0
            cmd_left.motor_cmd[i].tau = 0.0
            cmd_left.motor_cmd[i].kp = Kp[i]
            cmd_left.motor_cmd[i].kd = Kd[i]

            # Right
            cmd_right.motor_cmd[i].mode = mode_val
            cmd_right.motor_cmd[i].q = right_q_des[i]
            cmd_right.motor_cmd[i].dq = 0.0
            cmd_right.motor_cmd[i].tau = 0.0
            cmd_right.motor_cmd[i].kp = Kp[i]
            cmd_right.motor_cmd[i].kd = Kd[i]

        # Compute CRC if your firmware requires it
        # cmd_left.crc  = self.crc.Crc(cmd_left)
        # cmd_right.crc = self.crc.Crc(cmd_right)

        # Publish
        self.left_cmd_pub.Write(cmd_left)
        self.right_cmd_pub.Write(cmd_right)


if __name__ == "__main__":
    print("WARNING: Make sure your robot’s hands can move freely before running.")
    input("Press Enter to continue...")

    # Optionally pass a specific interface name, e.g. "enp37s0f0" or "eth0"
    if len(sys.argv) > 1:
        net_if = sys.argv[1]
    else:
        net_if = "enp36s0f1"

    # Create and start
    hand_control = HandControl(network_interface=net_if)
    hand_control.start()

    # Just wait
    while True:
        time.sleep(1)
