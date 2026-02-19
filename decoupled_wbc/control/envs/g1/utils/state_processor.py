import time

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    HandState_,
    IMUState_,
    LowState_ as LowState_hg,
    OdoState_,
)


class BodyStateProcessor:
    def __init__(self, config):
        self.config = config

        # Enter debug mode for real robot
        if self.config["ENV_TYPE"] == "real":
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()

            status, result = msc.CheckMode()
            print(status, result)
            while result["name"]:
                msc.ReleaseMode()
                status, result = msc.CheckMode()
                print(status, result)
                time.sleep(1)

        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(None, 0)
            self.robot_lowstate_subscriber.Init(None, 0)
        elif (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
        ):
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(None, 0)

            self.secondary_imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
            self.secondary_imu_subscriber.Init(None, 0)

            # Subscribe to odo state (only available in simulation)
            if self.config["ENV_TYPE"] == "sim":
                self.odo_state_subscriber = ChannelSubscriber("rt/odostate", OdoState_)
                self.odo_state_subscriber.Init(None, 0)
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported")

        self.num_dof = self.config["NUM_JOINTS"]
        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.ddq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(3 + 3 + self.num_dof)
        self.torso_quat = np.zeros(4)
        self.torso_ang_vel = np.zeros(3)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)
        self.robot_low_state = None
        self.secondary_imu_state = None
        self.odo_state = None

    def _prepare_low_state(self) -> np.ndarray:
        self.robot_low_state = self.robot_lowstate_subscriber.Read()
        self.secondary_imu_state = self.secondary_imu_subscriber.Read()

        if not self.robot_low_state:
            print("No low state received")
            return
        imu_state = self.robot_low_state.imu_state

        # Use odo_state for position and velocity if available, otherwise set to zero
        if self.config["ENV_TYPE"] == "sim":
            self.odo_state = self.odo_state_subscriber.Read()
            self.q[0:3] = self.odo_state.position
            self.dq[0:3] = self.odo_state.linear_velocity
        else:
            self.q[0:3] = [0.0, 0.0, 0.0]
            self.dq[0:3] = [0.0, 0.0, 0.0]

        self.q[3:7] = imu_state.quaternion  # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        self.ddq[0:3] = imu_state.accelerometer
        unitree_joint_state = self.robot_low_state.motor_state
        self.torso_quat = self.secondary_imu_state.quaternion
        self.torso_ang_vel = self.secondary_imu_state.gyroscope

        for i in range(self.num_dof):
            self.q[7 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].q
            self.dq[6 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].dq
            self.tau_est[6 + i] = unitree_joint_state[self.config["JOINT2MOTOR"][i]].tau_est

        robot_state_data = np.concatenate(
            [self.q, self.dq, self.tau_est, self.ddq, self.torso_quat, self.torso_ang_vel], axis=0
        ).reshape(1, -1)
        # (7 + 29) + (6 + 29) + (6 + 29) + (6 + 29) = 141 dim

        return robot_state_data


class HandStateProcessor:
    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        if self.is_left:
            self.state_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        else:
            self.state_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)

        self.state_sub.Init(None, 0)
        self.state_sub.Init(None, 0)
        self.state = None
        self.num_dof = 7  # for single hand

    def _prepare_low_state(self) -> np.ndarray:
        self.state = self.state_sub.Read()

        if not self.state:
            print("No state received")
            return

        state_data = (
            np.concatenate(
                [
                    [self.state.motor_state[i].q for i in range(self.num_dof)],
                    [self.state.motor_state[i].dq for i in range(self.num_dof)],
                    [self.state.motor_state[i].tau_est for i in range(self.num_dof)],
                    [self.state.motor_state[i].ddq for i in range(self.num_dof)],
                ],
                axis=0,
            )
            .astype(np.float64)
            .reshape(1, -1)
        )
        return state_data
