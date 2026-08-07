import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

import unitree_go2_const as go2


def LowStateHandler(msg: LowState_):
    
    # print front right hip motor states
    print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
    print("IMU state: ", msg.imu_state)
    print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)


if __name__ == "__main__":
    # Modify "enp2s0" to the actual network interface
    ChannelFactoryInitialize(0, "enp2s0")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    while True:
        time.sleep(10.0)
