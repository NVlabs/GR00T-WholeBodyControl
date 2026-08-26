import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread
import unitree_go2_const as go2

crc = CRC()
lowCmdThreadPtr=Thread()

if __name__ == '__main__':

    ChannelFactoryInitialize(1, "enp2s0")
    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("lowcmd", LowCmd_)
    pub.Init()

    while True:
        # Create a Userdata message
        cmd = unitree_go_msg_dds__LowCmd_()
        
        # Toque controle, set RL_2 toque
        cmd.motor_cmd[go2.LegID["RL_2"]].mode = 0x01
        cmd.motor_cmd[go2.LegID["RL_2"]].q = go2.PosStopF # Set to stop position(rad)
        cmd.motor_cmd[go2.LegID["RL_2"]].kp = 0
        cmd.motor_cmd[go2.LegID["RL_2"]].dq = go2.VelStopF # Set to stop angular velocity(rad/s)
        cmd.motor_cmd[go2.LegID["RL_2"]].kd = 0
        cmd.motor_cmd[go2.LegID["RL_2"]].tau = 1 # target toque is set to 1N.m

        # Poinstion(rad) control, set RL_0 rad
        cmd.motor_cmd[go2.LegID["RL_0"]].mode = 0x01
        cmd.motor_cmd[go2.LegID["RL_0"]].q = 0  # Taregt angular(rad)
        cmd.motor_cmd[go2.LegID["RL_0"]].kp = 10 # Poinstion(rad) control kp gain
        cmd.motor_cmd[go2.LegID["RL_0"]].dq = 0  # Taregt angular velocity(rad/ss)
        cmd.motor_cmd[go2.LegID["RL_0"]].kd = 1  # Poinstion(rad) control kd gain
        cmd.motor_cmd[go2.LegID["RL_0"]].tau = 0 # Feedforward toque 1N.m
        
        cmd.crc = crc.Crc(cmd)

        #Publish message
        if pub.Write(cmd):
            print("Publish success. msg:", cmd.crc)
        else:
            print("Waitting for subscriber.")

        time.sleep(0.002)

    pub.Close()
