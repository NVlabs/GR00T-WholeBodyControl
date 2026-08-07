import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

class Custom:
    def __init__(self):
        # create publisher #
        self.publisher = ChannelPublisher("rt/utlidar/switch", String_)
        self.publisher.Init()
        self.low_cmd = std_msgs_msg_dds__String_()   

    def go2_utlidar_switch(self,status):
        if status == "OFF":
            self.low_cmd.data = "OFF"
        elif status == "ON":
            self.low_cmd.data = "ON"

        self.publisher.Write(self.low_cmd)
        

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.go2_utlidar_switch("OFF")



