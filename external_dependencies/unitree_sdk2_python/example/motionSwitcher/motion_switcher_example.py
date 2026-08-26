import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class Custom:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def selectMode(self,name):
        ret = self.msc.SelectMode(name)
        return ret


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    selectMode = "ai" 
    # selectMode = "normal"
    # selectMode = "advanced" 
    # selectMode = "ai-w"  # for wheeled robot
    ret = custom.selectMode(selectMode) 
    print("ret: ",ret)

