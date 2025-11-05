import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    try:
        client = ObstaclesAvoidClient()
        client.SetTimeout(3.0)
        client.Init()

        while not client.SwitchGet()[1]:
            client.SwitchSet(True)
            time.sleep(0.1)

        print("obstacles avoid switch on")

        client.UseRemoteCommandFromApi(True)
        time.sleep(0.5)
        client.Move(0.5, 0.0, 0.0)
        time.sleep(1.0) # move 1s
        client.Move(0.0, 0.0, 0.0)
        client.UseRemoteCommandFromApi(False)

    except KeyboardInterrupt:
        client.Move(0.0, 0.0, 0.0)
        client.UseRemoteCommandFromApi(False)
        print("exit!!")
        