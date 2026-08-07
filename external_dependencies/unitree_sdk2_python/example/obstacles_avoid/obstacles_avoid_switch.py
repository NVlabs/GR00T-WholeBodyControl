import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient

if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = ObstaclesAvoidClient()
    client.SetTimeout(3.0)
    client.Init()

    while True:
        print("##################GetServerApiVersion###################")
        code, serverAPiVersion = client.GetServerApiVersion()
        if code != 0:
            print("get server api error. code:", code)
        else:
            print("get server api version:", serverAPiVersion)

        if serverAPiVersion != client.GetApiVersion():
            print("api version not equal.")

        time.sleep(3)

        print("##################SwitchGet###################")
        code, enable = client.SwitchGet()
        if code != 0:
            print("switch get error. code:", code)
        else:
            print("switch get success. enable:", enable)
            
        time.sleep(3)
        
        print("##################SwitchSet (on)###################")
        code = client.SwitchSet(True)
        if code != 0:
            print("switch set error. code:", code)
        else:
            print("switch set success.")
            
        time.sleep(3)

        print("##################SwitchGet###################")
        code, enable1 = client.SwitchGet()
        if code != 0:
            print("switch get error. code:", code)
        else:
            print("switch get success. enable:", enable1)
            
        time.sleep(3)

        print("##################SwitchSet (off)###################")
        code = client.SwitchSet(False)
        if code != 0:
            print("switch set error. code:", code)
        else:
            print("switch set success.")
            
        time.sleep(3)

        print("##################SwitchGet###################")
        code, enable1 = client.SwitchGet()
        if code != 0:
            print("switch get error. code:", code)
        else:
            print("switch get success. enable:", enable1)
            
        time.sleep(3)


        print("##################SwitchSet (enable)###################")

        code = client.SwitchSet(enable)
        if code != 0:
            print("switch set error. code:", code)
        else:
            print("switch set success. enable:", enable)
            
        time.sleep(3)

        print("##################SwitchGet###################")
        code, enable = client.SwitchGet()
        if code != 0:
            print("switch get error. code:", code)
        else:
            print("switch get success. enable:", enable)
            
        time.sleep(3)
        