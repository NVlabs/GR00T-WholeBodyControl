import time
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.vui.vui_client import VuiClient

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp2s0")

    client = VuiClient()
    client.SetTimeout(3.0)
    client.Init()

    for i in range(1, 11):
        print("#################GetBrightness####################")
        code, level = client.GetBrightness()

        if code != 0:
            print("get brightness error. code:", code)
        else:
            print("get brightness success. level:", level)

        time.sleep(1)

        print("#################SetBrightness####################")

        code = client.SetBrightness(i)

        if code != 0:
            print("set brightness error. code:", code)
        else:
            print("set brightness success. level:", i)

        time.sleep(1)

    print("#################SetBrightness 0####################")

    code  = client.SetBrightness(0)

    if code != 0:
        print("set brightness error. code:", code)
    else:
        print("set brightness 0 success.")

    for i in range(1, 11):
        print("#################GetVolume####################")
        code, level = client.GetVolume()

        if code != 0:
            print("get volume error. code:", code)
        else:
            print("get volume success. level:", level)

        time.sleep(1)

        print("#################SetVolume####################")

        code = client.SetVolume(i)

        if code != 0:
            print("set volume error. code:", code)
        else:
            print("set volume success. level:", i)

        time.sleep(1)

    print("#################SetVolume 0####################")

    code  = client.SetVolume(0)

    if code != 0:
        print("set volume error. code:", code)
    else:
        print("set volume 0 success.")
