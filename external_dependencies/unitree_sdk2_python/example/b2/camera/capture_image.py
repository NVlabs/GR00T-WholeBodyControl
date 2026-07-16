import time
import os
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient
from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 创建前置相机客户端
    front_client = FrontVideoClient()
    front_client.SetTimeout(3.0)
    front_client.Init()

    # 创建后置相机客户端
    back_client = BackVideoClient()
    back_client.SetTimeout(3.0)
    back_client.Init()

    print("##################Get Front Camera Image###################")
    # 获取前置相机图像
    front_code, front_data = front_client.GetImageSample()

    if front_code != 0:
        print("Get front camera image error. Code:", front_code)
    else:
        front_image_name = "./front_img.jpg"
        print("Front Image Saved as:", front_image_name)

        with open(front_image_name, "+wb") as f:
            f.write(bytes(front_data))

    print("##################Get Back Camera Image###################")
    # 获取后置相机图像
    back_code, back_data = back_client.GetImageSample()

    if back_code != 0:
        print("Get back camera image error. Code:", back_code)
    else:
        back_image_name = "./back_img.jpg"
        print("Back Image Saved as:", back_image_name)

        with open(back_image_name, "+wb") as f:
            f.write(bytes(back_data))

    time.sleep(1)
