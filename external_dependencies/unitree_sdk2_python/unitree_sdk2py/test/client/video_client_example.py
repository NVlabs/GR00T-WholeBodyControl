import time
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp2s0")

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    print("##################GetImageSample###################")
    code, data = client.GetImageSample()

    if code != 0:
        print("get image sample error. code:", code)
    else:
        imageName = os.path.dirname(__file__) + time.strftime('/%Y%m%d%H%M%S.jpg',time.localtime())
        print("ImageName:", imageName)
        
        with open(imageName, "+wb") as f:
            f.write(bytes(data))

    time.sleep(1)
