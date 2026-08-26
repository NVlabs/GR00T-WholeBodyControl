import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from user_data import *


if __name__ == "__main__":
    ChannelFactoryInitialize()

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("topic", UserData)
    pub.Init()

    for i in range(30):
        # Create a Userdata message
        msg = UserData(" ", 0)
        msg.string_data = "Hello world"
        msg.float_data = time.time()

        # Publish message
        if pub.Write(msg, 0.5):
            print("Publish success. msg:", msg)
        else:
            print("Waitting for subscriber.")

        time.sleep(1)

    pub.Close()
