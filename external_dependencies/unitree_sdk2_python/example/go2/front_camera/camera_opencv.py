from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()  # Create a video client
    client.SetTimeout(3.0)
    client.Init()

    code, data = client.GetImageSample()

    # Request normal when code==0
    while code == 0:
        # Get Image data from Go2 robot
        code, data = client.GetImageSample()

        # Convert to numpy image
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Display image
        cv2.imshow("front_camera", image)
        # Press ESC to stop
        if cv2.waitKey(20) == 27:
            break

    if code != 0:
        print("Get image sample error. code:", code)
    else:
        # Capture an image
        cv2.imwrite("front_image.jpg", image)

    cv2.destroyWindow("front_camera")
