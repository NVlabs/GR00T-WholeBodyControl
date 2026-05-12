from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient
from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient
import cv2
import numpy as np
import sys

def display_image(window_name, data):
    # If data is a list, we need to convert it to a bytes object
    if isinstance(data, list):
        data = bytes(data)
    
    # Now convert to numpy image
    image_data = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is not None:
        cv2.imshow(window_name, image)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    frontCameraClient = FrontVideoClient()  # Create a front camera video client
    frontCameraClient.SetTimeout(3.0)
    frontCameraClient.Init()

    backCameraClient = BackVideoClient()  # Create a back camera video client
    backCameraClient.SetTimeout(3.0)
    backCameraClient.Init()

    # Loop to continuously fetch images
    while True:
        # Get front camera image
        front_code, front_data = frontCameraClient.GetImageSample()
        if front_code == 0:
            display_image("Front Camera", front_data)

        # Get back camera image
        back_code, back_data = backCameraClient.GetImageSample()
        if back_code == 0:
            display_image("Back Camera", back_data)

        # Press ESC to stop
        if cv2.waitKey(20) == 27:
            break

    # Clean up windows
    cv2.destroyAllWindows()

