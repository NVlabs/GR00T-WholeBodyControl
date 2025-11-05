import json

from ...rpc.client import Client
from .front_video_api import *


"""
" class FrontVideoClient
"""
class FrontVideoClient(Client):
    def __init__(self):
        super().__init__(ROBOT_FRONT_VIDEO_SERVICE_NAME, False)


    def Init(self):
        # set api version
        self._SetApiVerson(ROBOT_FRONT_VIDEO_API_VERSION)
        # regist api
        self._RegistApi(ROBOT_FRONT_VIDEO_API_ID_GETIMAGESAMPLE, 0)

    # 1001
    def GetImageSample(self):
        return self._CallBinary(ROBOT_FRONT_VIDEO_API_ID_GETIMAGESAMPLE, [])
