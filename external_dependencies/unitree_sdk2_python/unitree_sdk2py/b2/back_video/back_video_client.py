import json

from ...rpc.client import Client
from .back_video_api import *


"""
" class FrontVideoClient
"""
class BackVideoClient(Client):
    def __init__(self):
        super().__init__(ROBOT_BACK_VIDEO_SERVICE_NAME, False)


    def Init(self):
        # set api version
        self._SetApiVerson(ROBOT_BACK_VIDEO_API_VERSION)
        # regist api
        self._RegistApi(ROBOT_BACK_VIDEO_API_ID_GETIMAGESAMPLE, 0)

    # 1001
    def GetImageSample(self):
        return self._CallBinary(ROBOT_BACK_VIDEO_API_ID_GETIMAGESAMPLE, [])
