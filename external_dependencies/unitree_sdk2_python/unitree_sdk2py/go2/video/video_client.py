import json

from ...rpc.client import Client
from .video_api import *


"""
" class VideoClient
"""
class VideoClient(Client):
    def __init__(self):
        super().__init__(VIDEO_SERVICE_NAME, False)


    def Init(self):
        # set api version
        self._SetApiVerson(VIDEO_API_VERSION)
        # regist api
        self._RegistApi(VIDEO_API_ID_GETIMAGESAMPLE, 0)

    # 1001
    def GetImageSample(self):
        return self._CallBinary(VIDEO_API_ID_GETIMAGESAMPLE, [])
