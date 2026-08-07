import json

from ...rpc.client import Client
from .g1_audio_api import *

"""
" class SportClient
"""
class AudioClient(Client):
    def __init__(self):
        super().__init__(AUDIO_SERVICE_NAME, False)
        self.tts_index = 0

    def Init(self):
        # set api version
        self._SetApiVerson(AUDIO_API_VERSION)

        # regist api
        self._RegistApi(ROBOT_API_ID_AUDIO_TTS, 0)
        self._RegistApi(ROBOT_API_ID_AUDIO_ASR, 0)
        self._RegistApi(ROBOT_API_ID_AUDIO_START_PLAY, 0)
        self._RegistApi(ROBOT_API_ID_AUDIO_STOP_PLAY, 0)
        self._RegistApi(ROBOT_API_ID_AUDIO_GET_VOLUME, 0)
        self._RegistApi(ROBOT_API_ID_AUDIO_SET_VOLUME, 0) 
        self._RegistApi(ROBOT_API_ID_AUDIO_SET_RGB_LED, 0) 

    ## API Call ##
    def TtsMaker(self, text: str, speaker_id: int):
        self.tts_index += self.tts_index
        p = {}
        p["index"] = self.tts_index
        p["text"] = text
        p["speaker_id"] = speaker_id
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_AUDIO_TTS, parameter)
        return code

    def GetVolume(self):
        p = {}
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_AUDIO_GET_VOLUME, parameter)
        if code == 0:
            return code, json.loads(data)
        else:
            return code, None

    def SetVolume(self, volume: int):
        p = {}
        p["volume"] = volume
        # p["name"] = 'volume'
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_AUDIO_SET_VOLUME, parameter)
        return code

    def LedControl(self, R: int, G: int, B: int):
        p = {}
        p["R"] = R
        p["G"] = G
        p["B"] = B
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_AUDIO_SET_RGB_LED, parameter)
        return code
