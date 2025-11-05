import json

from ...rpc.client import Client
from .vui_api import *


"""
" class VideoClient
"""
class VuiClient(Client):
    def __init__(self):
        super().__init__(VUI_SERVICE_NAME, False)

    def Init(self):
        # set api version
        self._SetApiVerson(VUI_API_VERSION)
        # regist api
        self._RegistApi(VUI_API_ID_SETSWITCH, 0)
        self._RegistApi(VUI_API_ID_GETSWITCH, 0)
        self._RegistApi(VUI_API_ID_SETVOLUME, 0)
        self._RegistApi(VUI_API_ID_GETVOLUME, 0)
        self._RegistApi(VUI_API_ID_SETBRIGHTNESS, 0)
        self._RegistApi(VUI_API_ID_GETBRIGHTNESS, 0)

    # 1001
    def SetSwitch(self, enable: int):
        p = {}
        p["enable"] = enable
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_SETSWITCH, parameter)
        return code

    # 1002
    def GetSwitch(self):
        p = {}
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_GETSWITCH, parameter)
        if code == 0:
            d = json.loads(data)
            return code, d["enable"]
        else:
            return code, None

    # 1003
    def SetVolume(self, level: int):
        p = {}
        p["volume"] = level
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_SETVOLUME, parameter)
        return code

    # 1006
    def GetVolume(self):
        p = {}
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_GETVOLUME, parameter)
        if code == 0:
            d = json.loads(data)
            return code, d["volume"]
        else:
            return code, None

    # 1005
    def SetBrightness(self, level: int):
        p = {}
        p["brightness"] = level
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_SETBRIGHTNESS, parameter)
        return code

    # 1006
    def GetBrightness(self):
        p = {}
        parameter = json.dumps(p)

        code, data = self._Call(VUI_API_ID_GETBRIGHTNESS, parameter)
        if code == 0:
            d = json.loads(data)
            return code, d["brightness"]
        else:
            return code, None