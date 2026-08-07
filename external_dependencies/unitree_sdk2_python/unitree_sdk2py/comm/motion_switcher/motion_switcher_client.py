import json

from ...rpc.client import Client
from .motion_switcher_api import *

"""
" class MotionSwitcherClient
"""
class MotionSwitcherClient(Client):
    def __init__(self):
        super().__init__(MOTION_SWITCHER_SERVICE_NAME, False)


    def Init(self):
        # set api version
        self._SetApiVerson(MOTION_SWITCHER_API_VERSION)
        
        # regist api
        self._RegistApi(MOTION_SWITCHER_API_ID_CHECK_MODE, 0)
        self._RegistApi(MOTION_SWITCHER_API_ID_SELECT_MODE, 0)
        self._RegistApi(MOTION_SWITCHER_API_ID_RELEASE_MODE, 0)
        self._RegistApi(MOTION_SWITCHER_API_ID_SET_SILENT, 0)
        self._RegistApi(MOTION_SWITCHER_API_ID_GET_SILENT, 0)

    # 1001
    def CheckMode(self):
        p = {}
        parameter = json.dumps(p)
        code, data = self._Call(MOTION_SWITCHER_API_ID_CHECK_MODE, parameter)
        if code == 0:
            return code, json.loads(data)
        else:
            return code, None

    # 1002
    def SelectMode(self, nameOrAlias):
        p = {}
        p["name"] = nameOrAlias
        parameter = json.dumps(p)
        code, data = self._Call(MOTION_SWITCHER_API_ID_SELECT_MODE, parameter)
      
        return code, None

    # 1003
    def ReleaseMode(self):
        p = {}
        parameter = json.dumps(p)
        code, data = self._Call(MOTION_SWITCHER_API_ID_RELEASE_MODE, parameter)
      
        return code, None
    