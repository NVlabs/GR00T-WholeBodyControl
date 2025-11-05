import json

from ...rpc.client import Client
from ...rpc.client_internal import *
from .robot_state_api import *


"""
" class ServiceState
"""
class ServiceState:
    def __init__(self, name: str = None, status: int = None, protect: bool = None):
        self.name = name
        self.status = status
        self.protect = protect

"""
" class RobotStateClient
"""
class RobotStateClient(Client):
    def __init__(self):
        super().__init__(ROBOT_STATE_SERVICE_NAME, False)

    def Init(self):
        # set api version
        self._SetApiVerson(ROBOT_STATE_API_VERSION)
        # regist api
        self._RegistApi(ROBOT_STATE_API_ID_SERVICE_SWITCH, 0)
        self._RegistApi(ROBOT_STATE_API_ID_REPORT_FREQ, 0)
        self._RegistApi(ROBOT_STATE_API_ID_SERVICE_LIST, 0)

    def ServiceList(self):
        p = {}
        parameter = json.dumps(p)

        code, data = self._Call(ROBOT_STATE_API_ID_SERVICE_LIST, parameter)

        if code != 0:
            return code, None

        lst = []

        d = json.loads(data)
        for t in d:
            s = ServiceState()
            s.name = t["name"]
            s.status = t["status"]
            s.protect = t["protect"]
            lst.append(s)
            
        return code, lst
            

    def ServiceSwitch(self, name: str, switch: bool):
        p = {}
        p["name"] = name
        p["switch"] = int(switch)
        parameter = json.dumps(p)
        
        code, data = self._Call(ROBOT_STATE_API_ID_SERVICE_SWITCH, parameter)
        
        if code != 0:
            return code
      
        d = json.loads(data)

        status = d["status"]
    
        if status == 5:
            return ROBOT_STATE_ERR_SERVICE_PROTECTED

        if status != 0 and status != 1:
            return ROBOT_STATE_ERR_SERVICE_SWITCH
        
        return code

    def SetReportFreq(self, interval: int, duration: int):
        p = {}
        p["interval"] = interval
        p["duration"] = duration
        parameter = json.dumps(p)
        
        code, data = self._Call(ROBOT_STATE_API_ID_REPORT_FREQ, p)
        return code
