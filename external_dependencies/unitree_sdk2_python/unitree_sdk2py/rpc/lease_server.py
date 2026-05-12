import time
import json

from threading import Lock

from ..idl.unitree_api.msg.dds_ import Request_ as Request
from ..idl.unitree_api.msg.dds_ import ResponseHeader_ as ResponseHeader
from ..idl.unitree_api.msg.dds_ import ResponseStatus_ as ResponseStatus
from ..idl.unitree_api.msg.dds_ import Response_ as Response

from .internal import *
from .server_base import ServerBase


"""
" class LeaseCache
"""
class LeaseCache:
    def __init__(self):
        self.lastModified = 0
        self.id = 0
        self.name = None
    
    def Set(self, id: int, name: str, lastModified: int) :
        self.id = id
        self.name = name
        self.lastModified = lastModified

    def Renewal(self, lastModified: int):
        self.lastModified = lastModified

    def Clear(self):
        self.id = 0
        self.lastModified = 0
        self.name = None


"""
" class LeaseServer
"""
class LeaseServer(ServerBase):
    def __init__(self, name: str, term: float):
        self.__term = int(term * 1000000)
        self.__lock = Lock()
        self.__cache = LeaseCache()
        super().__init__(name + "_lease")

    def Init(self):
        pass

    def Start(self, enablePrioQueue: bool = False):
        super()._SetServerRequestHandler(self.__ServerRequestHandler)
        super()._Start(enablePrioQueue)

    def CheckRequestLeaseDenied(self, leaseId: int):
        with self.__lock:
            if self.__cache.id == 0:
                return self.__cache.id != leaseId

            now = self.__Now()
            if now > self.__cache.lastModified + self.__term:
                self.__cache.Clear()
                return False
            else:
                return self.__cache.id  != leaseId

    def __Apply(self, parameter: str):
        name = ""
        data = ""

        try:
            p = json.loads(parameter)
            name = p.get("name")

        except:
            print("[LeaseServer] apply json loads error. parameter:", parameter)
            return RPC_ERR_SERVER_API_PARAMETER, data

        if not name:
            name = "anonymous"

        id = 0
        lastModified = 0
        setted = False

        now = self.__Now()

        with self.__lock:
            id = self.__cache.id
            lastModified = self.__cache.lastModified
    
            if id == 0 or now > lastModified + self.__term:
                if id != 0:
                    print("[LeaseServer] id expired:", id, ", name:", self.__cache.name)
        
                id = self.__GenerateId()
                self.__cache.Set(id, name, now)
                setted = True

                print("[LeaseServer] id stored:", id, ", name:", name)

        if setted:
            d = {}
            d["id"] = id
            d["term"] = self.__term
            data = json.dumps(d)
            return 0, data
        else:
            return RPC_ERR_SERVER_LEASE_EXIST, data


    def __Renewal(self, id: int):
        now = self.__Now()

        with self.__lock:
            if self.__cache.id != id:
                return RPC_ERR_SERVER_LEASE_NOT_EXIST
    
            if now > self.__cache.lastModified + self.__term:
                self.__cache.Clear()
                return RPC_ERR_SERVER_LEASE_NOT_EXIST
            else:
                self.__cache.Renewal(now)
                return 0

    def __ServerRequestHandler(self, request: Request):
        identity = request.header.identity
        parameter = request.parameter
        apiId = identity.api_id
        code = RPC_ERR_SERVER_API_NOT_IMPL
        data = ""

        if apiId == RPC_API_ID_LEASE_APPLY:
            code, data = self.__Apply(parameter)
        elif apiId == RPC_API_ID_LEASE_RENEWAL:
            code = self.__Renewal(request.header.lease.id)
        else:
            print("[LeaseServer] api is not implemented. apiId", apiId)

        if request.header.policy.noreply:
            return

        status = ResponseStatus(code)
        response = Response(ResponseHeader(identity, status), data, [])
        self._SendResponse(response)

    def __GenerateId(self):
        return self.__Now()
    
    def __Now(self):
        return int(time.time_ns()/1000)