import time

from typing import Callable, Any

from ..idl.unitree_api.msg.dds_ import Request_ as Request
from ..idl.unitree_api.msg.dds_ import ResponseStatus_ as ResponseStatus
from ..idl.unitree_api.msg.dds_ import ResponseHeader_ as ResponseHeader
from ..idl.unitree_api.msg.dds_ import Response_ as Response

from .server_base import ServerBase
from .lease_server import LeaseServer
from .internal import *

"""
" class Server
"""
class Server(ServerBase):
    def __init__(self, name: str):
        self.__apiVersion = ""
        self.__apiHandlerMapping = {}
        self.__apiBinaryHandlerMapping = {}
        self.__apiBinarySet = {}
        self.__enableLease = False
        self.__leaseServer = None
        super().__init__(name)

    def Init(self):
        pass

    def StartLease(self, term: float = 1.0):
        self.__enableLease = True
        self.__leaseServer = LeaseServer(self.GetName(), term)
        self.__leaseServer.Init()
        self.__leaseServer.Start(False)

    def Start(self, enablePrioQueue: bool = False):
        super()._SetServerRequestHandler(self.__ServerRequestHandler)
        super()._Start(enablePrioQueue)

    def GetApiVersion(self):
        return self.__apiVersion

    def _SetApiVersion(self, apiVersion: str):
        self.__apiVersion = apiVersion
        print("[Server] set api version:", self.__apiVersion)

    def _RegistHandler(self, apiId: int, handler: Callable, checkLease: bool):
        self.__apiHandlerMapping[apiId] = (handler, checkLease)

    def _RegistBinaryHandler(self, apiId: int, handler: Callable, checkLease: bool):
        self.__apiBinaryHandlerMapping[apiId] = (handler, checkLease)
        self.__apiBinarySet.add(apiId)

    def __GetHandler(self, apiId: int):
        if apiId in self.__apiHandlerMapping:
            return self.__apiHandlerMapping.get(apiId)
        else:
            return None, False

    def __GetBinaryHandler(self, apiId: int):
        if apiId in self.__apiBinaryHandlerMapping:
            return self.__apiBinaryHandlerMapping.get(apiId)
        else:
            return None, False

    def __IsBinary(self, apiId):
        return apiId in self.__apiBinarySet

    def __CheckLeaseDenied(self, leaseId: int):
        if (self.__enableLease):
            return self.__leaseServer.CheckRequestLeaseDenied(leaseId)
        else:
            return False

    def __ServerRequestHandler(self, request: Request):
        parameter = request.parameter
        parameterBinary = request.binary

        identity = request.header.identity
        leaseId = request.header.lease.id
        apiId = identity.api_id

        code = 0
        data = ""
        dataBinary = []

        if apiId == RPC_API_ID_INTERNAL_API_VERSION:
            data = self.__apiVersion
        else:
            requestHandler = None
            binaryRequestHandler = None
            checkLease = False
            
            if self.__IsBinary(apiId):
                binaryRequestHandler, checkLease = self.__GetBinaryHandler(apiId)
            else:
                requestHandler, checkLease = self.__GetHandler(apiId)

            if requestHandler is None and binaryRequestHandler is None:
                code = RPC_ERR_SERVER_API_NOT_IMPL
            elif checkLease and self.__CheckLeaseDenied(leaseId):
                code = RPC_ERR_SERVER_LEASE_DENIED
            else:
                try:
                    if binaryRequestHandler is None:
                        code, data = requestHandler(parameter)
                        if code != 0:
                            data = ""
                    else:
                        code, dataBinary = binaryRequestHandler(parameterBinary)
                        if code != 0:
                            dataBinary = []
                except:
                    code = RPC_ERR_SERVER_INTERNAL

        if request.header.policy.noreply:
            return

        status = ResponseStatus(code)
        response = Response(ResponseHeader(identity, status), data, dataBinary)

        self._SendResponse(response)