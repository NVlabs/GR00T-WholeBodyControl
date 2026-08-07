import time

from ..idl.unitree_api.msg.dds_ import Request_ as Request
from ..idl.unitree_api.msg.dds_ import RequestHeader_ as RequestHeader
from ..idl.unitree_api.msg.dds_ import RequestLease_ as RequestLease
from ..idl.unitree_api.msg.dds_ import RequestIdentity_ as RequestIdentity
from ..idl.unitree_api.msg.dds_ import RequestPolicy_ as RequestPolicy

from ..utils.future import FutureResult

from .client_stub import ClientStub
from .internal import *


"""
" class ClientBase
"""
class ClientBase:
    def __init__(self, serviceName: str):
        self.__timeout = 1.0
        self.__stub = ClientStub(serviceName)
        self.__stub.Init()

    def SetTimeout(self, timeout: float):
        self.__timeout = timeout

    def _CallBase(self, apiId: int, parameter: str, proirity: int = 0, leaseId: int = 0):
        # print("[CallBase] call apiId:", apiId, ", proirity:", proirity, ", leaseId:", leaseId)
        header = self.__SetHeader(apiId, leaseId, proirity, False)
        request = Request(header, parameter, [])

        future = self.__stub.SendRequest(request, self.__timeout)
        if future is None:
            return RPC_ERR_CLIENT_SEND, None

        result = future.GetResult(self.__timeout)

        if result.code != FutureResult.FUTURE_SUCC:
            self.__stub.RemoveFuture(request.header.identity.id)
            code = RPC_ERR_CLIENT_API_TIMEOUT if result.code == FutureResult.FUTUTE_ERR_TIMEOUT else RPC_ERR_UNKNOWN
            return code, None

        response = result.value

        if response.header.identity.api_id != apiId:
            return RPC_ERR_CLIENT_API_NOT_MATCH, None
        else:
            return response.header.status.code, response.data

    def _CallNoReplyBase(self, apiId: int, parameter: str, proirity: int, leaseId: int):
        header = self.__SetHeader(apiId, leaseId, proirity, True)
        request = Request(header, parameter, [])

        if self.__stub.Send(request, self.__timeout):
            return 0
        else:
            return RPC_ERR_CLIENT_SEND
    
    def _CallBinaryBase(self, apiId: int, parameter: list, proirity: int, leaseId: int):
        header = self.__SetHeader(apiId, leaseId, proirity, False)
        request = Request(header, "", parameter)
        
        future = self.__stub.SendRequest(request, self.__timeout)
        if future is None:
            return RPC_ERR_CLIENT_SEND, None

        result = future.GetResult(self.__timeout)
        if result.code != FutureResult.FUTURE_SUCC:
            self.__stub.RemoveFuture(request.header.identity.id)
            code = RPC_ERR_CLIENT_API_TIMEOUT if result.code == FutureResult.FUTUTE_ERR_TIMEOUT else RPC_ERR_UNKNOWN
            return code, None

        response = result.value

        if response.header.identity.api_id != apiId:
            return RPC_ERR_CLIENT_API_NOT_MATCH, None
        else:
            return response.header.status.code, response.binary

    def _CallBinaryNoReplyBase(self, apiId: int, parameter: list, proirity: int, leaseId: int):
        header = self.__SetHeader(apiId, leaseId, proirity, True)
        request = Request(header, "", parameter)

        if self.__stub.Send(request, self.__timeout):
            return 0
        else:
            return RPC_ERR_CLIENT_SEND
    
    def __SetHeader(self, apiId: int, leaseId: int, priority: int, noReply: bool):
        identity = RequestIdentity(time.monotonic_ns(), apiId)
        lease = RequestLease(leaseId)
        policy = RequestPolicy(priority, noReply)
        return RequestHeader(identity, lease, policy)
