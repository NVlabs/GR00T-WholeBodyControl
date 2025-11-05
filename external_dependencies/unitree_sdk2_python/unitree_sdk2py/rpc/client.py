from .client_base import ClientBase
from .lease_client import LeaseClient
from .internal import *

"""
" class Client
"""
class Client(ClientBase):
    def __init__(self, serviceName: str, enabaleLease: bool = False):
        super().__init__(serviceName)

        self.__apiMapping = {}
        self.__apiVersion = None
        self.__leaseClient = None
        self.__enableLease = enabaleLease

        if (self.__enableLease):
            self.__leaseClient = LeaseClient(serviceName)
            self.__leaseClient.Init()

    def WaitLeaseApplied(self):
        if self.__enableLease:
            self.__leaseClient.WaitApplied()

    def GetLeaseId(self):
        if self.__enableLease:
            return self.__leaseClient.GetId()
        else:
            return None

    def GetApiVersion(self):
        return self.__apiVersion
    
    def GetServerApiVersion(self):
        code, apiVerson = self._CallBase(RPC_API_ID_INTERNAL_API_VERSION, "{}", 0, 0)
        if code != 0:
            print("[Client] get server api version error:", code)
            return code, None
        else:
            return code, apiVerson

    def _SetApiVerson(self, apiVersion: str):
        self.__apiVersion = apiVersion

    def _Call(self, apiId: int, parameter: str):
        ret, proirity, leaseId = self.__CheckApi(apiId)
        if ret == 0:
            return self._CallBase(apiId, parameter, proirity, leaseId)
        else:
            return RPC_ERR_CLIENT_API_NOT_REG, None
            
    def _CallNoReply(self, apiId: int, parameter: str):
        ret, proirity, leaseId = self.__CheckApi(apiId)
        if ret == 0:
            return self._CallNoReplyBase(apiId, parameter, proirity, leaseId)
        else:
            return RPC_ERR_CLIENT_API_NOT_REG
    
    def _CallBinary(self, apiId: int, parameter: list):
        ret, proirity, leaseId = self.__CheckApi(apiId)
        if ret == 0:
            return self._CallBinaryBase(apiId, parameter, proirity, leaseId)
        else:
            return RPC_ERR_CLIENT_API_NOT_REG, None

    def _CallBinaryNoReply(self, apiId: int, parameter: list):
        ret, proirity, leaseId = self.__CheckApi(apiId)
        if ret == 0:
            return self._CallBinaryNoReplyBase(apiId, parameter, proirity, leaseId)
        else:
            return RPC_ERR_CLIENT_API_NOT_REG
    
    def _RegistApi(self, apiId: int, proirity: int):
        self.__apiMapping[apiId] = proirity
    
    def __CheckApi(self, apiId: int):
        proirity = 0
        leaseId = 0

        if apiId > RPC_INTERNAL_API_ID_MAX:
            proirity = self.__apiMapping.get(apiId)
            
            if proirity is None:
                return RPC_ERR_CLIENT_API_NOT_REG, proirity, leaseId
            
            if self.__enableLease:
                leaseId = self.__leaseClient.GetId()

        return 0, proirity, leaseId