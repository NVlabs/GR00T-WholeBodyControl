import time

from typing import Callable, Any

from ..idl.unitree_api.msg.dds_ import Request_ as Request
from ..idl.unitree_api.msg.dds_ import Response_ as Response

from .server_stub import ServerStub


"""
" class ServerBase
"""
class ServerBase:
    def __init__(self, name: str):
        self.__name = name
        self.__serverRequestHandler = None
        self.__serverStub = ServerStub(self.__name)

    def GetName(self):
        return self.__name

    def _Start(self, enablePrioQueue: bool = False):
        self.__serverStub.Init(self.__serverRequestHandler, enablePrioQueue)
        print("[ServerBase] server started. name:", self.__name, ", enable proirity queue:", enablePrioQueue)

    def _SetServerRequestHandler(self, serverRequestHandler: Callable):
        self.__serverRequestHandler = serverRequestHandler

    def _SendResponse(self, response: Response):
        if not self.__serverStub.Send(response, 1.0):
            print("[ServerBase] send response error.")
