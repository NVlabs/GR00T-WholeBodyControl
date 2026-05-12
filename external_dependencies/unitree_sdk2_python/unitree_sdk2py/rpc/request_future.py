from threading import Condition, Lock
from enum import Enum

from ..idl.unitree_api.msg.dds_ import Response_ as Response
from ..utils.future import Future, FutureResult


"""
" class RequestFuture
"""
class RequestFuture(Future):
    def __init__(self):
        self.__requestId = None
        super().__init__()

    def SetRequestId(self, requestId: int):
        self.__requestId = requestId

    def GetRequestId(self):
        return self.__requestId


class RequestFutureQueue:
    def __init__(self):
        self.__data = {}
        self.__lock = Lock()
        
    def Set(self, requestId: int, future: RequestFuture):
        if future is None:
            return False
        with self.__lock:
            self.__data[requestId] = future
            return True

    def Get(self, requestId: int):
        future = None
        with self.__lock:
            future = self.__data.get(requestId)
            if future is not None:
                self.__data.pop(requestId)
        return future

    def Remove(self, requestId: int):
        with self.__lock:
            if id in self.__data:
                self.__data.pop(requestId)