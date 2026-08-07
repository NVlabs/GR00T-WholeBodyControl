from threading import Condition
from typing import Any
from enum import Enum

"""
" Enum RequtestFutureState
"""
class FutureState(Enum):
    DEFER = 0
    READY = 1
    FAILED = 2

"""
" class FutureException
"""
class FutureResult:
    FUTURE_SUCC = 0
    FUTUTE_ERR_TIMEOUT = 1
    FUTURE_ERR_FAILED = 2
    FUTURE_ERR_UNKNOWN = 3

    def __init__(self, code: int, msg: str, value: Any = None):
        self.code = code
        self.msg = msg
        self.value = value

    def __str__(self):
        return f"FutureResult(code={str(self.code)}, msg='{self.msg}', value={self.value})"

class Future:
    def __init__(self):
        self.__state = FutureState.DEFER
        self.__msg = None
        self.__condition = Condition()
    
    def GetResult(self, timeout: float = None):
        with self.__condition:
            return self.__WaitResult(timeout)

    def Wait(self, timeout: float = None):
        with self.__condition:
            return self.__Wait(timeout)

    def Ready(self, value):
        with self.__condition:
            ready = self.__Ready(value)
            self.__condition.notify()
            return ready

    def Fail(self, reason: str):
        with self.__condition:
            fail = self.__Fail(reason)
            self.__condition.notify()
            return fail

    def __Wait(self, timeout: float = None):
        if not self.__IsDeferred():
            return True
        try:
            if timeout is None:
                return self.__condition.wait()
            else:
                return self.__condition.wait(timeout)
        except:
            print("[Future] future wait error")
            return False

    def __WaitResult(self, timeout: float = None):
        if not self.__Wait(timeout):
            return FutureResult(FutureResult.FUTUTE_ERR_TIMEOUT, "future wait timeout")

        if self.__IsReady():
            return FutureResult(FutureResult.FUTURE_SUCC, "success", self.__value)
        elif self.__IsFailed():
            return FutureResult(FutureResult.FUTURE_ERR_FAILED, self.__msg)
        else:
            return FutureResult(FutureResult.FUTURE_ERR_UNKNOWN, "future state error:" + str(self.__state))

    def __Ready(self, value):
        if not self.__IsDeferred():
            print("[Future] futrue state is not defer")
            return False
        else:
            self.__value = value
            self.__state = FutureState.READY
            return True

    def __Fail(self, message: str):
        if not self.__IsDeferred():
            print("[Future] futrue state is not DEFER")
            return False
        else:
            self.__msg = message
            self.__state = FutureState.FAILED
            return True

    def __IsDeferred(self):
        return self.__state == FutureState.DEFER
    
    def __IsReady(self):
        return self.__state == FutureState.READY
    
    def __IsFailed(self):
        return self.__state == FutureState.FAILED