from typing import Any
from collections import deque
from threading import Condition

class BQueue:
    def __init__(self, maxLen: int = 10):
        self.__curLen = 0
        self.__maxLen = maxLen
        self.__queue = deque()
        self.__condition = Condition()

    def Put(self, x: Any, replace: bool = False):
        noReplaced = True
        with self.__condition:
            if self.__curLen >= self.__maxLen:
                if not replace:
                    return False
                else:
                    noReplaced = False
                    self.__queue.popleft()
                    self.__curLen -= 1

            self.__queue.append(x)
            self.__curLen += 1
            self.__condition.notify()

            return noReplaced

    def Get(self, timeout: float = None):
        with self.__condition:
            if not self.__queue:
                try:
                    self.__condition.wait(timeout)
                except:
                    return None

                if not self.__queue:
                    return None
    
            self.__curLen -= 1
            return self.__queue.popleft()

    def Clear(self):
        with self.__condition:
            if self.__queue:
                self.__queue.clear()
                self.__curLen = 0

    def Size(self):
        with self.__condition:
            return self.__curLen

    def Interrupt(self, notifyAll: bool = False):
        with self.__condition:
            if notifyAll:
                self.__condition.notify()
            else:
                self.__condition.notify_all()
