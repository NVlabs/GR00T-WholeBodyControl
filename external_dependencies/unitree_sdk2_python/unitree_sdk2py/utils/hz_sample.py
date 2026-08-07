import time
from threading import Lock
from .thread import RecurrentThread

class HZSample:
    def __init__(self, interval: float = 1.0):
        self.__count = 0
        self.__inter = interval if interval > 0.0 else 1.0
        self.__lock = Lock()
        self.__thread = RecurrentThread(self.__inter, target=self.TimerFunc)

    def Start(self):
        self.__thread.Start()

    def Sample(self):
        with self.__lock:
            self.__count += 1

    def TimerFunc(self):
        count = 0
        with self.__lock:
            count = self.__count
            self.__count = 0
        print("HZ: {}".format(count/self.__inter))
