import math
import ctypes
from .clib_lookup import CLIBLookup

class timespec(ctypes.Structure):
    _fields_ = [("sec", ctypes.c_long), ("nsec", ctypes.c_long)]
    __slots__ = [name for name,type in _fields_]

    @classmethod
    def from_seconds(cls, secs):
        c = cls()
        c.seconds = secs
        return c
    
    @property
    def seconds(self):
        return self.sec + self.nsec / 1000000000

    @seconds.setter
    def seconds(self, secs):
        x, y = math.modf(secs)
        self.sec = int(y)
        self.nsec = int(x * 1000000000)


class itimerspec(ctypes.Structure):
    _fields_ = [("interval", timespec),("value", timespec)]
    __slots__ = [name for name,type in _fields_]
    
    @classmethod
    def from_seconds(cls, interval, value):
        spec = cls()
        spec.interval.seconds = interval
        spec.value.seconds = value
        return spec


# function timerfd_create
timerfd_create = CLIBLookup("timerfd_create", ctypes.c_int, (ctypes.c_long, ctypes.c_int))

# function timerfd_settime
timerfd_settime = CLIBLookup("timerfd_settime", ctypes.c_int, (ctypes.c_int, ctypes.c_int, ctypes.POINTER(itimerspec), ctypes.POINTER(itimerspec)))

# function timerfd_gettime
timerfd_gettime = CLIBLookup("timerfd_gettime", ctypes.c_int, (ctypes.c_int, ctypes.POINTER(itimerspec)))
