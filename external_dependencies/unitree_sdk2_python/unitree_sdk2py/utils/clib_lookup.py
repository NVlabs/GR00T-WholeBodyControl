import os
import ctypes

clib = ctypes.CDLL(None, use_errno=True)

def CLIBCheckError(ret, func, args):
    if ret < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return ret

def CLIBLookup(name, resType, argTypes):
    func = clib[name]
    func.restye = resType
    func.argtypes = argTypes
    func.errcheck = CLIBCheckError
    return func
