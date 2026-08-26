import time
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client

from test_api import *

"""
" class TestClient
"""
class TestClient(Client):
    def __init__(self, enableLease: bool = False):
        super().__init__("test", enableLease)

    def Init(self):
        self._RegistApi(TEST_API_ID_MOVE, 0)
        self._RegistApi(TEST_API_ID_STOP, 1)
        self._SetApiVerson(TEST_API_VERSION)

    def Move(self, vx: float, vy: float, vyaw: float):
        parameter = {}
        parameter["vx"] = vx
        parameter["vy"] = vy
        parameter["vyaw"] = vyaw
        p = json.dumps(parameter)

        c, d = self._Call(TEST_API_ID_MOVE, p)
        return c

    def Stop(self):
        parameter = {}
        p = json.dumps(parameter)
        
        c, d = self._Call(TEST_API_ID_STOP, p)
        return c

if __name__ ==  "__main__":
    # initialize channel factory.
    ChannelFactoryInitialize(0)

    # create client
    client = TestClient(True)
    client.Init()
    client.SetTimeout(5.0)

    # get server version
    code, serverApiVersion = client.GetServerApiVersion()
    print("server api version:", serverApiVersion)

    # wait lease applied
    client.WaitLeaseApplied()

    # test api
    while True:
        code = client.Move(0.2, 0, 0)
        print("client move ret:", code)
        time.sleep(1.0)

        code = client.Stop()
        print("client stop ret:", code)
        time.sleep(1.0)
