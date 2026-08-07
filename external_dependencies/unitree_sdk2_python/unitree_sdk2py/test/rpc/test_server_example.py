import time
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.server import Server

from test_api import *


"""
" class TestServer
"""
class TestServer(Server):
    def __init__(self):
        super().__init__("test")

    def Init(self):
        self._RegistHandler(TEST_API_ID_MOVE, self.Move, 1)
        self._RegistHandler(TEST_API_ID_STOP, self.Stop, 0)
        self._SetApiVersion(TEST_API_VERSION)

    def Move(self, parameter: str):
        p = json.loads(parameter)
        x = p["vx"]
        y = p["vy"]
        yaw = p["vyaw"]
        print("Move Called. vx:", x, ", vy:", y, ", vyaw:", yaw)
        return 0, ""

    def Stop(self, parameter: str):
        print("Stop Called.")
        return 0, ""

if __name__ ==  "__main__":
    # initialize channel factory.
    ChannelFactoryInitialize(0)

    # create server
    server = TestServer()
    server.Init()
    server.StartLease(1.0)
    server.Start(False)

    while True:
        time.sleep(10)