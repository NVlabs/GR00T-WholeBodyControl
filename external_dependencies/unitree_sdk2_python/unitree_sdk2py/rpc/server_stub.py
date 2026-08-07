import time

from enum import Enum
from threading import Thread, Condition
from typing import Callable, Any

from ..utils.bqueue import BQueue
from ..idl.unitree_api.msg.dds_ import Request_ as Request
from ..idl.unitree_api.msg.dds_ import Response_ as Response

from ..core.channel import ChannelFactory
from ..core.channel_name import ChannelType, GetServerChannelName


"""
" class ServerStub
"""
class ServerStub:
    def __init__(self, serviceName: str):
        self.__serviceName = serviceName
        self.__serverRquestHandler = None
        self.__sendChannel = None
        self.__recvChannel = None
        self.__enablePriority = None
        self.__queue = None
        self.__prioQueue = None
        self.__queueThread = None
        self.__prioQueueThread = None

    def Init(self, serverRequestHander: Callable, enablePriority: bool = False):
        self.__serverRquestHandler = serverRequestHander
        self.__enablePriority = enablePriority

        factory = ChannelFactory()

        # create channel
        self.__sendChannel = factory.CreateSendChannel(GetServerChannelName(self.__serviceName, ChannelType.SEND), Response)
        self.__recvChannel = factory.CreateRecvChannel(GetServerChannelName(self.__serviceName, ChannelType.RECV), Request, self.__Enqueue, 10)

        # start priority request thread
        self.__queue = BQueue(10)
        self.__queueThread = Thread(target=self.__QueueThreadFunc, name="server_queue", daemon=True)
        self.__queueThread.start()
        
        if enablePriority:
            self.__prioQueue = BQueue(5)
            self.__prioQueueThread = Thread(target=self.__PrioQueueThreadFunc, name="server_prio_queue", daemon=True)
            self.__prioQueueThread.start()

        # wait thread started
        time.sleep(0.5)

    def Send(self, response: Response, timeout: float):
        if self.__sendChannel.Write(response, timeout):
            return True
        else:
            print("[ServerStub] send error. id:", response.header.identity.id)
            return False

    def __Enqueue(self, request: Request):
        if self.__enablePriority and request.header.policy.priority > 0:
            self.__prioQueue.Put(request, True)
        else:
            self.__queue.Put(request, True)

    def __QueueThreadFunc(self):
        while True:
            request = self.__queue.Get()
            if request is None:
                continue
            self.__serverRquestHandler(request)

    def __PrioQueueThreadFunc(self):
        while True:
            request = self.__prioQueue.Get()
            if request is None:
                continue
            self.__serverRquestHandler(request)
