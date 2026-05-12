import time
import socket
import os
import json

from threading import Thread, Lock

from .client_base import ClientBase
from .internal import *


"""
" class LeaseContext
"""
class LeaseContext:
    def __init__(self):
        self.id = 0
        self.term = RPC_LEASE_TERM

    def Update(self, id, term):
        self.id = id
        self.term = term

    def Reset(self):
        self.id = 0
        self.term = RPC_LEASE_TERM

    def Valid(self):
        return self.id != 0


"""
" class LeaseClient
"""
class LeaseClient(ClientBase):
    def __init__(self, name: str):
        self.__name = name + "_lease"
        self.__contextName = socket.gethostname() + "/" + name + "/" + str(os.getpid())
        self.__context = LeaseContext()
        self.__thread = None
        self.__lock = Lock()
        super().__init__(self.__name)
        print("[LeaseClient] lease name:", self.__name, ", context name:", self.__contextName)
    
    def Init(self):
        self.SetTimeout(1.0)
        self.__thread = Thread(target=self.__ThreadFunc, name=self.__name, daemon=True)
        self.__thread.start()

    def WaitApplied(self):
        while True:
            with self.__lock:
                if self.__context.Valid():
                    break
            time.sleep(0.1)            
    
    def GetId(self):
            with self.__lock:
                return self.__context.id
    
    def Applied(self):
            with self.__lock:
                return self.__context.Valid()
    
    def __Apply(self):
        parameter = {}
        parameter["name"] = self.__contextName
        p = json.dumps(parameter)

        c, d = self._CallBase(RPC_API_ID_LEASE_APPLY, p)
        if c != 0:
            print("[LeaseClient] apply lease error. code:", c)
            return

        data = json.loads(d)
        
        id = data["id"]
        term = data["term"]

        print("[LeaseClient] lease applied id:", id, ", term:", term)

        with self.__lock:
            self.__context.Update(id, float(term/1000000))
    
    def __Renewal(self):
        parameter = {}
        p = json.dumps(parameter)

        c, d = self._CallBase(RPC_API_ID_LEASE_RENEWAL, p, 0, self.__context.id)
        if c != 0:
            print("[LeaseClient] renewal lease error. code:", c)
            if c == RPC_ERR_SERVER_LEASE_NOT_EXIST:
                with self.__lock:
                    self.__context.Reset()
    
    def __GetWaitSec(self):
        waitsec = 0.0
        if self.__context.Valid():
            waitsec = self.__context.term

        if waitsec <= 0:
            waitsec = RPC_LEASE_TERM

        return waitsec * 0.3

    def __ThreadFunc(self):
        while True:
            if self.__context.Valid():
                self.__Renewal()
            else:
                self.__Apply()
            # sleep waitsec 
            time.sleep(self.__GetWaitSec())
