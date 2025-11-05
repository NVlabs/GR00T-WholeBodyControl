from dataclasses import dataclass
from cyclonedds.idl import IdlStruct

@dataclass
class HelloWorld(IdlStruct, typename="HelloWorld"):
    data: str