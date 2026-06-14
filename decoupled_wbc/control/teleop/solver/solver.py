from abc import ABC, abstractmethod
from typing import Any


class Solver(ABC):
    def __init__(self):
        pass

    def register_robot(self, robot):
        pass

    def calibrate(self, data):
        pass

    @abstractmethod
    def __call__(self, target) -> Any:
        pass
