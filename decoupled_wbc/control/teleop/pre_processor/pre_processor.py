import abc


class PreProcessor(abc.ABC):
    def __init__(self, **kwargs):
        pass

    def register(self, robot):
        self.robot = robot

    @abc.abstractmethod
    def __call__(self, data) -> dict:
        pass
