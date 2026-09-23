from abc import abstractmethod

from decoupled_wbc.control.base.env import Env
from decoupled_wbc.control.base.sensor import Sensor
from decoupled_wbc.control.robot_model.robot_model import RobotModel


class Hands:
    """Container class for left and right hand environments.

    Attributes:
        left: Environment for the left hand
        right: Environment for the right hand
    """

    left: Env
    right: Env


class HumanoidEnv(Env):
    """Base class for humanoid robot environments.

    This class provides the interface for accessing the robot's body, hands, and sensors.
    """

    def body(self) -> Env:
        """Get the robot's body environment.

        Returns:
            Env: The body environment
        """
        pass

    def hands(self) -> Hands:
        """Get the robot's hands.

        Returns:
            Hands: Container with left and right hand environments
        """
        pass

    def sensors(self) -> dict[str, Sensor]:
        """Get the sensors of this environment

        Returns:
            dict: A dictionary of sensors
        """
        pass

    @abstractmethod
    def robot_model(self) -> RobotModel:
        """Get the robot model of this environment
        This robot model is used to dispatch whole body actions to body
        and hand actuators and to reconstruct proprioceptive
        observations from body and hands.

        Returns:
            RobotModel: The robot model
        """
        pass
