import gymnasium as gym


class Sensor:
    """Base class for implementing sensors in the Gr00t framework.

    A Sensor provides information about a specific sensor on the robot (e.g. camera, IMU,
    force sensor). This abstract base class defines the interface that all concrete sensor
    implementations must follow.
    """

    def read(self, **kwargs) -> any:
        """Read the current sensor value.

        Args:
            **kwargs: Additional parameters specific to the sensor implementation
                (e.g. camera resolution, sampling rate)

        Returns:
            The sensor reading value (e.g. image data, acceleration measurements)
        """
        pass

    def observation_space(self) -> gym.Space:
        """Get the observation space of this sensor.

        Returns:
            gym.Space: The observation space defining the shape and bounds of sensor readings
                (e.g. image dimensions for camera, measurement ranges for IMU)
        """
        pass

    def close(self):
        """Clean up any resources used by the sensor."""
        pass
