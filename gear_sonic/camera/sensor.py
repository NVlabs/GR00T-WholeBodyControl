"""Base sensor abstract class.

The ``gymnasium`` dependency is lazy-imported so the camera server can
run without it.
"""

from abc import abstractmethod
from typing import Any


class Sensor:
    """Base class for camera / sensor implementations.

    Concrete drivers (OAK, RealSense, ZED, USB, …) inherit from this and
    implement at least :meth:`read` and :meth:`serialize`.
    """

    def read(self, **kwargs) -> Any:
        """Read the current sensor value (e.g. a dict of images)."""

    def observation_space(self):
        """Return a ``gymnasium.Space`` describing the observation.

        Only used during init to report camera capabilities to the
        composed-camera orchestrator; not required for data collection.
        """

    @abstractmethod
    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Serialize the sensor reading for ZMQ transmission."""

    def close(self):
        """Release hardware resources."""
