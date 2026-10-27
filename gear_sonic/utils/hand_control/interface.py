"""Target/feedback API shared by the Inspire simulation and hardware adapters.

Other hands may reuse this API if it fits their implementation; inheriting it
is not a deployment requirement. Dex3 keeps its existing C++/DDS implementation.
See README.md for the project locations involved in adding a hand.

This module defines the interface only. It does not open devices, transmit
commands, apply limits, or assume a particular hand's number of joints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HandDescription:
    """Ordered control channels; both positions and targets use these units.

    Limits and actuator conversion belong to the selected backend's configuration.
    Channels may be coupled finger coordinates rather than every physical joint.
    """

    model: str
    left_joint_names: tuple[str, ...]
    right_joint_names: tuple[str, ...]
    position_unit: str = "rad"


@dataclass(frozen=True)
class HandState:
    """Latest measured feedback, never a fabricated copy of the target.

    None means unavailable. received_monotonic_ns is the time of receipt in
    this process, not a timestamp from a remote machine. Backends must document
    whether left/right measurements are simultaneous and how stale data is handled.
    """

    left_position: tuple[float, ...] | None = None
    right_position: tuple[float, ...] | None = None
    received_monotonic_ns: int | None = None
    error: str | None = None


class HandBackend(ABC):
    """Interface for a pair of hands in simulation or on hardware.

    Calls are serialized by the owner; thread safety is not implied. Constructing
    a backend must not connect or move hardware. The backend owns limit/rate
    checks and communication failures; these are not implemented by this base.
    """

    @property
    @abstractmethod
    def description(self) -> HandDescription:
        """Return the model, command channel order and position unit."""

    @abstractmethod
    def connect(self) -> None:
        """Initialize resources without issuing an unsolicited motion command.

        On failure, release resources acquired during this attempt and raise.
        """

    @abstractmethod
    def set_target(self, left: Sequence[float], right: Sequence[float]) -> None:
        """Submit positions in description order after connecting.

        Validate both sides' length, finite values and configured limits before
        sending either side. A normal return means submission, not physical
        arrival or a simultaneous two-hand hardware transaction. Raise on invalid
        input or disconnected/stopped state. Backends must bound queued commands.
        """

    @abstractmethod
    def read_state(self) -> HandState:
        """Return feedback after connecting, with absent measurements as None.

        Implementations must document timeout and feedback freshness behavior.
        """

    @abstractmethod
    def stop(self) -> None:
        """Cancel pending targets and reject further targets until reconnected.

        Document device behavior (e.g. hold last position); this is not an
        emergency stop or a guarantee that motors are de-energized. No automatic
        open/close gesture. Repeated calls must be safe.
        """

    @abstractmethod
    def close(self) -> None:
        """Cancel pending work and release resources, including after failures.

        Repeated calls must be safe. No implicit open/close gesture. The backend
        must document what the physical hand does after communication closes.
        """
