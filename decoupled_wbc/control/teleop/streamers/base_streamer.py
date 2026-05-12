from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict


@dataclass
class StreamerOutput:
    """Clean separation of different data types"""

    # Data that needs IK processing (ik_keys)
    ik_data: Dict[str, Any] = field(default_factory=dict)

    # Commands that pass directly to robot control loop (control_keys)
    control_data: Dict[str, Any] = field(default_factory=dict)

    # Commands used internally by teleop policy (teleop_keys)
    teleop_data: Dict[str, Any] = field(default_factory=dict)

    # Commands used for data collection (data_collection_keys)
    data_collection_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class BaseStreamer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def reset_status(self):
        pass

    @abstractmethod
    def start_streaming(self):
        pass

    @abstractmethod
    def get(self) -> StreamerOutput:
        """Return StreamerOutput with structured data"""
        pass

    @abstractmethod
    def stop_streaming(self):
        pass
