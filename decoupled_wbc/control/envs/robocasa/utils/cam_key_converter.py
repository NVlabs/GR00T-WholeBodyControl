from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from decoupled_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH


@dataclass
class CameraConfig:
    width: int
    height: int
    mapped_key: str


class CameraKeyMapper:
    def __init__(self):
        # Default camera dimensions
        self.default_width = RS_VIEW_CAMERA_WIDTH
        self.default_height = RS_VIEW_CAMERA_HEIGHT

        # Camera key mapping with custom dimensions
        self.camera_configs: Dict[str, CameraConfig] = {
            # GR1
            "egoview": CameraConfig(self.default_width, self.default_height, "ego_view"),
            "frontview": CameraConfig(self.default_width, self.default_height, "front_view"),
            # G1
            "robot0_rs_egoview": CameraConfig(self.default_width, self.default_height, "ego_view"),
            "robot0_rs_tppview": CameraConfig(self.default_width, self.default_height, "tpp_view"),
            "robot0_oak_egoview": CameraConfig(self.default_width, self.default_height, "ego_view"),
            "robot0_oak_left_monoview": CameraConfig(
                self.default_width, self.default_height, "ego_view_left_mono"
            ),
            "robot0_oak_right_monoview": CameraConfig(
                self.default_width, self.default_height, "ego_view_right_mono"
            ),
        }

    def get_camera_config(self, key: str) -> Optional[Tuple[str, int, int]]:
        """
        Get the mapped camera key and dimensions for a given camera key.

        Args:
            key: The input camera key

        Returns:
            Tuple of (mapped_key, width, height) if key exists, None otherwise
        """
        config = self.camera_configs.get(key.lower())
        if config is None:
            return None
        return config.mapped_key, config.width, config.height

    def add_camera_config(
        self, key: str, mapped_key: str, width: int = 256, height: int = 256
    ) -> None:
        """
        Add a new camera configuration or update an existing one.

        Args:
            key: The camera key to add/update
            mapped_key: The actual camera key to map to
            width: Camera width in pixels
            height: Camera height in pixels
        """
        self.camera_configs[key.lower()] = CameraConfig(width, height, mapped_key)

    def get_all_camera_keys(self) -> list:
        """
        Get all available camera keys.

        Returns:
            List of all camera keys
        """
        return list(self.camera_configs.keys())
