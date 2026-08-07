from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
import robocasa


class FactoryArena(Arena):
    """Factory workspace."""

    def __init__(self):
        super().__init__(
            xml_path_completion(
                "arenas/gear_factory/gear_factory.xml", root=robocasa.models.assets_root
            )
        )
