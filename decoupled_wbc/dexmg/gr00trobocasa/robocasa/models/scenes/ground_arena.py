from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa


class GroundArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(
            xml_path_completion("arenas/ground_arena.xml", root=robocasa.models.assets_root)
        )
