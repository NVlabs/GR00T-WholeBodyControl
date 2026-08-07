from robosuite.models.objects import CompositeBodyObject, BoxObject, Bin
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class BinWithHandles(CompositeBodyObject):
    """
    Bin with simple square handles on each side.
    """

    def __init__(
        self,
        name,
        bin_size,
        bin_wall_thickness,
        bin_transparent_walls,
        bin_upside_down,
        center_handle_size,
        adjacent_handle_size,
        joints="default",
        rgba=(0.2, 0.1, 0.0, 1.0),
        material=None,
        density=1000.0,
        friction=None,
    ):

        # Object properties

        # FULL size of bin
        self.bin_size = list(bin_size)
        self.bin_wall_thickness = bin_wall_thickness
        self.bin_transparent_walls = bin_transparent_walls
        self.bin_upside_down = bin_upside_down

        # half-sizes of box geom used for center part of handle (which you grab)
        self.center_handle_size = list(center_handle_size)

        # half-sizes of box geoms used for adjacent parts of handle (not grabbed)
        self.adjacent_handle_size = list(adjacent_handle_size)

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # bin
        self.bin = Bin(
            name="bin",
            bin_size=self.bin_size,
            wall_thickness=self.bin_wall_thickness,
            transparent_walls=self.bin_transparent_walls,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            upside_down=bin_upside_down,
        )
        objects.append(self.bin)
        object_locations.append([0.0, 0.0, 0.0])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # handles on each side

        left_handle_1_loc = [
            0.0,
            -(
                self.bin_size[1] / 2.0
                + 2.0 * self.adjacent_handle_size[1]
                + self.center_handle_size[1]
            ),
            0.0,
        ]
        left_handle_1_size = self.center_handle_size

        left_handle_2_loc = [
            (self.center_handle_size[0] - self.adjacent_handle_size[0]),
            -(self.bin_size[1] / 2.0 + self.adjacent_handle_size[1]),
            0.0,
        ]
        left_handle_2_size = self.adjacent_handle_size

        left_handle_3_loc = [
            -(self.center_handle_size[0] - self.adjacent_handle_size[0]),
            -(self.bin_size[1] / 2.0 + self.adjacent_handle_size[1]),
            0.0,
        ]
        left_handle_3_size = self.adjacent_handle_size

        right_handle_1_loc = [
            0.0,
            (
                self.bin_size[1] / 2.0
                + 2.0 * self.adjacent_handle_size[1]
                + self.center_handle_size[1]
            ),
            0.0,
        ]
        right_handle_1_size = self.center_handle_size

        right_handle_2_loc = [
            (self.center_handle_size[0] - self.adjacent_handle_size[0]),
            (self.bin_size[1] / 2.0 + self.adjacent_handle_size[1]),
            0.0,
        ]
        right_handle_2_size = self.adjacent_handle_size

        right_handle_3_loc = [
            -(self.center_handle_size[0] - self.adjacent_handle_size[0]),
            (self.bin_size[1] / 2.0 + self.adjacent_handle_size[1]),
            0.0,
        ]
        right_handle_3_size = self.adjacent_handle_size

        handle_locs = [
            left_handle_1_loc,
            left_handle_2_loc,
            left_handle_3_loc,
            right_handle_1_loc,
            right_handle_2_loc,
            right_handle_3_loc,
        ]
        handle_sizes = [
            left_handle_1_size,
            left_handle_2_size,
            left_handle_3_size,
            right_handle_1_size,
            right_handle_2_size,
            right_handle_3_size,
        ]
        handle_ind = 1
        for b_loc, b_size in zip(handle_locs, handle_sizes):
            this_handle = BoxObject(
                name="handle_{}".format(handle_ind),
                size=b_size,
                rgba=rgba,
                material=material,
                density=density,
                friction=friction,
                joints=None,
            )
            objects.append(this_handle)
            object_locations.append(b_loc)
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(None)
            handle_ind += 1

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            # total_size=body_total_size,
            # locations_relative_to_corner=True,
        )
