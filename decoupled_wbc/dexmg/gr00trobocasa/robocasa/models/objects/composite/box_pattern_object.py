import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
import robosuite.utils.transform_utils as T


class BoxPatternObject(CompositeObject):
    """
    Generates shapes by using a pattern of unit-size boxes.

    Args:
        name (str): Name of this Needle object
    """

    def __init__(
        self,
        name,
        unit_size,
        pattern,
        rgba=None,
        material=None,
        density=100.0,
        # solref=[0.02, 1.],
        # solimp=[0.9, 0.95, 0.001],
        friction=None,
    ):
        """
        Args:
            unit_size (3d array / list): size of each unit block in each dimension

            pattern (3d array / list): array of normalized sizes specifying the
                geometry of the shape. A "0" indicates the absence of a cube and
                a "1" indicates the presence of a full unit block. The dimensions
                correspond to z, x, and y respectively.
        """
        self._name = name
        self.rgba = rgba
        self.material = material
        self.density = density
        self.friction = friction

        # number of blocks in z, x, and y
        self.pattern = np.array(pattern)
        self.nz, self.nx, self.ny = self.pattern.shape
        self.unit_size = unit_size
        self.total_size = [
            self.nx * unit_size[0],
            self.ny * unit_size[1],
            self.nz * unit_size[2],
        ]

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        if self.material is not None:
            self.append_material(self.material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": self.total_size,
            "name": self.name,
            "locations_relative_to_center": False,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        geom_locations = []
        geom_sizes = []
        geom_names = []
        nz, nx, ny = self.pattern.shape
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    if self.pattern[k, i, j] > 0:
                        geom_sizes.append(
                            [
                                self.unit_size[0],
                                self.unit_size[1],
                                self.unit_size[2],
                            ]
                        )
                        geom_locations.append(
                            [
                                i * 2.0 * self.unit_size[0],
                                j * 2.0 * self.unit_size[1],
                                k * 2.0 * self.unit_size[2],
                            ]
                        )
                        geom_names.append("{}_{}_{}".format(k, i, j))

        # geom_rgbas = [rgba for _ in geom_locations]
        # geom_frictions = [friction for _ in geom_locations]
        for i in range(len(geom_locations)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                # needle geom needs to be offset from boundary in (x, z)
                geom_locations=tuple(geom_locations[i]),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=tuple(geom_sizes[i]),
                geom_names=geom_names[i],
                geom_rgbas=self.rgba,
                geom_materials=self.material.name if self.material is not None else None,
                geom_frictions=None,
            )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args
