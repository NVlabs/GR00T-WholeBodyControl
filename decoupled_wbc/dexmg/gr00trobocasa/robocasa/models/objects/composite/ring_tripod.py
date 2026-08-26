import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.utils.transform_utils as T


class RingTripodObject(CompositeObject):
    """
    Generates a tripod base with a small ring for threading a needle through it (used in Threading task)

    Args:
        name (str): Name of this RingTripod object
    """

    def __init__(
        self,
        name,
    ):

        ### TODO: make this object more general (with more args and configuration options) later ###

        # Set object attributes
        self._name = name
        self.tripod_mat_name = "lightwood_mat"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        tripod_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(tripod_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        total_size = (0.05, 0.05, 0.1)
        base_args = {
            "total_size": total_size,
            "name": self.name,
            "locations_relative_to_center": False,
            "obj_types": "all",
            "density": 100.0,
            # NOTE: this lower value of solref allows the thin hole wall to avoid penetration through it
            "solref": (0.02, 1.0),
            "solimp": (0.9, 0.95, 0.001),
        }
        obj_args = {}

        # pattern for threading ring
        unit_size = [0.005, 0.002, 0.002]
        pattern = np.ones((6, 1, 6))
        for i in range(1, 5):
            pattern[i][0][1:5] = np.zeros(4)
        ring_size = [
            unit_size[0] * pattern.shape[1],
            unit_size[1] * pattern.shape[2],
            unit_size[2] * pattern.shape[0],
        ]
        self.ring_size = np.array(ring_size)

        # ring offset for where the ring is located relative to the (0, 0, 0) corner
        ring_offset = [
            total_size[0] - ring_size[0],
            total_size[1] - ring_size[1],
            2.0 * (total_size[2] - ring_size[2]),
        ]

        # RING-GEOMS: use the pattern to instantiate geoms corresponding to the threading ring
        nz, nx, ny = pattern.shape
        self.num_ring_geoms = 0
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    if pattern[k, i, j] > 0:
                        add_to_dict(
                            dic=obj_args,
                            geom_types="box",
                            # needle geom needs to be offset from boundary in (x, z)
                            geom_locations=(
                                (i * 2.0 * unit_size[0]) + ring_offset[0],
                                (j * 2.0 * unit_size[1]) + ring_offset[1],
                                (k * 2.0 * unit_size[2]) + ring_offset[2],
                            ),
                            geom_quats=(1, 0, 0, 0),
                            geom_sizes=tuple(unit_size),
                            geom_names="ring_{}".format(self.num_ring_geoms),
                            geom_rgbas=None,
                            geom_materials=self.tripod_mat_name,
                            # make the ring low friction to ensure easy insertion
                            geom_frictions=(0.3, 5e-3, 1e-4),
                        )
                        self.num_ring_geoms += 1

        # TRIPOD-GEOMS: legs of the tripod
        tripod_capsule_r = 0.01
        tripod_capsule_h = 0.03
        tripod_geom_locations = [
            (0.0, 0.0, 0.0),
            (0.0, 2.0 * total_size[1] - 2.0 * tripod_capsule_r, 0.0),
            (
                2.0 * total_size[0] - 2.0 * tripod_capsule_r,
                total_size[1] - tripod_capsule_r,
                0.0,
            ),
        ]
        # rotate the legs to resemble a tripod
        tripod_center = np.array([total_size[0], total_size[1], 0.0])
        xy_offset = np.array([tripod_capsule_r, tripod_capsule_r, 0.0])
        rotation_angle = -np.pi / 6.0  # 30 degrees
        tripod_geom_quats = []
        for i in range(3):
            capsule_loc = np.array(tripod_geom_locations[i]) + xy_offset
            capsule_loc[2] = 0.0  # only care about location in x-y plane
            vec_to_center = tripod_center - capsule_loc
            vec_to_center = vec_to_center / np.linalg.norm(vec_to_center)
            # cross-product with z unit vector to get vector to rotate about
            rot_vec = np.cross(vec_to_center, np.array([0.0, 0.0, 1.0]))
            rot_quat = T.mat2quat(T.rotation_matrix(angle=rotation_angle, direction=rot_vec))
            tripod_geom_quats.append(T.convert_quat(rot_quat, to="wxyz"))

        for i in range(3):
            add_to_dict(
                dic=obj_args,
                geom_types="capsule",
                geom_locations=tripod_geom_locations[i],
                geom_quats=tripod_geom_quats[i],
                geom_sizes=(tripod_capsule_r, tripod_capsule_h),
                geom_names="tripod_{}".format(i),
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )

        # POST-GEOMS: mounted base + post
        base_thickness = 0.005
        post_size = 0.005
        post_geom_sizes = [
            (total_size[0], total_size[1], base_thickness),
            (
                post_size,
                post_size,
                total_size[2] - ring_size[2] - base_thickness - tripod_capsule_r - tripod_capsule_h,
            ),
        ]
        post_geom_locations = [
            (0.0, 0.0, 2.0 * (tripod_capsule_r + tripod_capsule_h)),
            (
                total_size[0] - post_size,
                total_size[1] - post_size,
                2.0 * (tripod_capsule_r + tripod_capsule_h + base_thickness),
            ),
        ]
        for i in range(2):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=post_geom_locations[i],
                geom_quats=(1, 0, 0, 0),
                geom_sizes=post_geom_sizes[i],
                geom_names="post_{}".format(i),
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args
