import numpy as np

from robosuite.models.objects import BoxObject, CompositeBodyObject, CylinderObject
from robosuite.utils.mjcf_utils import BLUE, RED, CustomMaterial, array_to_string


class SlidingBoxObject(CompositeBodyObject):
    """
    An example object that demonstrates the CompositeBodyObject functionality. This object consists of two cube bodies
    joined together by a slide joint allowing one box to slide on top of the other.

    Args:
        name (str): Name of this object

        box1_size (3-array): (L, W, H) half-sizes for the first box

        box2_size (3-array): (L, W, H) half-sizes for the second box

        use_texture (bool): set True if using wood textures for the blocks
    """

    def __init__(
        self,
        name,
        box1_size=(0.1, 0.1, 0.02),
        box2_size=(0.02, 0.02, 0.02),
        use_texture=True,
    ):
        # Set box sizes
        self.box1_size = np.array(box1_size)
        self.box2_size = np.array(box2_size)

        # Set box densities
        self.box1_density = 10000.0
        self.box2_density = 100.0

        # Set texture attributes
        self.use_texture = use_texture
        self.box1_material = None
        self.box2_material = None
        self.box1_rgba = RED
        self.box2_rgba = BLUE

        # Define materials we want to use for this object
        if self.use_texture:
            # Remove RGBAs
            self.box1_rgba = None
            self.box2_rgba = None

            # Set materials for each box
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            }
            self.box1_material = CustomMaterial(
                texture="WoodRed",
                tex_name="box1_tex",
                mat_name="box1_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.box2_material = CustomMaterial(
                texture="WoodBlue",
                tex_name="box2_tex",
                mat_name="box2_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )

        # Create objects
        objects = []
        for i, (size, mat, rgba, density) in enumerate(
            zip(
                (self.box1_size, self.box2_size),
                (self.box1_material, self.box2_material),
                (self.box1_rgba, self.box2_rgba),
                (self.box1_density, self.box2_density),
            )
        ):
            objects.append(
                BoxObject(
                    name=f"box{i + 1}",
                    size=size,
                    rgba=rgba,
                    material=mat,
                )
            )

        # Define slide joint
        rel_joint_pos = [0, 0, 0]  # at second box
        joint_lim = self.box1_size[1] - self.box2_size[1]
        slide_joint = {
            "name": "box_slide",
            "type": "slide",
            "axis": "0 1 0",  # y-axis slide
            "pos": array_to_string(rel_joint_pos),
            "springref": "0",
            "springdamper": "0.1 1.0",  # mass-spring system with 0.1 time constant, 1.0 damping ratio
            "limited": "true",
            "range": "{} {}".format(-joint_lim, joint_lim),
        }

        # Define positions -- second box should lie on top of first box
        positions = [
            np.zeros(3),  # First box is centered at top-level body anyways
            np.array([0, 0, self.box1_size[2] + self.box2_size[2]]),
        ]

        quats = [
            None,  # Default quaternion for box 1
            None,  # Default quaternion for box 2
        ]

        # Define parents -- which body each is aligned to
        parents = [
            None,  # box 1 attached to top-level body
            objects[0].root_body,  # box 2 attached to box 1
        ]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            body_joints={objects[1].root_body: [slide_joint]},
        )
