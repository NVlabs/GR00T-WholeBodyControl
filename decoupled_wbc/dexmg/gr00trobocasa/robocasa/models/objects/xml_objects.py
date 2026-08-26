import os
import time
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, string_to_array
import robocasa

XML_ASSETS_BASE_PATH = os.path.join(robocasa.__path__[0], "models/assets/gear_kitchen")


class BlenderObject(MujocoXMLObject):
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # modify mesh scales
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        for mesh in meshes:
            # if a scale already exists, multiply the scales
            scale_to_set = scale
            existing_scale = mesh.get("scale")
            if existing_scale is not None:
                scale_to_set = string_to_array(existing_scale) * scale
            mesh.set("scale", array_to_string(scale_to_set))

        # modify sites for collision (assumes we can just scale up the locations - may or may not work)
        for n in ["bottom_site", "top_site", "horizontal_radius_site"]:
            site = root.find("worldbody/body/site[@name='{}']".format(n))
            pos = string_to_array(site.get("pos"))
            pos = scale * pos
            site.set("pos", array_to_string(pos))

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        # xml_str = postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()
        # print(f"Write to {new_xml_path}")

        # initialize object with new xml we wrote
        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        geom_pairs = super(BlenderObject, self)._get_geoms(root=root, _parent=_parent)

        # modify geoms according to the attributes
        for i, (parent, element) in enumerate(geom_pairs):
            element.set("solref", array_to_string(self.solref))
            element.set("solimp", array_to_string(self.solimp))
            element.set("density", str(self.density))
            element.set("friction", array_to_string(self.friction))
            if self.margin is not None:
                element.set("margin", str(self.margin))

            if (self.rgba is not None) and (element.get("group") == "1"):
                element.set("rgba", array_to_string(self.rgba))

        return geom_pairs


class CoffeeMachinePodObject(MujocoXMLObject):
    """
    Coffee pod object (used in Coffee task).
    """

    def __init__(self, name, margin=1.0):
        self._horizontal_radius_margin = margin
        super().__init__(
            os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_pod.xml"),
            # name=name, joints=[dict(type="free", damping="0.0005")],
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return (
            self._horizontal_radius_margin * string_to_array(horizontal_radius_site.get("pos"))[0]
        )


class CoffeeMachineBodyObject(MujocoXMLObject):
    """
    Coffee machine body piece (used in Coffee task). Note that the piece is rigid (no joint is added).
    """

    def __init__(self, name):
        super().__init__(
            os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_body.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CoffeeMachineLidObject(MujocoXMLObject):
    """
    Coffee machine lid piece (used in Coffee task).
    """

    def __init__(self, name):
        super().__init__(
            os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_lid.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class CoffeeMachineBaseObject(MujocoXMLObject):
    """
    Coffee machine base piece (used in Coffee task). Note that the piece is rigid (no joint is added).
    """

    def __init__(self, name):
        super().__init__(
            os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_base.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class DrawerObject(MujocoXMLObject):
    """
    Custom version of cabinet object that differs from BUDs. It has manually specified top, bottom, and horizontal sites.
    """

    def __init__(self, name, joints=None):

        # original cabinet xml
        # path_to_cabinet_xml = os.path.join(robosuite_task_zoo.__path__[0], "models/hammer_place/cabinet.xml")

        # our custom cabinet xml - names the drawer bottom geom for easy contact checking
        # path_to_cabinet_xml = os.path.join(mimicgen.__path__[0], "envs/robosuite/assets/cabinet.xml")
        path_to_cabinet_xml = os.path.join(XML_ASSETS_BASE_PATH, "objects/cabinet.xml")

        # path to modified xml from Danfei's student
        # path_to_cabinet_xml = os.path.join(XML_ASSETS_BASE_PATH, "objects/cabinet1.xml")

        super().__init__(
            path_to_cabinet_xml,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    # NOTE: had to manually set these to get placement sampler working okay
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.065])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.065])

    @property
    def horizontal_radius(self):
        return 0.15

    @property
    def body_size(self):
        return (0.5, 0.4, 0.3)


class LongDrawerObject(MujocoXMLObject):
    """
    Drawer that has longer platform for easier grasping of objects inside.
    """

    def __init__(self, name, joints=None):

        # our custom cabinet xml - has some longer geoms for the drawer platform
        path_to_cabinet_xml = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer_long.xml")

        super().__init__(
            path_to_cabinet_xml,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    # NOTE: had to manually set these to get placement sampler working okay
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.065])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.065])

    @property
    def horizontal_radius(self):
        # TODO: this might need to change now
        return 0.15
