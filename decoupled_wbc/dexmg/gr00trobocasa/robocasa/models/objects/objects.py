import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.environments.robot_env import RobotEnv


class MJCFObject(MujocoXMLObject):
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
        priority=None,
        static=False,
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

        self.priority = priority

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = self.postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()

        # initialize object with new xml we wrote
        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=None if static else [dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,
        )

        self.spawns = []
        self._disabled_spawns = set()
        for s in self.worldbody.findall("./body/body/geom"):
            geom_name = s.get("name")
            if geom_name and geom_name.startswith("{}spawn_".format(self.naming_prefix)):
                self.spawns.append(s)

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)

    def postprocess_model_xml(self, xml_str):
        """
        New version of postprocess model xml that only replaces robosuite file paths if necessary (otherwise
        there is an error with the "max" operation)
        """

        path = os.path.split(robosuite.__file__)[0]
        path_split = path.split("/")

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
            if len(check_lst) > 0:
                ind = max(check_lst)  # last occurrence index
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

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
        geom_pairs = super(MJCFObject, self)._get_geoms(root=root, _parent=_parent)

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

            if self.priority is not None:
                # set high priorit
                element.set("priority", str(self.priority))

        return geom_pairs

    def get_joint(self, joint_name: str):
        _, _, joints = self._get_elements_by_name(
            geom_names=[], body_names=[], joint_names=[joint_name]
        )
        return joints[joint_name]

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = string_to_array(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    def get_bbox_points(self, trans=None, rot=None) -> list[np.ndarray]:
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bottom_offset = self.bottom_offset
        top_offset = self.top_offset
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        horiz_radius = string_to_array(horizontal_radius_site.get("pos"))[:2]
        return self._get_bbox_points(
            bottom_offset=bottom_offset,
            top_offset=top_offset,
            radius=horiz_radius,
            trans=trans,
            rot=rot,
        )

    @staticmethod
    def _get_bbox_points(
        bottom_offset, top_offset, radius, trans=None, rot=None
    ) -> list[np.ndarray]:
        """
        Helper function to get the full 8 bounding box points of the object.
        """
        center = np.mean([bottom_offset, top_offset], axis=0)
        half_size = [radius[0], radius[1], top_offset[2] - center[2]]

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),  # p0
            center + half_size * np.array([1, -1, -1]),  # px
            center + half_size * np.array([-1, 1, -1]),  # py
            center + half_size * np.array([-1, -1, 1]),  # pz
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    @staticmethod
    def get_spawn_bottom_offset(site: ET.Element) -> np.array:
        """
        Get bottom offset of the spawn zone.
        """
        site_pos = string_to_array(site.get("pos"))
        site_size = string_to_array(site.get("size")) if site.get("type") == "box" else np.zeros(3)
        return site_pos - np.array([0, 0, site_size[-1]])

    def get_random_spawn(self, rng, exclude_disabled: bool = False) -> tuple[int, ET.Element]:
        """
        Get random spawn site.
        """
        options = [o for o in range(0, len(self.spawns))]
        if exclude_disabled:
            options = [o for o in options if o not in self._disabled_spawns]
        spawn_id = rng.choice(options)
        return spawn_id, self.spawns[spawn_id]

    def set_spawn_active(self, spawn_id: int, active: bool):
        """
        Update the activity state of a spawn site. Disabled sites are excluded from random sampling.
        """
        if active and spawn_id in self._disabled_spawns:
            self._disabled_spawns.remove(spawn_id)
        elif not active:
            self._disabled_spawns.add(spawn_id)

    def closest_spawn_id(self, env: RobotEnv, obj: "MJCFObject", max_distance: float = 1.0) -> int:
        if len(self.spawns) == 0:
            return -1
        if not env.check_contact(self, obj):
            return -1
        obj_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)]
        distances = []
        for spawn_id in range(len(self.spawns)):
            spawn_pos = env.sim.data.get_geom_xpos(self.spawns[spawn_id].get("name"))
            distance = np.linalg.norm(spawn_pos - obj_pos)
            distances.append((spawn_id, distance))
        distances = sorted(distances, key=lambda item: item[1])
        obj_geom_ids = [env.sim.model.geom_name2id(g) for g in obj.contact_geoms]
        for spawn_id, distance in distances:
            spawn_geom_id = env.sim.model.geom_name2id(self.spawns[spawn_id].get("name"))
            for obj_geom_id in obj_geom_ids:
                real_distance = mujoco.mj_geomDistance(
                    m=env.sim.model._model,
                    d=env.sim.data._data,
                    geom1=spawn_geom_id,
                    geom2=obj_geom_id,
                    distmax=max_distance,
                    fromto=None,
                )
                if real_distance <= 0:
                    return spawn_id
                if real_distance >= max_distance:
                    return -1
        return -1
