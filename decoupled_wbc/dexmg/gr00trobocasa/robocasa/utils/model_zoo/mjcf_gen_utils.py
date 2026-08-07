import numpy as np
import trimesh
import os
import robosuite
import xml.etree.ElementTree as ET
import shutil
import json
from termcolor import colored
from copy import deepcopy
from pathlib import Path

# import robosuite_model_zoo
from robosuite.utils.transform_utils import mat2quat
import robocasa.utils.model_zoo.log_utils as LogUtils
import robocasa.utils.model_zoo.mtl_utils as MtlUtils

from robosuite.utils.mjcf_utils import array_to_string, string_to_array


def parse_model_info(
    model_path,
    model_name,
    coll_model_path,
    asset_path,
    obj_path=None,
    verbose=False,
    center=False,
    prescale=False,
    rot=None,
    transform=None,
):
    """
    Extract visual and collision geoms from
    main model file and collision model file
    """

    LogUtils.maybe_log_info("Parsing input models...", log=verbose)

    geoms, textures, materials, transform = parse_model(
        model_path,
        model_name,
        asset_path,
        obj_path=obj_path,
        center=center,
        prescale=prescale,
        rot=rot,
        transform=transform,
        # visual_only=(coll_model_path is None),
        visual_only=True,
        verbose=verbose,
    )

    coll_geoms, bb_info = parse_coll_model(
        coll_model_path,
        asset_path,
        verbose=verbose,
        transform=transform,
    )

    if coll_geoms is not None:
        geoms = geoms + coll_geoms

    geoms_to_print = deepcopy(geoms)
    for g in geoms_to_print:
        if "rot_orig" in g:
            del g["rot_orig"]
    LogUtils.maybe_log_info(
        "parsed geoms:\n" + json.dumps(geoms_to_print, indent=4, cls=LogUtils.NumpyEncoder),
        log=verbose,
        indent=True,
    )

    model_info = dict(
        geoms=geoms,
        textures=textures,
        materials=materials,
        bb=bb_info,
    )

    return model_info


def parse_model(
    model_path,
    model_name,
    asset_path,
    obj_path=None,
    visual_only=False,
    center=False,
    prescale=False,
    rot=None,
    transform=None,
    verbose=False,
):
    """
    Extract meshes from main model file
    """

    LogUtils.maybe_log_info(
        "parsing (main) model: {}".format(model_path),
        log=verbose,
        indent=True,
    )

    # load the model
    if obj_path is None:
        obj_files = [
            os.path.join(model_path, name)
            for name in os.listdir(model_path)
            if name.endswith(".obj")
        ]
        obj_file = obj_files[0]
    else:
        obj_file = obj_path

    geoms, textures, materials = [], [], []
    vis_path = os.path.join(asset_path, "visual")
    coll_path = os.path.join(asset_path, "collision")

    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    if not os.path.isdir(coll_path):
        os.makedirs(coll_path)

    resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(obj_file))
    model = trimesh.load(
        obj_file,
        resolver=resolver,
        split_object=True,
        # group_material=True,
        process=False,
        maintain_order=False,
    )

    # If transform is passed as an argument, center and prescale must be False. Rotation is applied afterward regardless.
    if transform is not None:
        assert center is False and prescale is False

    if center:
        mat_t = np.eye((4))
        center = (model.bounds[0] + model.bounds[1]) / 2
        mat_t[:3, 3] = -center
        transform = mat_t

    if prescale:
        mat_s = np.eye((4))
        scale_factor = 1 / np.max(model.bounds[1] - model.bounds[0])
        mat_s *= scale_factor

        if transform is None:
            transform = mat_s
        else:
            transform = np.matmul(mat_s, transform)

    rot = rot or []
    for r in rot:
        mat_R = np.eye(4)

        if r in ["x", "x90"]:
            mat_R[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        elif r in ["x180"]:
            mat_R[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        elif r in ["x270"]:
            mat_R[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

        elif r in ["y", "y90"]:
            mat_R[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        elif r in ["y180"]:
            mat_R[:3, :3] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        elif r in ["y270"]:
            mat_R[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

        elif r in ["z", "z90"]:
            mat_R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        elif r in ["z180"]:
            mat_R[:3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif r in ["z270"]:
            mat_R[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        elif r in ["z60"]:
            mat_R[:3, :3] = [
                [0.5000000, -0.8660254, 0.0000000],
                [0.8660254, 0.5000000, 0.0000000],
                [0, 0, 1],
            ]

        else:
            raise ValueError("Invalid choice of rotation {}".format(r))

        if transform is None:
            transform = mat_R
        else:
            transform = np.matmul(mat_R, transform)

    if transform is not None:
        if isinstance(model, trimesh.base.Trimesh):
            model.apply_transform(transform)
        else:
            for k in model.geometry:
                model.geometry[k].apply_transform(transform)

    # debugging
    # model.show()
    # exit()

    mtls = MtlUtils.get_mtls(obj_file, work_dir=vis_path)

    """
    save mtl info. this section adapted from obj2mjf (credit: Kevin Zakka)
    """
    for mtl in mtls:
        if mtl.map_Kd is not None:
            # Create the texture asset.
            texture_path = Path(mtl.map_Kd)
            texture = dict(
                type="2d",
                name=texture_path.stem,
                file=os.path.join("visual", texture_path.name),
            )
            # Reference the texture asset in a material asset.
            material = dict(
                name=mtl.name,
                texture=texture_path.stem,
                specular=mtl.mjcf_specular(),
                shininess=mtl.mjcf_shininess(),
            )

            textures.append(texture)
            materials.append(material)
        else:
            material = dict(
                name=mtl.name,
                specular=mtl.mjcf_specular(),
                shininess=mtl.mjcf_shininess(),
                rgba=mtl.mjcf_rgba(),
            )
            materials.append(material)

    submodels = []

    if isinstance(model, trimesh.base.Trimesh):
        submodels = [(materials[-1]["name"], model)]
    else:
        submodels = list(model.geometry.items())

    for sm_i, (sm_name, sm) in enumerate(submodels):
        # ch_mesh = sm.convex_hull
        # bb_mesh = sm.bounding_box_oriented

        # set visual geom path, copy to visual folder
        # model_file = os.path.basename(model_path)
        obj_name = os.path.basename(obj_file)[:-4]
        vis_geom_path = os.path.join(vis_path, "{}_{}.obj".format(obj_name, sm_i))
        # shutil.copyfile(
        #     obj_file,
        #     vis_geom_path,
        # )
        sm.export(vis_geom_path)

        vis_geom = dict(
            obj_type="visual",
            geom_type="mesh",
            path=vis_geom_path,
            name="{}_{}_vis".format(obj_name, sm_i),
            material=sm_name,
            # **get_bb_info(sm.bounding_box_oriented)
        )
        geoms.append(vis_geom)

        if visual_only:
            continue

        ch_mesh = sm.convex_hull
        # bb_mesh = ch_mesh.bounding_box_oriented

        # obj_name = os.path.basename(obj_file)[:-3]
        coll_geom_path = os.path.join(coll_path, "{}_{}.obj".format(obj_name, sm_i))
        coll_geom = dict(
            obj_type="collision",
            geom_type="mesh",
            # path=os.path.join(coll_path, "{}_ch.stl".format(model_name)),
            path=coll_geom_path,
            name="{}_{}_coll".format(obj_name, sm_i),
            **get_bb_info(ch_mesh.bounding_box_oriented),
        )
        geoms.append(coll_geom)
        ch_mesh.export(coll_geom["path"])

    return geoms, textures, materials, transform


def parse_coll_model(
    coll_model_path,
    asset_path,
    verbose=False,
    transform=None,
):
    """
    Extract list of primitive geoms from collision model file
    """

    if coll_model_path is None:
        return None, None

    LogUtils.maybe_log_info(
        "parsing collision model: {}".format(coll_model_path),
        log=verbose,
        indent=True,
    )

    # load the model
    obj_files = [
        os.path.join(coll_model_path, name)
        for name in os.listdir(coll_model_path)
        if name.endswith(".obj")
    ]

    geoms = []
    coll_path = os.path.join(asset_path, "collision")

    scene = trimesh.scene.scene.Scene()

    for obj_file in obj_files:
        resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(obj_file))
        model = trimesh.load(obj_file, resolver=resolver)

        if transform is not None:
            model.apply_transform(transform)
            model.export(obj_file)

        # ch_mesh = model.convex_hull
        # bb_mesh = ch_mesh.bounding_box_oriented

        scene.add_geometry(model)

        if not os.path.isdir(coll_path):
            os.makedirs(coll_path)

        # set visual geom path, copy to visual folder
        # model_file = os.path.basename(model_path)
        obj_name = os.path.basename(obj_file)[:-3]
        coll_geom_path = os.path.join(coll_path, os.path.basename(obj_file))
        # shutil.copyfile(
        #     obj_file,
        #     coll_geom_path,
        # )

        # obj_name = os.path.basename(obj_file)[:-3]
        coll_geom = dict(
            obj_type="collision",
            geom_type="mesh",
            # path=os.path.join(coll_path, "{}_ch.stl".format(model_name)),
            path=coll_geom_path,
            name="{}_coll".format(obj_name),
            # **get_bb_info(bb_mesh)
        )
        geoms.append(coll_geom)

    bb_info = get_bb_info(scene.bounding_box)

    # f = open(coll_model_path, "r")
    # lines = f.readlines()
    # f.close()

    # geoms = []

    # start_inds = [i for (i, line) in enumerate(lines) if line.startswith("o ")]

    # num_coll_meshes = 0
    # for start, end in zip(start_inds, start_inds[1:] + [len(lines)]):
    #     obj_lines = lines[start:end]

    #     # filter out lines corresponding to faces
    #     obj_lines_filtered = [line for line in obj_lines if not line.startswith("f ")]

    #     geom = dict()

    #     tmp = tempfile.NamedTemporaryFile(suffix=".obj")
    #     with open(tmp.name, "w") as f:
    #         [f.write(line) for line in obj_lines_filtered]

    #     mesh = trimesh.load(tmp.name)

    #     bb_mesh = mesh.bounding_box_oriented

    #     geom.update(
    #         obj_name=obj_lines[0],
    #         obj_type="collision",
    #         **get_bb_info(bb_mesh)
    #     )

    #     obj_type_substr = obj_lines_filtered[0][2:]
    #     if obj_type_substr.startswith("Cube"):
    #         geom_type = "box"
    #     elif obj_type_substr.startswith("Cylinder"):
    #         geom_type = "cylinder"

    #         """
    #         post-process size and orientation of cylinder
    #         """

    #         pos = geom["pos"]
    #         quat = geom["quat"]
    #         size = geom["size"]

    #         T = bb_mesh.primitive.transform

    #         trans_mesh = mesh.copy()
    #         trans_mesh.apply_transform(np.linalg.inv(T))

    #         V = np.array(trans_mesh.vertices)
    #         n = len(V) # number of vertices
    #         th = 1e-5 # threshold for checking vertices

    #         upright_axis = None
    #         for idx in [0, 1, 2]:
    #             V = np.take(V, indices=np.argsort(V, axis=0)[:,idx], axis=0)

    #             if np.std(V[:n//2,idx]) < th and np.std(V[-n//2:,idx]) < th:
    #                 upright_axis = idx

    #         assert upright_axis is not None
    #         circle_axes = np.delete(np.arange(3), upright_axis)
    #         assert np.abs(size[circle_axes][1] - size[circle_axes][0]) < 0.01, (
    #             "Detected elliptic cylinder! "
    #             "Mujoco only accepts circular cylinders."
    #         )

    #         if upright_axis == 0:
    #             geom["quat"] = mat2quat(
    #                 np.linalg.inv(
    #                     np.matmul(
    #                         np.array([
    #                             [0, 0, 1],
    #                             [0, 1, 0],
    #                             [-1, 0, 0],
    #                         ]),
    #                         np.linalg.inv(geom["rot_orig"])
    #                     )
    #                 )
    #             )[[3,0,1,2]]
    #         elif upright_axis == 1:
    #             raise NotImplementedError

    #         geom["size"] = [np.mean(size[circle_axes]), size[upright_axis]]

    #     elif obj_type_substr.startswith("Sphere"):
    #         geom_type = "ellipsoid"
    #     else:
    #         raise ValueError(
    #             "Detected non-primitive object in collision model:",
    #             obj_lines_filtered[0]
    #         )

    #     geom["geom_type"] = geom_type

    #     geoms.append(geom)

    return geoms, bb_info


def get_geom_element(model_name, geom_info, sc=1.0, show_if_coll_geom=False):
    """
    Generate xml element for mjcf geom
    """

    if geom_info.get("hide", False):
        return None

    elem = ET.Element("geom")

    elem.attrib["solimp"] = "0.998 0.998 0.001"
    elem.attrib["solref"] = "0.001 1"
    elem.attrib["density"] = "100"
    elem.attrib["friction"] = "0.95 0.3 0.1"

    if geom_info["geom_type"] == "mesh":
        elem.attrib["type"] = "mesh"
        elem.attrib["mesh"] = geom_info["name"]
    else:
        raise NotImplementedError
        elem.attrib["type"] = geom_info["geom_type"]
        elem.attrib["pos"] = " ".join(["{:.5f}".format(v * sc) for v in geom_info["pos"]])
        elem.attrib["quat"] = " ".join(["{:.5f}".format(v) for v in geom_info["quat"]])
        size = geom_info["size"]
        elem.attrib["size"] = " ".join(["{:.5f}".format(v * sc / 2) for v in size])

    if geom_info["obj_type"] == "visual":
        elem.attrib["conaffinity"] = "0"
        elem.attrib["contype"] = "0"
        elem.attrib["group"] = "1"
        elem.attrib["material"] = geom_info.get("material", model_name)
    elif geom_info["obj_type"] == "collision":
        elem.attrib["group"] = "0"
        # elem.attrib["condim"] = "4"

        if show_if_coll_geom:
            elem.attrib["rgba"] = "0.8 0.8 0.8 0.3"
        else:
            elem.attrib["rgba"] = "0.8 0.8 0.8 0.0"

    return elem


def get_texture_element(texture_info):
    elem = ET.Element("texture")

    for k, v in texture_info.items():
        elem.attrib[k] = str(v)

    return elem


def get_material_element(material_info):
    elem = ET.Element("material")

    for k, v in material_info.items():
        elem.attrib[k] = str(v)

    return elem


def get_mesh_element(geom_info, sc=1.0):
    """
    Generate xml element for mjcf mesh asset
    """
    assert geom_info["geom_type"] == "mesh"

    if geom_info.get("hide", False):
        return None

    elem = ET.Element("mesh")

    elem.attrib["file"] = "/".join(geom_info["path"].split("/")[-2:])
    elem.attrib["name"] = geom_info["name"]
    elem.attrib["scale"] = "{sc} {sc} {sc}".format(sc=sc)

    return elem


def generate_mjcf(
    asset_path,
    model_name,
    model_info,
    sc=1.0,
    texture_path=None,
    hide_vis_geoms=False,
    show_coll_geoms=False,
    show_sites=False,
    verbose=False,
):
    """
    Generate mjcf model for object
    """

    LogUtils.maybe_log_info("Generating MJCF...", log=verbose)

    geoms = model_info["geoms"]
    textures = model_info["textures"]
    materials = model_info["materials"]
    bb_info = model_info["bb"]

    ### for now, assume mtl doesn't contain texture ####
    # texture_from_mtl = False
    # if texture_path is None:
    #     # get the texture used in the mtl file
    #     mtl_path = model_path[:-3] + 'mtl'
    #     texture_dict = {}
    #     with open(mtl_path, "r") as f:
    #         for i, line in enumerate(f.readlines()):
    #             if len(line) > 1:
    #                 key, value = line.split(" ", 1)
    #                 texture_dict[key] = value.strip()
    #
    #     if "map_Kd" in texture_dict:
    #         texture_path = texture_dict["map_Kd"]
    #         texture_from_mtl = True

    # load template xml
    tree = ET.parse(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "object_template.xml")
    )
    root = tree.getroot()

    # set model name
    root.attrib["model"] = model_name

    # add mesh assets
    asset = root.find("asset")
    for geom in geoms:
        if geom["geom_type"] == "mesh":
            mesh_elem = get_mesh_element(geom, sc=sc)
            if mesh_elem is not None:
                asset.append(mesh_elem)

    # update default texture and material assets
    texture = asset.find("texture")
    texture.attrib["name"] = texture.attrib["name"].replace("template", model_name)
    if texture_path is not None:
        texture.attrib["file"] = "/".join(texture_path.split("/")[-1:])
    else:
        texture.attrib["file"] = os.path.join(
            os.path.dirname(robosuite.__file__), "models/assets/textures/ceramic.png"
        )
    material = asset.find("material")
    for k in ["name", "texture"]:
        material.attrib[k] = material.attrib[k].replace("template", model_name)

    # add textures
    for texture in textures:
        texture_elem = get_texture_element(texture)
        if texture_elem is not None:
            asset.append(texture_elem)

    # add materials
    for material in materials:
        material_elem = get_material_element(material)
        if material_elem is not None:
            asset.append(material_elem)

    # add geoms
    worldbody = root.find("worldbody")
    body = worldbody.find("body").find("body")
    for geom in geoms:
        if geom["obj_type"] == "visual" and hide_vis_geoms:
            continue

        geom_elem = get_geom_element(
            model_name,
            geom,
            sc=sc,
            show_if_coll_geom=show_coll_geoms,
        )
        if geom_elem is not None:
            body.append(geom_elem)

    sites = worldbody.find("body").findall("site")

    if show_sites:
        object_body = worldbody.find("body").find("body")
        vis_site_strings = [
            """<site rgba="1 0 0 1" size="0.005" pos="0 0 0" name="bottom_site"/>""",
            """<site rgba="0 1 0 1" size="0.005" pos="0 0 0" name="top_site"/>""",
            """<site rgba="0 0 1 1" size="0.005" pos="0 0 0" name="horizontal_radius_site"/>""",
        ]
        for s in vis_site_strings:
            object_body.append(ET.fromstring(s))

        # for debugging
        vis_sites = object_body.findall("site")
        sites = sites + vis_sites

    if bb_info is not None:
        for site in sites:
            if site.get("name") == "horizontal_radius_site":
                pos = bb_info["pos"].copy()
                pos[0] += bb_info["size"][0] / 2
                pos[1] += bb_info["size"][1] / 2
                pos = np.array([pos[0], pos[1], pos[2]])
                site.set("pos", array_to_string(pos * sc))
            elif site.get("name") == "bottom_site":
                pos = bb_info["pos"].copy()
                pos[2] -= bb_info["size"][2] / 2
                pos = np.array([pos[0], pos[1], pos[2]])
                site.set("pos", array_to_string(pos * sc))
            elif site.get("name") == "top_site":
                pos = bb_info["pos"].copy()
                pos[2] += bb_info["size"][2] / 2
                pos = np.array([pos[0], pos[1], pos[2]])
                site.set("pos", array_to_string(pos * sc))

    # save xml for new model
    mjcf_path = os.path.join(asset_path, "model.xml")
    tree.write(mjcf_path, encoding="utf-8")
    LogUtils.maybe_log_info("Wrote MJCF to: {}".format(mjcf_path), log=verbose, indent=True)

    # LogUtils.maybe_log_info(
    #     "\nTest generated MJCF with robosuite:\npython {base_path}/scripts/object_play.py --device spacemouse --mjcf_path {mjcf_path}".format(
    #         base_path=os.path.dirname(robosuite_model_zoo.__file__),
    #         mjcf_path=mjcf_path,
    #     ),
    #     log=True,
    #     color="green"
    # )

    return mjcf_path


def get_file_name_and_type(path):
    """
    Extract name of object model file and file format
    """
    split_name = os.path.basename(path).split(".")
    assert len(split_name) == 2
    return split_name[0], split_name[1]


def get_bb_info(bb_mesh):
    """
    Extract size, position, and orientation (quat) from bounding box mesh
    """
    ext = np.array(bb_mesh.primitive.extents)
    transform = np.array(bb_mesh.primitive.transform)

    center = transform[:-1, 3]
    rot = transform[:3, :3]
    quat = mat2quat(rot)[[3, 0, 1, 2]]

    return dict(
        pos=center,
        quat=quat,
        size=ext,
        rot_orig=rot,
    )


def decompose_convex(
    filename,
    work_dir,
    # V-HACD params
    max_output_convex_hulls=32,
    voxel_resolution=100000,
    volume_error_percent=1.0,
    max_hull_vert_count=64,
):
    """
    Adapted from https://github.com/kevinzakka/obj2mjcf/blob/main/obj2mjcf/_cli.py
    """

    _VHACD_EXECUTABLE = shutil.which("TestVHACD")
    _VHACD_OUTPUTS = ["decomp.obj", "decomp.stl"]

    obj_file = filename.resolve()

    import tempfile
    import subprocess
    from pathlib import Path
    import enum

    with tempfile.TemporaryDirectory() as tmpdirname:
        prev_dir = os.getcwd()
        os.chdir(tmpdirname)

        # Copy the obj file to the temporary directory.
        shutil.copy(obj_file, tmpdirname)

        # Call V-HACD, suppressing output.
        ret = subprocess.run(
            [
                f"{_VHACD_EXECUTABLE}",
                obj_file.name,
                "-o",
                "obj",
                "-h",
                f"{max_output_convex_hulls}",  # max_output_convex_hulls
                "-r",
                f"{voxel_resolution}",  # voxel_resolution
                "-e",
                f"{volume_error_percent}",  # volume_error_percent
                "-d",
                f"{14}",  # max_recursion_depth
                "-s",
                f"{int(not False)}",  # disable_shrink_wrap
                # "-f",
                # f"{enum.auto().name.lower()}", # fill_mode.name.lower()
                "-v",
                f"{max_hull_vert_count}",  # max_hull_vert_count
                "-a",
                f"{int(not False)}",  # disable_async
                "-l",
                f"{2}",  # min_edge_length
                "-p",
                f"{int(False)}",  # split_hull
            ],
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT,
            check=True,
        )
        # if ret.returncode != 0:
        #     logging.error(f"V-HACD failed on {filename}")
        #     return False

        # Remove the original obj file and the V-HACD output files.
        for name in _VHACD_OUTPUTS + [obj_file.name]:
            file_to_delete = Path(tmpdirname) / name
            if file_to_delete.exists():
                file_to_delete.unlink()

        os.chdir(prev_dir)

        # Get list of sorted collisions.
        collisions = list(Path(tmpdirname).glob("*.obj"))
        collisions.sort(key=lambda x: x.stem)

        os.makedirs(work_dir.resolve(), exist_ok=True)

        for i, filename in enumerate(collisions):
            savename = str(work_dir / f"{obj_file.stem}_collision_{i}.obj")
            shutil.move(str(filename), savename)

    return True
