from __future__ import annotations

import dataclasses
import os
import re

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import joblib
import pxr

from gear_sonic.envs.manager_env.mdp import terrain
from gear_sonic.envs.manager_env.robots import g1, h2
from gear_sonic.trl.utils import common


def _load_opencv_params_from_usd(usd_path: str) -> dict:
    """Parse a USD file to extract OpenCV lens distortion parameters.

    Args:
        usd_path: Path to the USD file containing camera with OpenCV distortion.

    Returns:
        Dictionary with keys: fx, fy, cx, cy, k1-k6, p1, p2, s1-s4

    Raises:
        FileNotFoundError: If USD file doesn't exist
        ValueError: If required intrinsics (fx, fy, cx, cy) are missing
    """
    with open(usd_path) as f:
        content = f.read()

    params = {}

    # Parse the USD text format for omni:lensdistortion:opencvPinhole attributes
    patterns = {
        "fx": r"omni:lensdistortion:opencvPinhole:fx\s*=\s*([-\d.]+)",
        "fy": r"omni:lensdistortion:opencvPinhole:fy\s*=\s*([-\d.]+)",
        "cx": r"omni:lensdistortion:opencvPinhole:cx\s*=\s*([-\d.]+)",
        "cy": r"omni:lensdistortion:opencvPinhole:cy\s*=\s*([-\d.]+)",
        "k1": r"omni:lensdistortion:opencvPinhole:k1\s*=\s*([-\d.]+)",
        "k2": r"omni:lensdistortion:opencvPinhole:k2\s*=\s*([-\d.]+)",
        "k3": r"omni:lensdistortion:opencvPinhole:k3\s*=\s*([-\d.]+)",
        "k4": r"omni:lensdistortion:opencvPinhole:k4\s*=\s*([-\d.]+)",
        "k5": r"omni:lensdistortion:opencvPinhole:k5\s*=\s*([-\d.]+)",
        "k6": r"omni:lensdistortion:opencvPinhole:k6\s*=\s*([-\d.]+)",
        "p1": r"omni:lensdistortion:opencvPinhole:p1\s*=\s*([-\d.]+)",
        "p2": r"omni:lensdistortion:opencvPinhole:p2\s*=\s*([-\d.]+)",
        "s1": r"omni:lensdistortion:opencvPinhole:s1\s*=\s*([-\d.]+)",
        "s2": r"omni:lensdistortion:opencvPinhole:s2\s*=\s*([-\d.]+)",
        "s3": r"omni:lensdistortion:opencvPinhole:s3\s*=\s*([-\d.]+)",
        "s4": r"omni:lensdistortion:opencvPinhole:s4\s*=\s*([-\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            params[key] = float(match.group(1))

    # Assert required intrinsics exist
    required_intrinsics = ["fx", "fy", "cx", "cy"]
    missing = [k for k in required_intrinsics if k not in params]
    if missing:
        raise ValueError(f"USD file {usd_path} missing required OpenCV intrinsics: {missing}")

    # Assert at least k1 distortion exists (otherwise why use OpenCV mode?)
    if "k1" not in params:
        raise ValueError(
            f"USD file {usd_path} missing distortion coefficient k1 - use pinhole camera instead"
        )

    print(f"Loaded OpenCV params from {usd_path}: {params}")  # noqa: T201
    return params


@clone
def spawn_opencv_camera(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> pxr.Usd.Prim:
    """Create a camera with OpenCV lens distortion applied at spawn time.

    This spawner creates a standard camera and then applies the OmniLensDistortionOpenCvPinholeAPI
    schema with the distortion parameters from the config. This happens BEFORE environment cloning,
    so all cloned environments will have the correct distortion properties.
    """
    import isaacsim.core.utils.prims as prim_utils

    # Create the camera prim
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(
            prim_path, "Camera", translation=translation, orientation=orientation
        )
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.get_prim_at_path(prim_path)

    # Set clipping range (the only standard camera param that matters for OpenCV mode)
    if hasattr(cfg, "clipping_range") and cfg.clipping_range is not None:
        prim.GetAttribute("clippingRange").Set(cfg.clipping_range)

    # Apply OmniLensDistortionOpenCvPinholeAPI schema via Sdf layer
    # Note: This only works with headless=false (use xvfb-run for headless rendering)
    stage = prim.GetStage()
    layer = stage.GetRootLayer()
    prim_spec = layer.GetPrimAtPath(prim_path)
    if prim_spec is None:
        prim_spec = pxr.Sdf.CreatePrimInLayer(layer, prim_path)  # noqa: F823

    api_schemas = prim_spec.GetInfo("apiSchemas")
    if api_schemas is None:
        api_schemas = pxr.Sdf.TokenListOp()
    prepend_items = list(api_schemas.prependedItems) if api_schemas.prependedItems else []
    if "OmniLensDistortionOpenCvPinholeAPI" not in prepend_items:
        prepend_items.append("OmniLensDistortionOpenCvPinholeAPI")
        api_schemas.prependedItems = prepend_items
        prim_spec.SetInfo("apiSchemas", api_schemas)

    # Set the distortion model
    prim.CreateAttribute("omni:lensdistortion:model", pxr.Sdf.ValueTypeNames.Token).Set(
        "opencvPinhole"
    )

    # Set OpenCV distortion parameters from config
    opencv_attrs = {
        "omni:lensdistortion:opencvPinhole:fx": ("opencv_fx", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:fy": ("opencv_fy", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:cx": ("opencv_cx", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:cy": ("opencv_cy", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k1": ("opencv_k1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k2": ("opencv_k2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k3": ("opencv_k3", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k4": ("opencv_k4", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k5": ("opencv_k5", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k6": ("opencv_k6", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:p1": ("opencv_p1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:p2": ("opencv_p2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s1": ("opencv_s1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s2": ("opencv_s2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s3": ("opencv_s3", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s4": ("opencv_s4", pxr.Sdf.ValueTypeNames.Float),
    }

    for attr_name, (cfg_name, attr_type) in opencv_attrs.items():
        if hasattr(cfg, cfg_name):
            value = getattr(cfg, cfg_name)
            if value is not None:
                prim.CreateAttribute(attr_name, attr_type).Set(value)

    # Set image size
    if hasattr(cfg, "opencv_image_size") and cfg.opencv_image_size is not None:
        import pxr

        prim.CreateAttribute(
            "omni:lensdistortion:opencvPinhole:imageSize", pxr.Sdf.ValueTypeNames.Int2
        ).Set(pxr.Gf.Vec2i(cfg.opencv_image_size[0], cfg.opencv_image_size[1]))
        print(  # noqa: T201
            f"[DEBUG] spawn_opencv_camera: Set imageSize to {cfg.opencv_image_size}"
        )  # noqa: T201

    # Debug: print all the values that were set
    print(f"[DEBUG] spawn_opencv_camera at {prim_path}:")  # noqa: T201
    print(  # noqa: T201
        f"[DEBUG]   fx={getattr(cfg, 'opencv_fx', None)}, fy={getattr(cfg, 'opencv_fy', None)}"
    )  # noqa: T201
    print(  # noqa: T201
        f"[DEBUG]   cx={getattr(cfg, 'opencv_cx', None)}, cy={getattr(cfg, 'opencv_cy', None)}"
    )  # noqa: T201
    print(f"[DEBUG]   image_size={getattr(cfg, 'opencv_image_size', None)}")  # noqa: T201

    return prim


@configclass
class OpenCVCameraCfg:
    """Camera config with OpenCV lens distortion parameters."""

    func = spawn_opencv_camera
    copy_from_source: bool = True  # Required by IsaacLab spawner

    clipping_range: tuple[float, float] = (0.01, 20.0)

    # These must exist or Camera.__init__ crashes, but values don't matter -
    # OpenCV distortion API (fx/fy/cx/cy) overrides the projection at render time
    horizontal_aperture: float = 1.0
    vertical_aperture: float | None = None

    # OpenCV intrinsics (required)
    opencv_fx: float = None
    opencv_fy: float = None
    opencv_cx: float = None
    opencv_cy: float = None
    opencv_image_size: tuple[int, int] = None

    # OpenCV distortion coefficients (required - at least k1 should be non-zero)
    opencv_k1: float = None
    opencv_k2: float = 0.0
    opencv_k3: float = 0.0
    opencv_k4: float = 0.0
    opencv_k5: float = 0.0
    opencv_k6: float = 0.0
    opencv_p1: float = 0.0
    opencv_p2: float = 0.0
    opencv_s1: float = 0.0
    opencv_s2: float = 0.0
    opencv_s3: float = 0.0
    opencv_s4: float = 0.0


def _resolve_object_usd_paths(usd_path):
    def expand_regex(path_pattern):
        if os.path.exists(path_pattern):
            return [os.path.abspath(path_pattern)]
        directory = os.path.dirname(path_pattern) or "."
        if not os.path.isdir(directory):
            return []
        base_pattern = os.path.basename(path_pattern)
        try:
            regex = re.compile(base_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid object_usd_path regex: {base_pattern}") from exc
        matched = [
            os.path.abspath(os.path.join(directory, name))
            for name in os.listdir(directory)
            if regex.fullmatch(name)
        ]
        matched.sort(key=lambda p: os.path.basename(p))
        return matched

    if isinstance(usd_path, list | tuple):
        raw_paths = list(usd_path)
    else:
        raw_paths = [usd_path]

    expanded_paths = []
    used_pattern = False
    for path in raw_paths:
        if isinstance(path, str):
            pattern_matches = expand_regex(path)
            if len(pattern_matches) == 1:
                expanded_paths.append(os.path.abspath(pattern_matches[0]))
            elif len(pattern_matches) > 1:
                used_pattern = True
                expanded_paths.extend(pattern_matches)
            else:
                raise ValueError(f"object_usd_path regex did not match any files: {path}")
        else:
            expanded_paths.append(path)

    if not expanded_paths:
        expanded_paths.append(os.path.abspath(raw_paths[0]))
    if used_pattern:
        expanded_paths.sort(key=lambda p: os.path.basename(p))
    return expanded_paths


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __init__(self, config, **kwargs):  # noqa: ARG002
        super().__init__()

        self.num_envs = config.get("num_envs", 4096)

        self.env_spacing = config.get("env_spacing", 2.5)

        # Allow config to override replicate_physics (default True)
        # Set to False for per-environment randomization (e.g., table size)
        self.replicate_physics = config.get("replicate_physics", True)

        self.eval_camera = None
        if config.get("render_results", False):
            self.eval_camera = TiledCameraCfg(
                prim_path="/World/envs/env_.*/eval_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0,
                    focus_distance=50.0,
                    horizontal_aperture=5,
                    clipping_range=(0.1, 20.0),
                ),
                width=config.get("render_width", 1920),
                height=config.get("render_height", 1080),
            )

        # Single global overview camera (not per-env) for replay videos
        self.overview_camera = None
        if config.get("overview_camera", False):
            self.overview_camera = CameraCfg(
                prim_path="/World/OverviewCamera",
                offset=CameraCfg.OffsetCfg(pos=(0, 0, 50), rot=(1, 0, 0, 0), convention="world"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0,
                    focus_distance=100.0,
                    horizontal_aperture=10.0,  # Wider FOV for overview
                    clipping_range=(0.1, 500.0),  # Extended for large grids
                ),
                width=config.get("render_width", 1920),
                height=config.get("render_height", 1080),
            )

        # ground terrain
        terrain_type = config.get("terrain_type", "plane")
        if terrain_type == "plane":
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
            )
        elif terrain_type == "trimesh":
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain.ROUGH_TERRAINS_CFG,
                max_init_terrain_level=10,
                collision_group=1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

        # robots
        self.robot: ArticulationCfg = dataclasses.MISSING

        # lights
        if not config.get("render_ego_random", False):
            self.light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
            )

            self.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
            )
        else:
            self.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
            )
            # from isaac_playground.env_rand.domelight import RandomDomeLightCfg
            # self.sky_light = RandomDomeLightCfg(
            #     prim_path="/World/skyLight",
            #     texture_file_folder="../rl_data/HDRIs",
            #     dynamic_randomize_texture=True,
            #     dynamic_randomize_texture_interval=1.0
            # )

        self.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            force_threshold=10.0,
            debug_vis=False,
        )

        # Check if robot has hands (43 DOF robots) - used for hand-related sensors
        robot_type = config.get("robot", {}).get("type", "g1")
        robot_has_hands = "43dof" in robot_type or "hand" in robot_type

        motion_meta_info_path = config.get("motion_meta_info_path", None)
        motion_meta_info = None

        usd_path = config.get("object_usd_path", "")
        if motion_meta_info_path is None and isinstance(usd_path, str) and os.path.isfile(usd_path):
            motion_meta_info_path = usd_path.replace(".usd", ".pkl").replace("object_usd", "meta")
        if motion_meta_info_path is not None and os.path.isfile(motion_meta_info_path):
            motion_meta_info = joblib.load(motion_meta_info_path)

        # Scene scale: config override > meta file > default 1.0
        scene_scale = config.get("scene_scale", 1.0)
        if motion_meta_info is not None:
            scene_scale = motion_meta_info.get("scene_scale", 1.0)

        # Ensure scene_scale is never None
        if scene_scale is None:
            scene_scale = 1.0

        # Object with rigid body (optional, can be configured via config)
        # Supports: single file, directory, list, or regex patterns for object_usd_path
        #
        # multi_object_per_env=True:  All objects spawned in every env (one active at a time)
        # multi_object_per_env=False: One object per env (MultiUsdFileCfg cycles through)
        if config.get("add_object", False):
            usd_path = config.get("object_usd_path", f"{os.getcwd()}/data/wheelchair.usd")
            object_is_dynamic = config.get("object_is_dynamic", False)
            object_collision_enabled = config.get("object_collision_enabled", True)
            multi_object_per_env = config.get("multi_object_per_env", False)
            object_color = config.get("object_color", None)  # e.g. [0.6, 0.4, 0.2] for wood-brown

            # --- Step 1: Resolve usd_path to a list of absolute paths ---
            if isinstance(usd_path, list):
                resolved_paths = _resolve_object_usd_paths(usd_path)
            elif os.path.isdir(usd_path):
                resolved_paths = sorted(
                    os.path.abspath(os.path.join(usd_path, f))
                    for f in os.listdir(usd_path)
                    if f.endswith(".usd")
                )
            else:
                resolved_paths = [os.path.abspath(usd_path)]

            # --- Step 2: Spawn based on multi_object_per_env flag ---
            if multi_object_per_env:
                # MULTI-OBJECT MODE: One RigidObjectCfg per USD, all present in every env.
                # commands.py detects this by checking for object_* entries in scene.rigid_objects.
                #
                # CRITICAL: Initial positions must be spread apart to avoid collision pairs!
                # If all objects spawn at (0,0,0), PhysX creates O(N²) pairs at scene creation.
                # Spread in Z (vertical) since envs only vary in X,Y.
                z_spacing = 10.0
                for idx, path in enumerate(resolved_paths):
                    obj_name = os.path.splitext(os.path.basename(path))[0]
                    obj_name_safe = obj_name.replace("-", "_")
                    init_z = -100.0 - idx * z_spacing
                    setattr(
                        self,
                        f"object_{obj_name_safe}",
                        RigidObjectCfg(
                            prim_path=f"{{ENV_REGEX_NS}}/Object_{obj_name_safe}",
                            spawn=sim_utils.UsdFileCfg(
                                usd_path=path,
                                activate_contact_sensors=True,
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    kinematic_enabled=not object_is_dynamic,
                                    max_depenetration_velocity=1.0,
                                ),
                                collision_props=sim_utils.CollisionPropertiesCfg(
                                    collision_enabled=object_collision_enabled,
                                ),
                            ),
                            init_state=RigidObjectCfg.InitialStateCfg(pos=(1000.0, 0.0, init_z)),
                        ),
                    )
                print(  # noqa: T201
                    f"[Multi-Object Mode] Spawned {len(resolved_paths)} objects at spread Z positions"
                )
            elif len(resolved_paths) == 1:
                # SINGLE OBJECT MODE
                object_mass = config.get("object_mass", None)
                mass_props = (
                    sim_utils.MassPropertiesCfg(mass=object_mass)
                    if object_mass is not None
                    else None
                )
                object_opacity = config.get("object_opacity", 1.0)
                if object_color is not None:
                    visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(object_color[:3]),
                        opacity=object_opacity,
                        metallic=object_color[3] if len(object_color) > 3 else 0.0,
                        roughness=object_color[4] if len(object_color) > 4 else 0.5,
                    )
                elif object_opacity < 1.0:
                    visual_material = sim_utils.PreviewSurfaceCfg(opacity=object_opacity)
                else:
                    visual_material = None
                self.object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=resolved_paths[0],
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=not object_is_dynamic,
                            max_depenetration_velocity=1.0,
                        ),
                        mass_props=mass_props,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=object_collision_enabled,
                        ),
                        visual_material=visual_material,
                        scale=(scene_scale, scene_scale, scene_scale),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(config.get("object_position", [2.0, 0.0, 0.0]))
                    ),
                )
            else:
                # ONE-PER-ENV MODE: Different object per env via MultiUsdFileCfg
                self.replicate_physics = False
                self.object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=resolved_paths,
                        random_choice=False,
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=not object_is_dynamic,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=object_collision_enabled,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(config.get("object_position", [2.0, 0.0, 0.0]))
                    ),
                )

            if robot_has_hands:
                # Frame transformer for hand-object tracking
                self.object_to_hand_frame_transformer = FrameTransformerCfg(
                    prim_path="{ENV_REGEX_NS}/Object",  # Source: Object
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            name="right_hand_thumb",
                            prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
                        ),
                        FrameTransformerCfg.FrameCfg(
                            name="right_hand_index",
                            prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                        ),
                        FrameTransformerCfg.FrameCfg(
                            name="right_hand_middle",
                            prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                        ),
                    ],
                )

                # Object-to-hand contact sensor for grasp detection (only for robots with hands)
                # force_matrix_w gives shape [num_envs, 1, N, 3] - force on Object from each finger link
                # Configurable via contact_sensor_finger_links (list of link names without Robot/ prefix)
                # Default: tips + palm (4 links). For fuller grasp detection, use all finger segments:
                #   [right_hand_palm_link, right_hand_thumb_0_link, right_hand_thumb_1_link,
                #    right_hand_thumb_2_link, right_hand_index_0_link, right_hand_index_1_link,
                #    right_hand_middle_0_link, right_hand_middle_1_link]
                custom_finger_links = config.get("contact_sensor_finger_links", None)
                if custom_finger_links is not None:
                    right_finger_tip_bodies = [
                        f"{{ENV_REGEX_NS}}/Robot/{link}" for link in custom_finger_links
                    ]
                else:
                    # Default: fingertips + palm (4 links)
                    right_finger_tip_bodies = [
                        "{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
                        "{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                        "{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                        "{ENV_REGEX_NS}/Robot/right_hand_palm_link",
                    ]
                self.object_to_hand_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    filter_prim_paths_expr=right_finger_tip_bodies,
                    history_length=2,
                    track_air_time=False,
                )

                left_finger_tip_bodies = [
                    "{ENV_REGEX_NS}/Robot/left_hand_thumb_2_link",
                    "{ENV_REGEX_NS}/Robot/left_hand_index_1_link",
                    "{ENV_REGEX_NS}/Robot/left_hand_middle_1_link",
                    "{ENV_REGEX_NS}/Robot/left_hand_palm_link",
                ]
                self.object_to_left_hand_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    filter_prim_paths_expr=left_finger_tip_bodies,
                    history_length=2,
                    track_air_time=False,
                )

        if config.get("add_table", False):
            # Table initial position and orientation from config or meta
            # (per-env update in commands.py for multi-motion)
            if "table_position" in config:
                table_pos = config["table_position"]
            elif motion_meta_info is not None and "table_pos" in motion_meta_info:
                table_pos = motion_meta_info["table_pos"]
            else:
                table_pos = [0.0, 0.0, 0.8]

            # Apply table_offset if configured (additive offset to table position)
            table_offset = config.get("table_offset", None)
            if table_offset is not None:
                table_pos[0] += table_offset[0]
                table_pos[1] += table_offset[1]
                table_pos[2] += table_offset[2]

            # Table quaternion (w, x, y, z) - GRAB data uses 90° X-rotation
            if "table_quat" in config:
                table_quat = tuple(config["table_quat"])
            elif motion_meta_info is not None and "table_quat" in motion_meta_info:
                table_quat = tuple(motion_meta_info["table_quat"])
            else:
                # Default for GRAB table USD: 90° rotation around X-axis
                table_quat = (0.7071068, -0.7071068, 0.0, 0.0)

            # Use table USD when: scene_scale is explicitly set OR table_usd_path is provided OR meta info exists
            use_table_usd = config.get("table_usd_path") is not None

            if not use_table_usd:
                # Use simple cuboid table for GeniHOI or default scenarios
                # Table size can be configured via config or meta file
                # For GeniHOI: table_size = [width, depth, thickness] from meta
                # Default: 1.0 x 0.6 x 0.04 (reasonable for kitchen/desk scenarios)

                if "table_size" in config:
                    table_size = config["table_size"]
                    table_width = table_size[0]
                    table_depth = table_size[1]
                    table_thickness = table_size[2]
                elif motion_meta_info is not None and "table_size" in motion_meta_info:
                    table_size = motion_meta_info["table_size"]
                    table_width = table_size[0]
                    table_depth = table_size[1]
                    table_thickness = table_size[2]
                else:
                    # Default cuboid size - large enough for most GeniHOI scenarios
                    table_width = 1.0
                    table_depth = 0.6
                    table_thickness = 0.04

                # For cuboid tables, use identity quaternion (no rotation needed)
                table_quat_cuboid = (1.0, 0.0, 0.0, 0.0)

                self.table = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.CuboidCfg(
                        activate_contact_sensors=True,
                        size=(table_width, table_depth, table_thickness),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            max_depenetration_velocity=1.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(table_pos), rot=table_quat_cuboid
                    ),
                )
            else:
                # Use table USD with scene_scale and proper rotation
                self.table = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UsdFileCfg(
                        activate_contact_sensors=True,
                        usd_path=config.get(
                            "table_usd_path", "data/motion_lib_grab/common/table.usda"
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        scale=(scene_scale, scene_scale, scene_scale),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(table_pos), rot=table_quat),
                )

            # Table-to-hand contact sensor for walk_stand_grasp rewards (only for robots with hands)
            if robot_has_hands:
                # Use Table as prim_path and explicitly list hand bodies in filter_prim_paths_expr
                table_contact_bodies = [
                    # Right wrist
                    "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
                    "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                    # Right hand
                    "{ENV_REGEX_NS}/Robot/right_hand_palm_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_index_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_middle_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
                ]
                self.table_to_hand_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    filter_prim_paths_expr=table_contact_bodies,
                    history_length=2,
                    track_air_time=False,
                )

            # Object-to-table contact sensor for detecting object-table contact forces
            # Sensor is attached to Object, filters for contact with Table
            # force_matrix_w gives shape [num_envs, 1, 1, 3] - force on Object from Table
            if config.get("add_object", False):
                self.object_to_table_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
                    history_length=2,
                    track_air_time=False,
                )

                # Object-to-robot contact sensor: detect when right hand/wrist touches object
                # Used for termination logic (robot must touch object before table)
                # NOTE: IsaacLab ContactSensor only supports one-to-many filtering,
                # so prim_path must be single body (Object), filter can be multiple (robot links)
                right_hand_wrist_links = [
                    # Right wrist
                    "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
                    "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                    # Right hand
                    "{ENV_REGEX_NS}/Robot/right_hand_palm_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_index_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_middle_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_0_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_1_link",
                    "{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
                ]
                self.object_to_robot_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    filter_prim_paths_expr=right_hand_wrist_links,
                    history_length=2,
                    track_air_time=False,
                )

            # Table-to-robot contact sensor: detect when right hand/wrist touches table
            # Used for termination logic (terminate if table contact before object contact)
            self.table_to_robot_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                filter_prim_paths_expr=right_hand_wrist_links,
                history_length=2,
                track_air_time=False,
            )

        # # TODO: Do this better.  # noqa: TD002, TD003
        # self.ego_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/head_link/HeadCamera",
        #     # parent=SceneEntityCfg(
        #     #     name="robot",
        #     #     body_names=["head_link"],
        #     # ),
        #     # offset=TiledCameraCfg.OffsetCfg(pos=camera_pos_offset, rot=camera_rot_offset, convention="world"),
        #     data_types=['rgb'],
        #     spawn=sim_utils.PinholeCameraCfg(focal_length=5.0, focus_distance=50.0, horizontal_aperture=5, clipping_range=(0.1, 20.0)),  # noqa: E501
        #     width=384,
        #     height=384,
        #     debug_vis=True,
        # )

        # TODO: Do this better.  # noqa: TD002, TD003
        # Copied from gear_sonic/config/simulator/isaacsim.yaml
        # enable_cameras flag creates the ego camera for vision-based policies
        if config.get("enable_cameras", False) or config.get("render_ego", False):
            # Get camera config from nested cameras dict
            cameras_cfg = config.get("cameras", {})

            # Choose camera type: fisheye, pinhole, or opencv (with lens distortion)
            if config.get("render_ego_opencv_usd", False):
                # Load OpenCV distortion parameters from USD file
                opencv_usd_path = cameras_cfg.get(
                    "opencv_usd_path", "runs/oak_camera.usda"  # Default path
                )
                print(  # noqa: T201
                    f"USING OPENCV CAMERA - loading distortion from: {opencv_usd_path}"
                )  # noqa: T201

                # Parse the USD file to extract distortion parameters
                opencv_params = _load_opencv_params_from_usd(opencv_usd_path)

                # Camera resolution [H, W] - required, no fallback
                camera_resolution = cameras_cfg["camera_resolution"]

                # Use custom OpenCV spawner that applies distortion at spawn time (before cloning)
                print(  # noqa: T201
                    f"[DEBUG] Creating OpenCVCameraCfg with resolution={camera_resolution} -> image_size={(camera_resolution[1], camera_resolution[0])}"  # noqa: E501
                )
                print(  # noqa: T201
                    f"[DEBUG] fx={opencv_params['fx']}, fy={opencv_params['fy']}, cx={opencv_params['cx']}, cy={opencv_params['cy']}"  # noqa: E501
                )
                print(f"[DEBUG] k1={opencv_params['k1']}")  # noqa: T201
                camera_spawn_cfg = OpenCVCameraCfg(
                    clipping_range=(0.01, 20.0),
                    opencv_fx=opencv_params["fx"],
                    opencv_fy=opencv_params["fy"],
                    opencv_cx=opencv_params["cx"],
                    opencv_cy=opencv_params["cy"],
                    opencv_k1=opencv_params["k1"],  # Required - validated by parser
                    opencv_k2=opencv_params.get("k2", 0.0),
                    opencv_k3=opencv_params.get("k3", 0.0),
                    opencv_k4=opencv_params.get("k4", 0.0),
                    opencv_k5=opencv_params.get("k5", 0.0),
                    opencv_k6=opencv_params.get("k6", 0.0),
                    opencv_p1=opencv_params.get("p1", 0.0),
                    opencv_p2=opencv_params.get("p2", 0.0),
                    opencv_s1=opencv_params.get("s1", 0.0),
                    opencv_s2=opencv_params.get("s2", 0.0),
                    opencv_s3=opencv_params.get("s3", 0.0),
                    opencv_s4=opencv_params.get("s4", 0.0),
                    opencv_image_size=(camera_resolution[1], camera_resolution[0]),  # (W, H)
                )
            elif config.get("render_ego_fisheye", False):
                print("USING FISHEYE CAMERA" * 1000)  # noqa: T201
                camera_spawn_cfg = sim_utils.FisheyeCameraCfg(
                    projection_type=config.get("render_ego_fisheye_projection", "fisheyeSpherical"),
                    fisheye_max_fov=config.get("render_ego_fisheye_fov", 180.0),
                    focus_distance=0.5,
                    clipping_range=(0.1, 20.0),
                )
            elif config.get("render_ego_fisheye_polynomial", False):
                camera_spawn_cfg = sim_utils.FisheyeCameraCfg(
                    projection_type="fisheyePolynomial",
                    fisheye_max_fov=360.0,
                    focus_distance=0.5,
                    clipping_range=(0.1, 20.0),
                )
            else:
                # Convert clipping_range to tuple if it's a list or string (from YAML config)
                clipping_range = cameras_cfg.get("camera_clipping_range", (0.1, 20.0))
                if isinstance(clipping_range, str):
                    # Parse string like "(0.01,20.0)" to tuple
                    clipping_range = tuple(map(float, clipping_range.strip("()").split(",")))
                elif isinstance(clipping_range, list):
                    clipping_range = tuple(clipping_range)
                camera_spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=cameras_cfg.get("camera_focal_length", 1.88),
                    focus_distance=cameras_cfg.get("camera_focus_distance", 0.5),
                    horizontal_aperture=cameras_cfg.get("camera_horizontal_aperture", 2.6035),
                    vertical_aperture=cameras_cfg.get("camera_vertical_aperture", 1.4621),
                    clipping_range=clipping_range,
                )

            # Camera attachment link (e.g., "d435_link" for RealSense D435)
            # If not specified, camera is attached to robot root
            camera_attached_link = cameras_cfg.get("camera_attached_link", "")
            if camera_attached_link:
                camera_prim_path = f"{{ENV_REGEX_NS}}/Robot/{camera_attached_link}/ego_camera"
            else:
                camera_prim_path = "{ENV_REGEX_NS}/Robot/ego_camera"

            # Camera position and rotation offsets (relative to attached link)
            camera_pos_offset = tuple(cameras_cfg.get("camera_pos_offset", [0.0, 0.0, 0.0]))
            camera_rot_offset = tuple(
                cameras_cfg.get("camera_rot_offset", [1.0, 0.0, 0.0, 0.0])
            )  # wxyz quaternion

            # Camera data types (e.g., ["rgb"], ["rgb", "depth"])
            camera_data_types = cameras_cfg.get("camera_data_types", ["rgb"])

            # Camera resolution [H, W]
            camera_resolution = cameras_cfg.get("camera_resolution", [108, 192])

            self.ego_camera = TiledCameraCfg(
                prim_path=camera_prim_path,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=camera_pos_offset, rot=camera_rot_offset, convention="world"
                ),
                data_types=camera_data_types,
                spawn=camera_spawn_cfg,
                height=camera_resolution[0],
                width=camera_resolution[1],
                debug_vis=True,
                update_period=0.0,
                update_latest_camera_pose=True,
            )


@configclass
class ModularTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Modular configuration for the tracking environment that uses Hydra composition."""

    def __init__(  # noqa: D417
        self,
        config,
        actions,
        observations,
        rewards,
        terminations,
        commands,
        events,
        curriculum,
        recorders,
        **kwargs,  # noqa: ARG002
    ):
        """Initialize the configuration with Hydra config.

        Args:
            hydra_cfg: Hydra configuration containing component specifications
        """
        super().__init__()
        self._setup_from_hydra(
            config,
            actions,
            observations,
            rewards,
            terminations,
            commands,
            events,
            curriculum,
            recorders,
        )
        self.override_settings()

    def _setup_from_hydra(
        self,
        config,
        actions,
        observations,
        rewards,
        terminations,
        commands,
        events,
        curriculum,
        recorders,
    ):
        """Setup configuration from Hydra config."""
        self.config = config
        # Scene settings
        self.scene = MySceneCfg(config=config)
        # Instantiate components using Hydra
        self.actions = common.custom_instantiate(actions, _recursive=True)
        self.observations = common.custom_instantiate(observations, _recursive=True)
        self.rewards = common.custom_instantiate(rewards, _recursive=True)
        self.terminations = common.custom_instantiate(terminations, _recursive=True)
        self.commands = common.custom_instantiate(commands, _recursive=True)
        self.events = common.custom_instantiate(events, _recursive=True)
        # Debug: Print instantiated events
        print(f"[DEBUG] Instantiated events: {self.events}")  # noqa: T201
        if hasattr(self.events, "__dict__"):
            for event_name, event_cfg in self.events.__dict__.items():
                if event_cfg is not None and not event_name.startswith("_"):
                    print(  # noqa: T201
                        f"[DEBUG]   Event '{event_name}': {type(event_cfg).__name__}"
                    )  # noqa: T201
        self.curriculum = common.custom_instantiate(curriculum, _recursive=True)
        self.recorders = common.custom_instantiate(recorders, _recursive=True)

    def override_settings(self):
        config = self.config
        # General settings
        self.decimation = config.get("decimation", 4)
        self.episode_length_s = config.get("episode_length_s", 10.0)

        # Simulation settings
        self.sim.dt = config.get("sim_dt", 0.005)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Increase collision stack size for scenes with complex collision meshes (e.g. staircases)
        gpu_collision_stack_size_exp = config.get("gpu_collision_stack_size_exp", 26)
        self.sim.physx.gpu_collision_stack_size = 2**gpu_collision_stack_size_exp

        # Increase PhysX GPU memory only for multi-object scenes
        # These prevent "totalAggregatePairsCapacity" errors when many objects are spawned
        # Check if object_usd_path is a directory (multi-object mode)
        object_usd_path = config.get("object_usd_path", "")
        if config.get("add_object", False) and (
            isinstance(object_usd_path, list) or os.path.isdir(object_usd_path)
        ):
            # With proper Z-spacing of initial positions, collision pairs should be minimal
            # These are moderate values that should work for 1000+ envs
            self.sim.physx.gpu_found_lost_pairs_capacity = 2**24  # ~16M
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**24
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21  # ~2M

        # Viewer settings
        viewer_config = config.get("viewer", {})
        self.viewer = ViewerCfg(
            eye=viewer_config.get("eye", [4.5, 0.0, 4.0]),
            lookat=viewer_config.get("lookat", [0.0, 0.0, 0.0]),
        )

        robot_mapping = {
            "g1_model_12_dex": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
            },
            "h2": {
                "robot_cfg": h2.H2_CFG,
                "action_scale": h2.H2_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": h2.H2_ISAACLAB_TO_MUJOCO_MAPPING,
            },
        }

        robot_type = config["robot"].get("type", "g1")
        self.scene.robot = robot_mapping[robot_type]["robot_cfg"].replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions.joint_pos.scale = robot_mapping[config["robot"].get("type", "g1")][
            "action_scale"
        ]
        self.isaaclab_to_mujoco_mapping = robot_mapping[config["robot"].get("type", "g1")][
            "isaaclab_to_mujoco_mapping"
        ]

        # curriculum? WARNING HARDCODED
        import importlib

        if (
            hasattr(self.curriculum, "force_push_curriculum")
            and self.curriculum.force_push_curriculum
        ):
            module = importlib.import_module("gear_sonic.envs.manager_env.mdp")
            self.curriculum.force_push_curriculum.params["modify_fn"] = getattr(
                module, "step_curriculum"
            )

        if (
            hasattr(self.curriculum, "force_push_linear_curriculum")
            and self.curriculum.force_push_linear_curriculum
        ):
            module = importlib.import_module("gear_sonic.envs.manager_env.mdp")
            self.curriculum.force_push_linear_curriculum.params["modify_fn"] = getattr(
                module, "linear_curriculum"
            )
