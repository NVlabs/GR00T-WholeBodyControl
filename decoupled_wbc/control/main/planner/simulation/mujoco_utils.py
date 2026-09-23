import mujoco
import numpy as np


########## Joint Operations ##########
def sample_qpos(
    model: mujoco.MjModel,
    joint_ids: list[int] | None = None,
    joint_names: list[str] | None = None,
    mean: np.ndarray | None = None,
    sigma: np.ndarray | float = 0.4,
):
    """Sample a qpos within the limits of the given joint ids or names"""
    if joint_ids is None and joint_names is not None:
        joint_ids = joint_names_to_joint_ids(model, joint_names)
    if joint_ids is None and joint_names is None:
        raise ValueError("Either joint_ids or joint_names must be provided")
    lo, hi = joints_to_limits(model, joint_ids)

    # This function only works with regular hinge joints and slide joints
    for j_id in joint_ids:
        if model.jnt_type[j_id] not in [
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ]:
            raise ValueError(
                f"Joint {j_id} is not a regular hinge or slide joint"
                + f"Sample qpos is not supported for {model.jnt_type[j_id]}"
            )

    # Sample completely randomly
    if mean is None:
        return np.random.uniform(low=lo, high=hi)

    # Sample around mean
    if isinstance(sigma, float):
        sigma = np.full(len(joint_ids), sigma)
    q = mean + sigma * np.random.randn(len(joint_ids))
    q = np.clip(q, lo, hi)
    return q


def joint_type_sizes(joint_type: mujoco.mjtJoint) -> tuple[int, int]:
    """Return the size of the given joint type"""
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1, 1
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return 1, 1
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3  # quaternion
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6  # pos(3) + quat(4)
    raise ValueError(f"Unknown joint type {joint_type}")


def joint_names_to_joint_ids(
    model: mujoco.MjModel, joint_names: list[str]
) -> np.ndarray:
    """Return array of joint ids for the given joint names"""
    ids = []
    for name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        ids.append(j_id)
    return np.array(ids, dtype=int)


def joints_to_qpos_dof_ids(
    model: mujoco.MjModel,
    joint_ids: list[int] | None = None,
    joint_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tuple of (qpos_ids, dof_ids) for the given joint ids or names"""
    if joint_ids is None and joint_names is not None:
        joint_ids = joint_names_to_joint_ids(model, joint_names)
    if joint_ids is None and joint_names is None:
        raise ValueError("Either joint_ids or joint_names must be provided")

    qpos_ids = []
    dof_ids = []
    for j_id in joint_ids:
        adr = int(model.jnt_qposadr[j_id])
        qpos_size, dof_size = joint_type_sizes(model.jnt_type[j_id])
        qpos_ids.extend(range(adr, adr + qpos_size))
        dof_ids.extend(range(adr, adr + dof_size))
    return np.array(qpos_ids, dtype=int), np.array(dof_ids, dtype=int)


def joints_to_limits(
    model: mujoco.MjModel,
    joint_ids: list[int] | None = None,
    joint_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the limits of the given joint ids or names"""
    if joint_ids is None and joint_names is not None:
        joint_ids = joint_names_to_joint_ids(model, joint_names)
    if joint_ids is None and joint_names is None:
        raise ValueError("Either joint_ids or joint_names must be provided")

    lo = np.full(len(joint_ids), -np.inf)
    hi = np.full(len(joint_ids), np.inf)
    for i, j_id in enumerate(joint_ids):
        if model.jnt_limited[j_id]:
            lo[i] = model.jnt_range[j_id][0]
            hi[i] = model.jnt_range[j_id][1]
    return lo, hi


########## Collision Checking ##########
def get_geoms_from_group(
    model: mujoco.MjModel, group_id: int, root_link: str | None = None
) -> set[int]:
    """Get the geoms from the given group id"""
    # Mark bodies in subtree to check
    if root_link is None:
        in_subtree = [True] * model.nbody
    else:
        # Get the root body id
        root_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, root_link
        )
        if root_body_id < 0:
            raise ValueError(f"Root body '{root_link}' not found")

        # Mark bodies in root subtree
        in_subtree = [False] * model.nbody
        in_subtree[root_body_id] = True
        for b in range(model.nbody):
            p = model.body_parentid[b]
            if p >= 0 and in_subtree[p]:
                in_subtree[b] = True

    # Get geoms that are in the subtree and are in given group
    geoms = set()
    for g in range(model.ngeom):
        if in_subtree[model.geom_bodyid[g]]:
            if model.geom_group[g] == group_id:
                geoms.add(g)
    return geoms


def geoms_in_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geoms_ids: set[int],
    tolerance: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """Check if the given geoms is in contact with others"""
    for i in range(data.ncon):
        c = data.contact[i]

        if c.dist < -tolerance and (
            c.geom1 in geoms_ids or c.geom2 in geoms_ids
        ):
            if verbose:
                g1 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1
                )
                g2 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2
                )
                print(f"contact {i}: {g1}, {g2} with distance {c.dist}")
            return True

    # no contact found
    return False


########## Visualization ##########
def render_mp4(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_sequence: list[np.ndarray],
    dt: float,
    file_path: str,
    height: int = 720,
    width: int = 1280,
    scene_option_flags: dict[int, bool] | None = None,
    cam_lookat: np.ndarray = np.array([0, 0, 1.0], dtype=float),
    cam_distance: float = 2.0,
    cam_angles: tuple[float, float] = (180.0, -45.0),
) -> list[np.ndarray]:
    """
    Render a sequence of qpos data into mp4 video

    qpos_sequence should be a list of qpos data that represents
    the qpos data at each time step
    """
    import imageio

    # Initialize renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    if scene_option_flags is not None:
        for flag, value in scene_option_flags.items():
            scene_option.flags[flag] = value
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_distance
    cam.azimuth = cam_angles[0]
    cam.elevation = cam_angles[1]

    # Render frames
    mujoco.mj_resetData(model, data)
    frames = []
    for qpos in qpos_sequence:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
    renderer.close()

    # Save frames to mp4
    frame_rate = 1 / dt
    with imageio.get_writer(file_path, fps=frame_rate) as writer:
        for frame in frames:
            writer.append_data(frame)

    return frames
