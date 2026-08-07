"""
Collection of constants for cameras / robots / etc
in kitchen environments
"""

import numpy as np
import matplotlib.pyplot as plt


# https://github.com/yusukeurakami/mujoco_2d_projection
def global2label(obj_pos, cam_pos, cam_ori, output_size=[64, 64], fov=90, s=1):
    """
    :param obj_pos: 3D coordinates of the joint from MuJoCo in nparray [m]
    :param cam_pos: 3D coordinates of the camera from MuJoCo in nparray [m]
    :param cam_ori: camera 3D rotation (Rotation order of x->y->z) from MuJoCo in nparray [rad]
    :param fov: field of view in integer [degree]
    :return: Heatmap of the object in the 2D pixel space.
    """

    e = np.array([output_size[0] / 2, output_size[1] / 2, 1])
    fov = np.array([fov])

    # Converting the MuJoCo coordinate into typical computer vision coordinate.
    cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
    obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
    cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

    obj_pos_in_2D, obj_pos_from_cam = get_2D_from_3D(obj_pos_cv, cam_pos_cv, cam_ori_cv, fov, e)
    label = gkern(
        output_size[0],
        output_size[1],
        (obj_pos_in_2D[1], output_size[0] - obj_pos_in_2D[0]),
        sigma=s,
    )
    return label


def get_2D_from_3D(a, c, theta, fov, e):
    """
    :param a: 3D coordinates of the joint in nparray [m]
    :param c: 3D coordinates of the camera in nparray [m]
    :param theta: camera 3D rotation (Rotation order of x->y->z) in nparray [rad]
    :param fov: field of view in integer [degree]
    :param e:
    :return:
        - (bx, by) ==> 2D coordinates of the obj [pixel]
        - d ==> 3D coordinates of the joint (relative to the camera) [m]
    """

    # Get the vector from camera to object in global coordinate.
    ac_diff = a - c

    # Rotate the vector in to camera coordinate
    x_rot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), np.sin(theta[0])],
            [0, -np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    y_rot = np.array(
        [
            [np.cos(theta[1]), 0, -np.sin(theta[1])],
            [0, 1, 0],
            [np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    z_rot = np.array(
        [
            [np.cos(theta[2]), np.sin(theta[2]), 0],
            [-np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    transform = z_rot.dot(y_rot.dot(x_rot))
    d = transform.dot(ac_diff)

    # scaling of projection plane using fov
    fov_rad = np.deg2rad(fov)
    e[2] *= e[0] * 1 / np.tan(fov_rad / 2.0)

    # Projection from d to 2D
    bx = e[2] * d[0] / (d[2]) + e[0]
    by = e[2] * d[1] / (d[2]) + e[1]

    return (bx, by), d


def gkern(h, w, center, sigma=1):
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def compute_2d_projection(
    obj_pos=np.array([0.2, 0.25, 1.0]),
    cam_pos=np.array([0.7, 0, 1.5]),
    cam_ori=np.array([0.2, 1.2, 1.57]),
    fov=90,
    output_size=[64, 64],
):
    e = np.array([output_size[0] / 2, output_size[1] / 2, 1])
    fov = np.array([fov])

    # Converting the MuJoCo coordinate into typical computer vision coordinate.
    cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
    obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
    cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

    obj_pos_in_2D, obj_pos_from_cam = get_2D_from_3D(obj_pos_cv, cam_pos_cv, cam_ori_cv, fov, e)

    return obj_pos_in_2D[1], output_size[0] - obj_pos_in_2D[0]


def global2label(obj_pos, cam_pos, cam_ori, output_size=[64, 64], fov=90, s=1):
    """
    :param obj_pos: 3D coordinates of the joint from MuJoCo in nparray [m]
    :param cam_pos: 3D coordinates of the camera from MuJoCo in nparray [m]
    :param cam_ori: camera 3D rotation (Rotation order of x->y->z) from MuJoCo in nparray [rad]
    :param fov: field of view in integer [degree]
    :return: Heatmap of the object in the 2D pixel space.
    """

    e = np.array([output_size[0] / 2, output_size[1] / 2, 1])
    fov = np.array([fov])

    # Converting the MuJoCo coordinate into typical computer vision coordinate.
    cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
    obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
    cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

    obj_pos_in_2D, obj_pos_from_cam = get_2D_from_3D(obj_pos_cv, cam_pos_cv, cam_ori_cv, fov, e)
    label = gkern(
        output_size[0],
        output_size[1],
        (obj_pos_in_2D[1], output_size[0] - obj_pos_in_2D[0]),
        sigma=s,
    )
    return label


def visualize_2d_projection(
    obj_pos=np.array([0.2, 0.25, 1.0]),
    cam_pos=np.array([0.7, 0, 1.5]),
    cam_ori=np.array([0.2, 1.2, 1.57]),
    fov=90,
    output_size=[64, 64],
):
    s = 1  # std for heapmap signal
    label = global2label(obj_pos, cam_pos, cam_ori, output_size, fov=fov, s=s)
    plt.imshow(label)
    plt.show()


# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    0: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208,
    ),
    1: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    2: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    # 3: dict(
    #     lookat=[11.44842548, -11.47664723, 11.24115989],
    #     distance=43.923271794728187,
    #     azimuth=227.12928449329333,
    #     elevation=-16.495686334624907,
    # ),
    4: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}

CAM_CONFIGS = dict(
    robot0_agentview_center=dict(
        pos=[-0.6, 0.0, 1.15],
        quat=[
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_left=dict(
        pos=[-0.5, 0.35, 1.05],
        quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_agentview_right=dict(
        pos=[-0.5, -0.35, 1.05],
        quat=[
            0.6775091886520386,
            0.3767866790294647,
            -0.2993525564670563,
            -0.55623859167099,
        ],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_frontview=dict(
        pos=[-0.50, 0, 0.95],
        quat=[
            0.6088936924934387,
            0.3814677894115448,
            -0.3673907518386841,
            -0.5905545353889465,
        ],
        camera_attribs=dict(fovy="60"),
        parent_body="mobilebase0_support",
    ),
    robot0_eye_in_hand=dict(
        pos=[0.05, 0, 0],
        quat=[0, 0.707107, 0.707107, 0],
        parent_body="robot0_right_hand",
    ),
)
