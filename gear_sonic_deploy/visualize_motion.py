import argparse
import csv
import os
import time
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
import numpy as np

from lxml import etree

import zmq
import threading
import msgpack

def key_call_back(keycode):
    global \
        curr_start, \
        num_motions, \
        motion_id, \
        motion_acc, \
        time_step, \
        dt, \
        paused, \
        data_csv_dict, \
        frame_idx, \
        anim_idx
    
    try:
        c = chr(keycode)
    except:
        c = ""
    if c == "R":
        print("Reset")
        frame_idx = int(0)
    elif c == " ":
        print("Paused")
        paused = not paused
    elif c == ".":
        frame_idx = frame_idx + 1
        print("frame", frame_idx)
    elif c == ",":
        frame_idx = frame_idx - 1
        print("frame", frame_idx)
    elif c == "=":
        anim_idx = anim_idx + 1
        print("anim", anim_idx)
    elif c == "-":
        anim_idx = anim_idx - 1
        print("anim", anim_idx)
    else:
        print("not mapped", c)


def load_anim_data(csv_path: str):

    ret = []
    if os.path.isdir(csv_path):

        joint_pos_path = os.path.join(csv_path, "joint_pos.csv")
        body_pos_path = os.path.join(csv_path, "body_pos.csv")
        body_quat_path = os.path.join(csv_path, "body_quat.csv")

        isaaclab_to_mujoco = [0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                              11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

        with open(joint_pos_path, mode="r", newline="") as joint_pos_file, open(body_pos_path, mode="r", newline="") as body_pos_file, open(body_quat_path, mode="r", newline="") as body_quat_file:
            firstRow = True
            joint_pos_rowlist = []
            body_pos_rowlist = []
            body_quat_rowlist = []
            for joint_pos_row, body_pos_row, body_quat_row in zip(joint_pos_file, body_pos_file, body_quat_file):
                if firstRow:
                    firstRow = False
                    continue
                
                joint_pos_row = np.array([float(x) for x in joint_pos_row.split(",")])
                body_pos_row = np.array([float(x) for x in body_pos_row.split(",")])
                body_quat_row = np.array([float(x) for x in body_quat_row.split(",")])

                joint_pos_rowlist.append(joint_pos_row)
                body_pos_rowlist.append(body_pos_row)
                body_quat_rowlist.append(body_quat_row)

            ret.append({
                "dof": np.array(joint_pos_rowlist)[:, isaaclab_to_mujoco],
                "root_rot": np.array(body_quat_rowlist)[:, [0, 1, 2, 3]],  # [x, y, z, w]
                "root_trans_offset": np.array(body_pos_rowlist)[:, :3],
            })

    else:
        csv_data = []
        current_rowlist = []
        with open(csv_path, mode="r", newline="") as file:

            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row):
                    r = [x for x in row if x]
                    assert len(r) == 36
                    current_rowlist.append(r)
                else:
                    csv_data.append(current_rowlist)
                    current_rowlist = []

            if current_rowlist:
                csv_data.append(current_rowlist)

        for d in csv_data:
            ret.append({
                "dof": np.array(d)[:, 7:],
                "root_rot": np.array(d)[:, 3:7][:, [0, 1, 2, 3]],  # [x, y, z, w]
                "root_trans_offset": np.array(d)[:, :3],
            })

    return ret


def receive_realtime_debug_messages(socket, data_csv_dicts, topic):
    while True:
        message = socket.recv()

        # Remove any header or leading bytes (should be exactly 8 bytes for "g1_debug")
        data = message.split(topic.encode())[1]

        result = msgpack.unpackb(data)

        data_csv_dicts[0]["root_trans_offset"][0, ...] = result["base_trans_target"]
        data_csv_dicts[0]["root_rot"][0, ...] = result["base_quat_target"]
        data_csv_dicts[0]["dof"][0, ...] = result["body_q_target"]

        data_csv_dicts[0]["root_trans_offset_measured"][0, ...] = result["base_trans_measured"]
        data_csv_dicts[0]["root_rot_measured"][0, ...] = result["base_quat_measured"]
        data_csv_dicts[0]["dof_measured"][0, ...] = result["body_q_measured"]

        data_csv_dicts[0]["vr_3point_position"] = np.array(result["vr_3point_position"]).reshape(3,3)
        data_csv_dicts[0]["vr_3point_orientation"] = np.array(result["vr_3point_orientation"]).reshape(3,4)
        data_csv_dicts[0]["vr_3point_compliance"] = np.array(result["vr_3point_compliance"]).reshape(3)

def main(args) -> None:
    global \
        curr_start, \
        num_motions, \
        motion_id, \
        motion_acc, \
        time_step, \
        dt, \
        paused, \
        data_csv_dict, \
        frame_idx, \
        anim_idx
        
    fps = 50
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, frame_idx, anim_idx = 0, 1, 0, set(), 0, 1 / fps, False, int(0), 0

    def prepend_names(elem, prefix):
        # If element has a 'name' attribute, prepend the prefix
        if 'name' in elem.attrib:
            elem.attrib['name'] = prefix + elem.attrib['name']
        # Recurse for all child elements
        for child in elem:
            prepend_names(child, prefix)

    def replace_attribute(elem, attribute, value):
        # If element has a 'name' attribute, prepend the prefix
        if attribute in elem.attrib:
            elem.attrib[attribute] = value
        # Recurse for all child elements
        for child in elem:
            replace_attribute(child, attribute, value)

    main_scene = etree.parse('g1/scene_empty.xml')
    robot1 = etree.parse('g1/g1_29dof_old.xml')
    robot_asset = robot1.find('asset')
    scene_asset = main_scene.find('asset')
    for mesh in robot_asset.findall('mesh'):
        # INSERT_YOUR_CODE
        mesh.set("file", os.path.join("g1","meshes", mesh.get('file')))
        scene_asset.append(mesh)
    
    robot_default = robot1.find('default')
    scene_default = main_scene.find('default')
    for default in robot_default.findall('default'):
        scene_default.append(default)

    scene_worldbody = main_scene.find('worldbody')
    robot1_body = robot1.find('worldbody').find('body')
    prepend_names(robot1_body, "robot1_")
    scene_worldbody.append(robot1_body)

    robot2 = etree.parse('g1/g1_29dof_old.xml')
    robot2_body = robot2.find('worldbody').find('body')
    prepend_names(robot2_body, "robot2_")
    replace_attribute(robot2_body, "rgba", "0.5 0.1 0.1 1")
    robot2_body.set("pos", "0 -1 -10")
    scene_worldbody.append(robot2_body)

    robot3 = etree.parse('g1/g1_29dof_old.xml')
    robot3_body = robot3.find('worldbody').find('body')
    prepend_names(robot3_body, "robot3_")
    replace_attribute(robot3_body, "rgba", "0.1 0.5 0.1 0.2")
    robot3_body.set("pos", "0 -2 -10")
    scene_worldbody.append(robot3_body)
    
    mj_model = mujoco.MjModel.from_xml_string(etree.tostring(main_scene, pretty_print=True, encoding="unicode"))
    mj_data = mujoco.MjData(mj_model)

    # Disable advanced visual effects for better performance
    mj_model.vis.global_.offwidth = 1920
    mj_model.vis.global_.offheight = 1080
    mj_model.vis.quality.shadowsize = 0  # Disable shadows
    mj_model.vis.quality.offsamples = 1  # Reduce anti-aliasing
    mj_model.vis.rgba.fog = [0, 0, 0, 0]  # Disable fog

    # Disable advanced lighting effects
    mj_model.vis.headlight.ambient = [0.8, 0.8, 0.8]  # Increase ambient light
    mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]  # Increase diffuse light
    mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]  # Reduce specular highlights
    
    if args.realtime_debug_url:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(args.realtime_debug_url)
        socket.setsockopt(zmq.SUBSCRIBE, args.realtime_debug_topic.encode())

        data_csv_dicts = [{
            "dof": np.zeros((1,29), dtype=np.float64),
            "root_rot": np.array([[0.0, 0.0, 0.0, 1.0]]),  # [x, y, z, w]
            "root_trans_offset": np.array([[0.0, 0.0, .9]], dtype=np.float64),
            "dof_measured": np.zeros((1,29), dtype=np.float64),
            "root_rot_measured": np.array([[0.0, 0.0, 0.0, 1.0]]),
            "root_trans_offset_measured": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            "vr_3point_position": np.zeros((3,3), dtype=np.float64),
            "vr_3point_orientation": np.zeros((3,4), dtype=np.float64),
            "vr_3point_compliance": np.zeros((3), dtype=np.float64),
        }]

        threading.Thread(target=receive_realtime_debug_messages, args=(socket, data_csv_dicts, args.realtime_debug_topic)).start()

    elif args.motion_dir:
        data_csv_dicts = load_anim_data(args.motion_dir)
    elif args.csv_path:
        data_csv_dicts = load_anim_data(args.csv_path)
    else:
        raise ValueError("Either --realtime_debug_url, --motion_dir, or --csv_path must be provided")

    RECORDING = False
    mj_model.opt.timestep = dt
    try:
        context = mujoco.GLContext(1920, 1080)
        context.make_current()
        print("✓ GPU acceleration enabled")
    except Exception as e:
        print(f"✗ GPU acceleration not available: {e}")
        context = None

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_call_back,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Set camera position to be further away
        viewer.cam.distance = 15.0  # Increase distance from the scene
        viewer.cam.azimuth = 90.0  # Set azimuth angle
        viewer.cam.elevation = -20.0  # Set elevation angle
        
        while viewer.is_running():
            motion_len = data_csv_dicts[anim_idx % len(data_csv_dicts)]["dof"].shape[0]
            step_start = time.time()
            time_idx = frame_idx % motion_len
            data_dict = data_csv_dicts[anim_idx % len(data_csv_dicts)]
            mj_data.qpos[:3] = data_dict["root_trans_offset"][time_idx]
            mj_data.qpos[3:7] = data_dict["root_rot"][time_idx]
            mj_data.qpos[7:7+29] = data_dict["dof"][time_idx]

            if "dof_measured" in data_dict:
                mj_data.qpos[36:36+3] = data_dict["root_trans_offset_measured"][time_idx]
                mj_data.qpos[39:39+4] = data_dict["root_rot_measured"][time_idx]
                mj_data.qpos[43:43+29] = data_dict["dof_measured"][time_idx]


                mj_data.qpos[43+29:43+29+3] = data_dict["root_trans_offset_measured"][time_idx]
                mj_data.qpos[43+29+3:43+29+3+4] = data_dict["root_rot"][time_idx]
                mj_data.qpos[43+29+3+4:43+29+3+4+29] = data_dict["dof"][time_idx]

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                frame_idx += 1
            
            viewer.user_scn.ngeom = 0
            if "vr_3point_position" in data_dict:
                # Get root pose for transforming root-relative coordinates to world space
                # VR 3-point data from C++ is normalized relative to root (see g1_deploy_onnx_ref.cpp)
                root_trans = data_dict["root_trans_offset_measured"][time_idx]
                root_quat_wxyz = data_dict["root_rot_measured"][time_idx]  # [w, x, y, z] format (MuJoCo/C++ convention)
                root_rot = R.from_quat(root_quat_wxyz, scalar_first=True)
                
                for i in range(3):
                    # VR 3-point position is in root-relative coordinates, transform to world
                    vr_pos_root_frame = data_dict["vr_3point_position"][i]
                    # vr_pos_world = root_trans + root_rot.apply(vr_pos_root_frame)
                    vr_pos_world = vr_pos_root_frame + data_dict["root_trans_offset_measured"][time_idx]
                    
                    if np.linalg.norm(data_dict["vr_3point_orientation"][i]) > 0:
                        # VR orientation is also root-relative, transform to world
                        # C++ quaternion is in [w, x, y, z] format (scalar_first=True)
                        vr_quat_root_frame = R.from_quat(data_dict["vr_3point_orientation"][i], scalar_first=True)
                        vr_rot_world = root_rot * vr_quat_root_frame  # Quaternion multiplication
                        mat = vr_rot_world.as_matrix()
                    else:
                        mat = root_rot.as_matrix()  # If no VR orientation, use root orientation

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.05, 0.01, 0.01],
                        pos=vr_pos_world,
                        mat=mat.flatten(),
                        rgba=0.5*np.array([1, 1, 0, 2])
                    )
                    viewer.user_scn.ngeom += 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize retargeted motion data in MuJoCo"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to the CSV file containing retargeted motion data",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="",
        help="Path to the CSV file containing retargeted motion data",
    )
    parser.add_argument(
        "--realtime_debug_url",
        type=str,
        default="",
        help="URL to receive realtime debug messages from",
    )
    parser.add_argument(
        "--realtime_debug_topic",
        type=str,
        default="g1_debug",
        help="Topic to receive realtime debug messages from",
    )
    args = parser.parse_args()

    main(args)