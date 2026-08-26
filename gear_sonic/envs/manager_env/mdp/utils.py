"""
Utility functions for manager environment MDP, including debug visualization.
"""

import torch


def debug_visualize_object_projection(
    env, camera, rgb_image, debug_env_idx: int = 0, show_predicted: bool = False
):
    """
    Debug visualization: Project ground truth and predicted object positions onto camera image.

    This function computes the camera pose from the robot's d435_link and camera offset,
    transforms the object position to the camera frame, and projects it to 2D pixel coordinates.

    The coordinate transformation pipeline:
    1. Get object position in robot base (pelvis) frame
    2. Get d435_link pose in robot base frame
    3. Apply camera offset (from config) + random extrinsics delta
    4. Transform object from base frame to camera frame
    5. Map camera axes to OpenCV convention:
       - x_cv = -camera_Y  (camera left → OpenCV right)
       - y_cv = -camera_Z  (camera up → OpenCV down)
       - z_cv = camera_X   (camera forward → OpenCV depth)
    6. Project using pinhole camera model: u = fx*x/z + cx, v = fy*y/z + cy

    Args:
        env: The environment object
        camera: The TiledCamera sensor
        rgb_image: RGB image tensor [num_envs, H, W, 3]
        debug_env_idx: Which environment to visualize
        show_predicted: Whether to also visualize predicted object position (in red)
    """
    import cv2
    from isaaclab.utils.math import (
        quat_apply,
        quat_apply_inverse,
        quat_conjugate,
        quat_from_euler_xyz,
        quat_mul,
    )

    # Get the image for the specified environment
    debug_img = rgb_image[debug_env_idx].detach().cpu().numpy()
    if debug_img.max() <= 1.0:
        debug_img = (debug_img * 255).astype("uint8")
    else:
        debug_img = debug_img.astype("uint8")
    debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    H, W = debug_img_bgr.shape[:2]

    try:
        # Find object in scene
        object_name = None
        if hasattr(env.scene, "rigid_objects"):
            for name in env.scene.rigid_objects.keys():
                if "object" in name.lower() or "obj" in name.lower():
                    object_name = name
                    break

        if object_name is not None and "d435_link" in env.scene["robot"].body_names:
            robot = env.scene["robot"]
            object_asset = env.scene[object_name]
            device = robot.data.root_pos_w.device

            # Step 1: Object position in robot base frame
            obj_pos_w = object_asset.data.root_pos_w[debug_env_idx]
            robot_pos_w = robot.data.root_pos_w[debug_env_idx]
            robot_quat_w = robot.data.root_quat_w[debug_env_idx]
            obj_rel_w = obj_pos_w - robot_pos_w
            obj_pos_base = quat_apply_inverse(robot_quat_w.unsqueeze(0), obj_rel_w.unsqueeze(0))[0]

            # Step 2: d435_link pose in robot base frame
            link_idx = robot.body_names.index("d435_link")
            link_pos_w = robot.data.body_link_pos_w[debug_env_idx, link_idx]
            link_quat_w = robot.data.body_link_quat_w[debug_env_idx, link_idx]
            link_rel_w = link_pos_w - robot_pos_w
            link_pos_base = quat_apply_inverse(robot_quat_w.unsqueeze(0), link_rel_w.unsqueeze(0))[
                0
            ]
            link_quat_base = quat_mul(
                quat_conjugate(robot_quat_w.unsqueeze(0)), link_quat_w.unsqueeze(0)
            )[0]

            # Step 3: Camera extrinsics (offset from d435_link)
            base_pos_offset = [0.0, 0.0, 0.0]
            base_rot_offset_quat = [1.0, 0.0, 0.0, 0.0]
            if hasattr(camera, "cfg") and hasattr(camera.cfg, "offset"):
                offset_cfg = camera.cfg.offset
                if hasattr(offset_cfg, "pos"):
                    base_pos_offset = list(offset_cfg.pos)
                if hasattr(offset_cfg, "rot"):
                    base_rot_offset_quat = list(offset_cfg.rot)

            # Random extrinsics delta from wrapper
            pos_delta = [0.0, 0.0, 0.0]
            rot_delta = [0.0, 0.0, 0.0]
            if hasattr(env, "wrapper") and hasattr(env.wrapper, "_camera_random_deltas"):
                deltas = env.wrapper._camera_random_deltas.get(debug_env_idx, None)
                if deltas is not None:
                    pos_delta = deltas["pos_delta"]
                    rot_delta = [deltas["roll_delta"], deltas["pitch_delta"], deltas["yaw_delta"]]

            cam_offset_pos = torch.tensor(
                [
                    base_pos_offset[0] + pos_delta[0],
                    base_pos_offset[1] + pos_delta[1],
                    base_pos_offset[2] + pos_delta[2],
                ],
                device=device,
                dtype=torch.float32,
            )
            base_rot_quat = torch.tensor(base_rot_offset_quat, device=device, dtype=torch.float32)

            # Step 4: Camera pose in robot base frame
            cam_offset_world_base = quat_apply(
                link_quat_base.unsqueeze(0), cam_offset_pos.unsqueeze(0)
            )[0]
            cam_pos_base = link_pos_base + cam_offset_world_base
            cam_quat_base = quat_mul(link_quat_base.unsqueeze(0), base_rot_quat.unsqueeze(0))[0]
            if any(r != 0 for r in rot_delta):
                delta_quat = quat_from_euler_xyz(
                    torch.tensor([rot_delta[0]], device=device),
                    torch.tensor([rot_delta[1]], device=device),
                    torch.tensor([rot_delta[2]], device=device),
                )[0]
                cam_quat_base = quat_mul(cam_quat_base.unsqueeze(0), delta_quat.unsqueeze(0))[0]

            # Step 5: Object in camera frame
            obj_rel_cam = obj_pos_base - cam_pos_base
            obj_pos_cam = quat_apply_inverse(cam_quat_base.unsqueeze(0), obj_rel_cam.unsqueeze(0))[
                0
            ]

            # Step 6: Map to OpenCV convention
            # Camera frame: X=forward, Y=left, Z=up
            # OpenCV frame: X=right, Y=down, Z=depth
            x_cv = -obj_pos_cam[1].item()  # -Y (left→right)
            y_cv = -obj_pos_cam[2].item()  # -Z (up→down)
            z_cv = obj_pos_cam[0].item()  # X (forward→depth)

            # Step 7: Project using intrinsics
            K = camera.data.intrinsic_matrices[debug_env_idx].cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            if z_cv > 0.01:  # Object in front of camera
                u = int(fx * x_cv / z_cv + cx)
                v = int(fy * y_cv / z_cv + cy)

                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(debug_img_bgr, (u, v), 12, (0, 255, 0), 2)
                    cv2.circle(debug_img_bgr, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(
                        debug_img_bgr,
                        f"GT d={z_cv:.2f}m",
                        (u + 15, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                else:
                    u_c = max(10, min(W - 10, u))
                    v_c = max(10, min(H - 10, v))
                    cv2.circle(debug_img_bgr, (u_c, v_c), 8, (0, 0, 255), -1)
                    cv2.putText(
                        debug_img_bgr,
                        "OOB",
                        (u_c + 10, v_c),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
            else:
                cv2.putText(
                    debug_img_bgr,
                    "Object behind camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Visualize predicted object position (in RED) if available
            # NOTE: pred_pos is in robot BASE (pelvis) frame, same as GT during training
            if show_predicted and hasattr(env, "wrapper"):
                pred_pos = env.wrapper._last_predicted_object_pos
                if pred_pos is not None and pred_pos.shape[0] > debug_env_idx:
                    # pred_pos is in robot BASE (pelvis) frame - need to transform to camera frame
                    pred_pos_base = pred_pos[debug_env_idx]

                    # Transform from base frame to camera frame (same as GT)
                    pred_rel_cam = pred_pos_base - cam_pos_base
                    pred_pos_cam = quat_apply_inverse(
                        cam_quat_base.unsqueeze(0), pred_rel_cam.unsqueeze(0)
                    )[0]

                    # Map to OpenCV convention (same as GT)
                    pred_x_cv = -pred_pos_cam[1].item()  # -Y (left→right)
                    pred_y_cv = -pred_pos_cam[2].item()  # -Z (up→down)
                    pred_z_cv = pred_pos_cam[0].item()  # X (forward→depth)

                    if pred_z_cv > 0.01:  # Predicted object in front of camera
                        pred_u = int(fx * pred_x_cv / pred_z_cv + cx)
                        pred_v = int(fy * pred_y_cv / pred_z_cv + cy)

                        if 0 <= pred_u < W and 0 <= pred_v < H:
                            # Draw RED circle for predicted position
                            cv2.circle(debug_img_bgr, (pred_u, pred_v), 12, (0, 0, 255), 2)
                            cv2.circle(debug_img_bgr, (pred_u, pred_v), 4, (0, 0, 255), -1)
                            cv2.putText(
                                debug_img_bgr,
                                f"PRED d={pred_z_cv:.2f}m",
                                (pred_u + 15, pred_v + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                        else:
                            # Out of bounds - draw at edge
                            pred_u_c = max(10, min(W - 10, pred_u))
                            pred_v_c = max(10, min(H - 10, pred_v))
                            cv2.circle(debug_img_bgr, (pred_u_c, pred_v_c), 8, (255, 0, 255), -1)
                            cv2.putText(
                                debug_img_bgr,
                                "PRED OOB",
                                (pred_u_c + 10, pred_v_c),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 0, 255),
                                1,
                            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        cv2.putText(
            debug_img_bgr,
            f"Error: {str(e)[:50]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    # Save and display
    cv2.imwrite("/tmp/camera_debug.png", debug_img_bgr)
    try:
        cv2.imshow(f"Ego Camera (env {debug_env_idx})", debug_img_bgr)
        cv2.waitKey(1)
    except Exception:
        pass
