def tensor_needed(motion_rep):
    needs = set()

    for feature in motion_rep.body_keys:
        if feature in ["ric_data", "local_vel", "foot_contacts"]:
            needs.add("posed_joints")
        elif feature in ["rot_data"]:
            needs.add("local_joint_rots")
        elif feature in ["global_rot_data"]:
            needs.add("global_joint_rots")
        else:
            raise ValueError("This body feature is not recognised")
    return needs
