IMAGE_TOPIC_NAME = "realsense/color/image_raw"
STATE_TOPIC_NAME = "G1Env/env_state_act"
CONTROL_GOAL_TOPIC = "ControlPolicy/upper_body_pose"
ROBOT_CONFIG_TOPIC = "WBCPolicy/robot_config"
KEYBOARD_INPUT_TOPIC = "/keyboard_input"
LOCO_MANIP_TASK_STATUS_TOPIC = "LocoManipPolicy/task_status"
LOCO_NAV_TASK_STATUS_TOPIC = "NavigationPolicy/task_status"
LOWER_BODY_POLICY_STATUS_TOPIC = "ControlPolicy/lower_body_policy_status"
JOINT_SAFETY_STATUS_TOPIC = "ControlPolicy/joint_safety_status"


DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]
DEFAULT_BASE_HEIGHT = 0.74
DEFAULT_WRIST_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2  # x, y, z + w, x, y, z

DEFAULT_MODEL_SERVER_PORT = 5555  # port used to host the model server
