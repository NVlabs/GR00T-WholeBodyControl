"""Composition root for BrainCo dataset collection."""

from gear_sonic.data.features_sonic_vla import (
    get_g1_robot_model, get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.zmq_state_subscriber import poll_robot_config_zmq
from gear_sonic.g1_upper_body_telemetry.receiver import TelemetryReceiver

from .collector import BraincoGrootDataCollector
from .config import BraincoDataExporterConfig
from .constants import (
    BRAINCO_MOTOR_NAMES, BRAINCO_NUM_MOTORS, DEPTH_CAMERA_RGB_FEATURE,
    G1_DATASET_ACTION_FIELDS, G1_DATASET_OBSERVATION_FIELDS,
    G1_RAW_TELEMETRY_ARRAY_FIELDS, G1_UPPER_BODY_DAMPING_KD,
    G1_UPPER_BODY_JOINT_NAMES, G1_UPPER_BODY_STIFFNESS_KP, WEBCAM_FEATURE,
)
from .dds import BraincoHandDDSSubscriber
from .external_camera import ExternalViewCameraSubscriber
from .exporter import BraincoDataExporter
from .schema import (
    BraincoSchema, configure_camera_schema, get_brainco_features,
    get_brainco_modality_config,
)
from .session_profile import collect_session_profile

def main(config: BraincoDataExporterConfig) -> None:
    session_profile = collect_session_profile()
    config.task_prompt = session_profile.interaction_class
    config.dataset_name = session_profile.dataset_name()
    print(f"Dataset name: {config.dataset_name}")

    if config.ignore_ego_view and config.ignore_head:
        raise ValueError(
            "At least one camera must be enabled: --ignore-ego-view and "
            "--ignore-head cannot be used together"
        )
    enabled_camera_streams = tuple(
        stream
        for stream, ignored in (
            ("ego_view", config.ignore_ego_view),
            ("head", config.ignore_head),
        )
        if not ignored
    )

    robot_model = get_g1_robot_model()
    schema = BraincoSchema(robot_model)
    features = get_brainco_features(robot_model, schema)
    modality_config = get_brainco_modality_config(robot_model, schema)
    configure_camera_schema(features, modality_config, config)

    external_view_camera = None
    if config.external_view_camera_host is not None:
        external_view_camera = ExternalViewCameraSubscriber(
            host=config.external_view_camera_host,
            port=config.external_view_camera_port,
        )
        try:
            external_view_camera.wait_until_ready(config.external_view_camera_timeout)
        except Exception:
            external_view_camera.close()
            raise
        print(
            "[ExternalViewCamera] Recording ZMQ stream "
            f"tcp://{config.external_view_camera_host}:"
            f"{config.external_view_camera_port} as external-view-camera "
            f"({config.external_view_camera_width}x{config.external_view_camera_height})"
        )

    if config.record_wrist_cameras:
        features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            modality_config.setdefault(key, {}).update(value)

    print("config.brainco_network_interface",config.brainco_network_interface)
    brainco_subscriber = BraincoHandDDSSubscriber(
        domain_id=config.brainco_dds_domain_id,
        network_interface=config.brainco_network_interface,
    )
    try:
        brainco_subscriber.wait_until_ready(config.brainco_message_timeout)
    except Exception:
        brainco_subscriber.close()
        if external_view_camera is not None:
            external_view_camera.close()
        raise

    telemetry_receiver = TelemetryReceiver(
        config.g1_telemetry_zmq_host,
        config.g1_telemetry_zmq_port,
        expected_hz=config.g1_telemetry_expected_hz,
    )
    try:
        telemetry_receiver.wait_until_ready(config.g1_telemetry_message_timeout)
        # Startup packets precede the first episode and must not be persisted.
        telemetry_receiver.drain_raw()
    except Exception:
        telemetry_receiver.close()
        brainco_subscriber.close()
        if external_view_camera is not None:
            external_view_camera.close()
        raise

    try:
        robot_config = poll_robot_config_zmq(
            config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
        )
    except Exception:
        telemetry_receiver.close()
        brainco_subscriber.close()
        if external_view_camera is not None:
            external_view_camera.close()
        raise
    data_exporter = BraincoDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "session_profile": session_profile.to_dict(),
            "interaction_metadata_path": "meta/interaction_metadata.jsonl",
            "record_wrist_cameras": config.record_wrist_cameras,
            "enabled_camera_streams": list(enabled_camera_streams),
            "ignore_ego_view": config.ignore_ego_view,
            "ignore_head": config.ignore_head,
            "depth_camera_rgb_image_key": config.depth_camera_rgb_image_key,
            "depth_camera_image_key": config.depth_camera_image_key,
            "webcam_image_key": config.webcam_image_key,
            "head_depth_camera_image_key": config.head_depth_camera_image_key,
            "external_view_camera": {
                "enabled": config.external_view_camera_host is not None,
                "dataset_modality": "external-view-camera",
                "source": "standalone_zmq_publisher",
                "host": config.external_view_camera_host,
                "port": config.external_view_camera_port,
                "resolution": [
                    config.external_view_camera_width,
                    config.external_view_camera_height,
                ],
            },
            "depth_camera_resolution": [
                config.depth_camera_width,
                config.depth_camera_height,
            ],
            "depth_stream_resolution": [
                config.depth_stream_width,
                config.depth_stream_height,
            ],
            "webcam_resolution": [config.webcam_width, config.webcam_height],
            "head_depth_stream_resolution": [
                config.head_depth_stream_width,
                config.head_depth_stream_height,
            ],
            "record_depth_video": config.record_depth_video,
            "record_raw_depth": config.record_raw_depth,
            "depth_video_encoding": "grayscale_rgb8_preview",
            "depth_video_max_meters": config.depth_video_max_meters,
            "raw_depth_encoding": "uint16_mm_png",
            "raw_depth_root": {
                stream: ("depth" if stream == "ego_view" else "depth_head")
                for stream in enabled_camera_streams
            },
            "raw_depth_unit_meters": 0.001,
            "raw_depth_invalid_value": 0,
            "raw_depth_aligned_to": {
                stream: (
                    DEPTH_CAMERA_RGB_FEATURE if stream == "ego_view" else WEBCAM_FEATURE
                )
                for stream in enabled_camera_streams
            },
            "episode_grip_timestamps_key": BraincoDataExporter.GRIP_TIMESTAMPS_KEY,
            "episode_trigger_timestamp_keys": list(
                BraincoDataExporter.TRIGGER_TIMESTAMP_KEYS.values()
            ),
            "hand_type": "brainco_revo2",
            "hand_dof": BRAINCO_NUM_MOTORS,
            "hand_coordinates": "normalized_0_to_1",
            "hand_motor_order": list(BRAINCO_MOTOR_NAMES),
            "brainco_state_source": "dds_feedback",
            "brainco_tau_est_source": "MotorStates_.states[].tau_est",
            "brainco_action_source": "dds_command",
            "g1_upper_body_state_source": "rt/lowstate via robot ZMQ telemetry publisher",
            "g1_upper_body_command_source": "rt/lowcmd via robot ZMQ telemetry publisher",
            "g1_upper_body_telemetry_zmq_port": config.g1_telemetry_zmq_port,
            "g1_upper_body_telemetry_expected_hz": config.g1_telemetry_expected_hz,
            "g1_upper_body_residual_sign": "estimated_minus_commanded",
            "g1_upper_body_dataset_action_fields": list(G1_DATASET_ACTION_FIELDS),
            "g1_upper_body_dataset_observation_fields": list(
                G1_DATASET_OBSERVATION_FIELDS
            ),
            "record_raw_telemetry": config.record_raw_telemetry,
            "raw_telemetry_root": "raw-telemetry",
            "raw_telemetry_encoding": "numpy_npz_compressed",
            "raw_telemetry_array_fields": list(G1_RAW_TELEMETRY_ARRAY_FIELDS),
            "g1_upper_body_motor_indices": list(range(12, 29)),
            "g1_upper_body_joint_names": list(G1_UPPER_BODY_JOINT_NAMES),
            "g1_upper_body_gain_source": "rt/lowcmd motor_cmd[].kp/kd per sample",
            "g1_upper_body_reference_stiffness_kp": list(G1_UPPER_BODY_STIFFNESS_KP),
            "g1_upper_body_reference_damping_kd": list(G1_UPPER_BODY_DAMPING_KD),
            "g1_upper_body_reference_gain_source": (
                "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
                "policy_parameters.hpp"
            ),
            "g1_upper_body_reference_gain_formula": {
                "stiffness_kp": "armature * (2*pi*10Hz)^2",
                "damping_kd": "2 * damping_ratio(2.0) * armature * (2*pi*10Hz)",
            },
        },
    )
    data_exporter.record_raw_depth = config.record_raw_depth
    data_exporter.enabled_camera_streams = enabled_camera_streams
    data_exporter.record_raw_telemetry = config.record_raw_telemetry
    data_exporter.session_profile = session_profile.to_dict()

    text_to_speech = TextToSpeech() if config.text_to_speech else None
    collector = BraincoGrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=robot_model,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        episode_status_zmq_port=config.episode_status_zmq_port,
        episode_status_publish_hz=config.episode_status_publish_hz,
        brainco_subscriber=brainco_subscriber,
        brainco_schema=schema,
        depth_camera_rgb_image_key=config.depth_camera_rgb_image_key,
        depth_camera_image_key=config.depth_camera_image_key,
        webcam_image_key=config.webcam_image_key,
        head_depth_camera_image_key=config.head_depth_camera_image_key,
        depth_video_max_meters=config.depth_video_max_meters,
        record_raw_depth=config.record_raw_depth,
        enabled_camera_streams=enabled_camera_streams,
        telemetry_receiver=telemetry_receiver,
        telemetry_max_age=config.g1_telemetry_max_age,
        record_raw_telemetry=config.record_raw_telemetry,
        external_view_camera=external_view_camera,
    )
    collector.run()
