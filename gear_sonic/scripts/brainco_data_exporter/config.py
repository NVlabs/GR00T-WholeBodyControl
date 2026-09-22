"""CLI configuration for BrainCo dataset collection."""

from dataclasses import dataclass

from gear_sonic.scripts.run_data_exporter import SonicDataExporterConfig

@dataclass
class BraincoDataExporterConfig(SonicDataExporterConfig):
    """CLI configuration for native 6-DOF BrainCo dataset collection."""

    brainco_dds_domain_id: int = 0
    """DDS domain used by brainco_hand_service."""

    brainco_network_interface: str | None = "wlp128s20f3"
    """DDS network interface, for example eth0. None enables auto-selection."""

    brainco_message_timeout: float = 10.0
    """Seconds to wait for both hand state messages (0 = forever)."""

    depth_camera_rgb_image_key: str = "ego_view"
    """RGB image key published by the depth camera."""

    depth_camera_image_key: str = "ego_view_depth"
    """Depth image key published by the depth camera."""

    webcam_image_key: str = "head"
    """RGB image key published by the head camera."""

    head_depth_camera_image_key: str = "head_depth"
    """Depth image key published by the head RealSense camera."""

    external_view_camera_host: str | None = None
    """Host publishing external-view-camera over ZMQ. None disables recording it."""

    external_view_camera_port: int = 5582
    """ZMQ port of the standalone external-view-camera publisher."""

    external_view_camera_width: int = 640
    """Requested width of the host-local external-view camera."""

    external_view_camera_height: int = 480
    """Requested height of the host-local external-view camera."""

    external_view_camera_timeout: float = 10.0
    """Seconds to wait for the first external-view frame (0 = forever)."""

    ignore_ego_view: bool = False
    """Exclude ego-view RGB/depth streams from the dataset."""

    ignore_head: bool = False
    """Exclude head RGB/depth streams from the dataset."""

    depth_camera_width: int = 1280
    """Width of the recorded ego-view RGB stream."""

    depth_camera_height: int = 720
    """Height of the recorded ego-view RGB stream."""

    depth_stream_width: int = 640
    """Width of the recorded ego-view depth stream."""

    depth_stream_height: int = 480
    """Height of the recorded ego-view depth stream."""

    webcam_width: int = 1280
    """Width of the recorded head-camera RGB stream."""

    webcam_height: int = 720
    """Height of the recorded head-camera RGB stream."""

    head_depth_stream_width: int = 640
    """Width of the recorded head-camera depth stream."""

    head_depth_stream_height: int = 480
    """Height of the recorded head-camera depth stream."""

    depth_video_max_meters: float = 5.0
    """Depth mapped to white in the 8-bit depth visualization video."""

    record_depth_video: bool = False
    """Record enabled depth streams as 8-bit grayscale preview videos."""

    record_raw_depth: bool = False
    """Record enabled metric depth streams as uint16 millimetre PNG sequences."""

    g1_telemetry_zmq_host: str = "192.168.50.132"
    """Robot host publishing upper-body telemetry."""

    g1_telemetry_zmq_port: int = 5560
    """ZMQ port of the robot-side G1 telemetry publisher."""

    g1_telemetry_message_timeout: float = 10.0
    """Seconds to wait for initial G1 telemetry (0 = forever)."""

    g1_telemetry_max_age: float = 0.25
    """Maximum age in seconds of telemetry used in an aligned dataset frame."""

    g1_telemetry_expected_hz: float = 50.0
    """Expected robot publisher rate; used to size the host receive queue."""

    record_raw_telemetry: bool = True
    """Save every robot telemetry packet per episode under raw-telemetry/."""
