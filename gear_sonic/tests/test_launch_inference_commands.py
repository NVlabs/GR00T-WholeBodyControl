import importlib
import sys
import types
from pathlib import Path


sys.modules.setdefault("tyro", types.SimpleNamespace(cli=lambda _config_type: None))
launch_inference = importlib.import_module("gear_sonic.scripts.launch_inference")


def test_sim_command_disables_image_publish_with_episode_video_override() -> None:
    config = launch_inference.InferenceLaunchConfig(
        sim=True,
        episode_video_path="/tmp/episode.mp4",
    )

    command = launch_inference._build_sim_command(Path("/repo"), config)

    assert "--enable-image-publish" not in command
    assert "--enable-offscreen" in command
    assert "--camera-port 5555" in command


def test_episode_video_camera_command_uses_composed_camera_ego_view_mp4() -> None:
    config = launch_inference.InferenceLaunchConfig(
        sim=True,
        episode_video_path="/tmp/episode.mp4",
        camera_port=5559,
    )

    command = launch_inference._build_episode_video_camera_command(Path("/repo"), config)

    assert "PYTHONPATH=. python -m gear_sonic.camera.composed_camera" in command
    assert "--ego-view-camera /tmp/episode.mp4" in command
    assert "--port 5559" in command


def test_inference_command_passes_calib_full_initial_pose() -> None:
    config = launch_inference.InferenceLaunchConfig(
        initial_pose="calib_full",
        initial_pose_ramp_s=2.0,
    )

    command = launch_inference._build_inference_command(Path("/repo"), config)

    assert "PYTHONPATH=. python gear_sonic/scripts/run_vla_inference.py" in command
    assert "--initial-pose calib_full" in command
    assert "--initial-pose-ramp-s 2.0" in command
    assert "--standing-ramp-s 2.0" in command


def test_deploy_command_uses_real_mode_by_default() -> None:
    config = launch_inference.InferenceLaunchConfig()

    command = launch_inference._build_deploy_command(Path("/repo"), config)

    assert "cd /repo/gear_sonic_deploy" in command
    assert "./deploy.sh" in command
    assert "--input-type zmq_manager" in command
    assert "--zmq-host localhost" in command
    assert command.endswith(" real")


def test_no_deploy_config_skips_local_deploy_start() -> None:
    config = launch_inference.InferenceLaunchConfig(deploy=False)

    assert launch_inference._should_start_deploy(config) is False


def test_no_deploy_inference_command_targets_external_deploy_defaults() -> None:
    config = launch_inference.InferenceLaunchConfig(
        deploy=False,
        camera_host="192.168.0.223",
    )

    command = launch_inference._build_inference_command(Path("/repo"), config)

    assert "--state-zmq-host 192.168.0.223" in command
    assert "--state-zmq-port 5557" in command
    assert "--action-zmq-host '*'" in command
    assert "--action-zmq-port 5556" in command
