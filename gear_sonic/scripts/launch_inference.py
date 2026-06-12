"""
All-in-one tmux launcher for SONIC VLA inference.

Starts the inference stack in a single tmux session:

    Window 0 — inference (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: VLA Inference │
    │ (gear_sonic_deploy)   │ (.venv_inference)     │
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: Keyboard Pub  │ Pane 3: Data Exporter │
    │ (.venv_inference)     │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_inference.sh     -> .venv_inference
        bash install_scripts/install_data_collection.sh -> .venv_data_collection (optional, for recording)
    - gear_sonic_deploy built (see docs)
    - Isaac-GR00T PolicyServer running separately

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_inference.py                        # real robot
    python gear_sonic/scripts/launch_inference.py --sim                  # MuJoCo sim
    python gear_sonic/scripts/launch_inference.py --no-data-exporter     # no recording pane
    python gear_sonic/scripts/launch_inference.py --no-deploy            # deploy runs elsewhere
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import os
import shutil
import signal
import socket
import base64
import subprocess
import sys
import textwrap
import time
from typing import Literal


def _bootstrap_venv():
    """Re-exec with the .venv_inference Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_inference" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_inference not found.\n"
            "  Run: bash install_scripts/install_inference.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class InferenceLaunchConfig:
    """CLI config for the all-in-one VLA inference tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim instead of real robot."""

    # C++ deploy options
    deploy: bool = True
    """Start local gear_sonic_deploy in the tmux session. Disable when deploy runs externally."""

    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh. Leave empty for default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # VLA inference options
    policy_host: str = "localhost"
    """Isaac-GR00T PolicyServer host."""

    policy_port: int = 5550
    """Isaac-GR00T PolicyServer port."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    prompt: str = "demo"
    """Language prompt for inference."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy."""

    initial_pose: Literal["calib_full", "standing"] = "calib_full"
    """Initial pose for VLA inference: calib_full or standing."""

    initial_pose_ramp_s: float = 2.0
    """Seconds for standing-to-CALIB_FULL ramp when initial_pose is calib_full."""

    standing_ramp_s: float = 2.0
    """Seconds for CALIB_FULL-to-standing ramp when stopping inference."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state and action bridge
    state_zmq_host: str = ""
    """Robot state ZMQ host. Empty = localhost for local deploy, camera_host for --no-deploy."""

    state_zmq_port: int = 5557
    """Robot state ZMQ port from C++ deploy."""

    action_zmq_host: str = ""
    """Action PUB bind host. Empty = localhost for local deploy, '*' for --no-deploy."""

    action_zmq_port: int = 5556
    """Action PUB bind port consumed by C++ deploy."""

    episode_video_path: str = ""
    """Optional mp4 path to publish as ego_view camera in sim instead of simulator images."""

    # Data exporter (optional recording during inference)
    data_exporter: bool = True
    """Start the data exporter pane for recording during inference."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    task_prompt: str = ""
    """Task prompt for the data exporter. Defaults to the inference prompt if empty."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate."""


SESSION_NAME = "sonic_inference"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _build_sim_command(repo_root: Path, config: InferenceLaunchConfig) -> str:
    image_publish_flag = "" if config.episode_video_path else "--enable-image-publish "
    return (
        f"cd {_quote(repo_root)} && "
        f"source .venv_sim/bin/activate && "
        f"python gear_sonic/scripts/run_sim_loop.py "
        f"{image_publish_flag}--enable-offscreen "
        f"--camera-port {config.camera_port}"
    )


def _build_episode_video_camera_command(repo_root: Path, config: InferenceLaunchConfig) -> str:
    return (
        f"cd {_quote(repo_root)} && "
        f"source .venv_inference/bin/activate && "
        f"PYTHONPATH=. python -m gear_sonic.camera.composed_camera "
        f"--ego-view-camera {_quote(config.episode_video_path)} "
        f"--port {config.camera_port}"
    )


def _should_start_deploy(config: InferenceLaunchConfig) -> bool:
    return config.deploy


def _resolved_state_zmq_host(config: InferenceLaunchConfig) -> str:
    if config.state_zmq_host:
        return config.state_zmq_host
    if _should_start_deploy(config):
        return "localhost"
    return config.camera_host


def _resolved_action_zmq_host(config: InferenceLaunchConfig) -> str:
    if config.action_zmq_host:
        return config.action_zmq_host
    if _should_start_deploy(config):
        return "localhost"
    return "*"


def _build_inference_command(repo_root: Path, config: InferenceLaunchConfig) -> str:
    return (
        f"cd {_quote(repo_root)} && "
        f"source .venv_inference/bin/activate && "
        f"PYTHONPATH=. python gear_sonic/scripts/run_vla_inference.py "
        f"--host {config.policy_host} "
        f"--port {config.policy_port} "
        f"--embodiment-tag {config.embodiment_tag} "
        f"--prompt {_quote(config.prompt)} "
        f"--action-publish-rate {config.action_publish_rate} "
        f"--action-horizon {config.action_horizon} "
        f"--initial-pose {config.initial_pose} "
        f"--initial-pose-ramp-s {config.initial_pose_ramp_s} "
        f"--standing-ramp-s {config.standing_ramp_s} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port} "
        f"--state-zmq-host {_quote(_resolved_state_zmq_host(config))} "
        f"--state-zmq-port {config.state_zmq_port} "
        f"--action-zmq-host {_quote(_resolved_action_zmq_host(config))} "
        f"--action-zmq-port {config.action_zmq_port}"
    )


def _build_deploy_command(repo_root: Path, config: InferenceLaunchConfig) -> str:
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {_quote(repo_root / 'gear_sonic_deploy')} && "
        f"./deploy.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {_quote(config.deploy_checkpoint)} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {_quote(config.deploy_obs_config)} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {_quote(config.deploy_planner)} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {_quote(config.deploy_motion_data)} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    deploy_cmd += deploy_mode
    return deploy_cmd


def _check_prerequisites(config: InferenceLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_inference" / "bin" / "activate").exists():
        errors.append(
            ".venv_inference not found. Run: bash install_scripts/install_inference.sh"
        )

    if _should_start_deploy(config):
        deploy_dir = repo_root / "gear_sonic_deploy"
        if not (deploy_dir / "deploy.sh").exists():
            errors.append(
                f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
                "Ensure the deploy directory is set up."
            )

    if config.data_exporter:
        if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
            errors.append(
                ".venv_data_collection not found (needed for data exporter). Run: "
                "bash install_scripts/install_data_collection.sh"
            )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first."
        )

    if config.episode_video_path:
        if not config.sim:
            errors.append("--episode-video-path is only supported with --sim")
        if not Path(config.episode_video_path).exists():
            errors.append(f"episode video not found: {config.episode_video_path}")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session():
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "inference"],
    )

    # Split into 4 panes: 0|1 / 2|3
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    target = f"{SESSION_NAME}:0.{pane_index}"
    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: InferenceLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(config)
    _kill_existing_session()

    exporter_prompt = config.task_prompt if config.task_prompt else config.prompt

    print("=" * 60)
    print("  SONIC VLA Inference Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  PolicyServer:    {config.policy_host}:{config.policy_port}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Action rate:     {config.action_publish_rate} Hz")
    print(f"  Action horizon:  {config.action_horizon}")
    print(f"  Initial pose:    {config.initial_pose} ({config.initial_pose_ramp_s:.2f}s ramp)")
    print(f"  Standing ramp:   {config.standing_ramp_s:.2f}s")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(
        f"  GEAR-SONIC:      "
        f"{'Local tmux deploy' if _should_start_deploy(config) else 'External deploy (not started)'}"
    )
    print(
        f"  Action ZMQ:      bind tcp://{_resolved_action_zmq_host(config)}:{config.action_zmq_port}"
    )
    print(
        f"  State ZMQ:       subscribe tcp://{_resolved_state_zmq_host(config)}:{config.state_zmq_port}"
    )
    if config.episode_video_path:
        print(f"  Camera override: {config.episode_video_path}")
    print(f"  Data exporter:   {'Yes' if config.data_exporter else 'No'}")
    if config.data_exporter:
        print(f"    DC frequency:  {config.data_exporter_frequency} Hz")
        print(f"    Task prompt:   {exporter_prompt}")
    print(f"  PC IP:           {_get_local_ip()}")
    print("=" * 60)

    _create_tmux_session()
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = _build_sim_command(repo_root, config)
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        if config.episode_video_path:
            subprocess.run(
                ["tmux", "new-window", "-t", SESSION_NAME, "-n", "camera"],
            )
            camera_cmd = _build_episode_video_camera_command(repo_root, config)
            camera_target = f"{SESSION_NAME}:camera"
            subprocess.run(
                ["tmux", "send-keys", "-t", camera_target, camera_cmd, "C-m"],
            )
            print("Starting episode video ego_view camera replay (window: camera)...")
            time.sleep(2.0)

        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )

    # --- Pane 0 (top-left): C++ Deploy / external deploy notice ---
    if _should_start_deploy(config):
        deploy_cmd = _build_deploy_command(repo_root, config)
        print("Starting C++ deploy (pane 0)...")
        _send_to_pane(0, deploy_cmd, wait=3.0)

        if not _check_pane_alive(0):
            print("WARNING: C++ deploy pane may have failed to start.")
    else:
        local_ip = _get_local_ip()
        external_note = textwrap.dedent(
            f"""\
            printf '%s\\n' \\
              'External GEAR-SONIC deploy mode.' \\
              'This tmux session did not start gear_sonic_deploy.' \\
              'On PC2, run deploy.sh with --input-type zmq_manager and --zmq-host {local_ip}.' \\
              'Local VLA action socket: tcp://{_resolved_action_zmq_host(config)}:{config.action_zmq_port}' \\
              'Robot state expected from: tcp://{_resolved_state_zmq_host(config)}:{config.state_zmq_port}'
            """
        ).strip()
        print("Skipping C++ deploy (pane 0); expecting external GEAR-SONIC deploy.")
        _send_to_pane(0, external_note, wait=0.2)

    # --- Pane 2 (bottom-left): Keyboard Publisher ---
    keyboard_script = textwrap.dedent("""\
        import zmq, time
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind('tcp://localhost:5580')
        time.sleep(0.5)
        print('Keyboard publisher ready. Keys: p=pause, k=start/stop, i=init pose, [/]=toggle hands, t=prompt')
        while True:
            key = input()
            if key.startswith('t '):
                pub.send_string('prompt:' + key[2:])
                print('Sent prompt: ' + key[2:])
            else:
                pub.send_string(key)
                print('Sent: ' + key)
    """)
    encoded = base64.b64encode(keyboard_script.encode()).decode()
    keyboard_cmd = (
        f"cd {repo_root} && "
        f"source .venv_inference/bin/activate && "
        f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    )

    print("Starting keyboard publisher (pane 2)...")
    _send_to_pane(1, keyboard_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Data Exporter (optional) ---
    if config.data_exporter:
        exporter_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt '{exporter_prompt}' "
            f"--data-collection-frequency {config.data_exporter_frequency} "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--state-zmq-host {_quote(_resolved_state_zmq_host(config))} "
            f"--state-zmq-port {config.state_zmq_port} "
            f"--sonic-zmq-host localhost "
            f"--sonic-zmq-port {config.action_zmq_port}"
        )
        if config.dataset_name:
            exporter_cmd += f" --dataset-name '{config.dataset_name}'"

        print("Starting data exporter (pane 3)...")
        _send_to_pane(3, exporter_cmd, wait=2.0)

    # --- Pane 1 (top-right): VLA Inference ---
    inference_cmd = _build_inference_command(repo_root, config)

    print("Starting VLA inference (pane 1)...")
    _send_to_pane(2, inference_cmd, wait=1.0)

    # Select the VLA inference pane
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
        if config.episode_video_path:
            print("  Window 'camera':")
            print("    Episode video ego_view replay (.venv_inference)")
            print()
    print("  Window 'inference':")
    if _should_start_deploy(config):
        print("    Pane 0 (top-left):     C++ Deploy")
    else:
        print("    Pane 0 (top-left):     External Deploy Notice")
    print("    Pane 1 (bottom-left):  Keyboard Publisher")
    print("    Pane 2 (top-right):    VLA Inference  <-- you are here")
    if config.data_exporter:
        print("    Pane 3 (bottom-right): Data Exporter")
    print()
    if _should_start_deploy(config):
        print("  ** deploy.sh (pane 0) is waiting for confirmation --")
        print("     click on pane 0 and press Enter to proceed **")
        print()
    else:
        print("  ** gear_sonic_deploy is not started in this tmux session **")
        print("     Start it on PC2 and point --zmq-host to this workstation IP.")
        print()
    print("  Keyboard controls (type in pane 1):")
    print("    p        - Pause / resume inference")
    print("    k        - Start / stop C++ control loop")
    print(f"    i        - Send initial pose ({config.initial_pose})")
    print("    [        - Toggle left hand open/closed (initial pose)")
    print("    ]        - Toggle right hand open/closed (initial pose)")
    print("    t <text> - Change inference prompt")
    if config.data_exporter:
        print("    c        - Start recording episode")
        print("    s        - Stop recording (success)")
        print("    f        - Stop recording (failure)")
    print()
    print("  Navigation:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig, _frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(InferenceLaunchConfig)
    main(config)
