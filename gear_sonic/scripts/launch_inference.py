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
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import signal
import socket
import base64
import subprocess
import sys
import textwrap
import time


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

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

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
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
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
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

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
            f"--camera-port {config.camera_port}"
        )
        if config.dataset_name:
            exporter_cmd += f" --dataset-name '{config.dataset_name}'"

        print("Starting data exporter (pane 3)...")
        _send_to_pane(3, exporter_cmd, wait=2.0)

    # --- Pane 1 (top-right): VLA Inference ---
    inference_cmd = (
        f"cd {repo_root} && "
        f"source .venv_inference/bin/activate && "
        f"python gear_sonic/scripts/run_vla_inference.py "
        f"--host {config.policy_host} "
        f"--port {config.policy_port} "
        f"--embodiment-tag {config.embodiment_tag} "
        f"--prompt '{config.prompt}' "
        f"--action-publish-rate {config.action_publish_rate} "
        f"--action-horizon {config.action_horizon} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port}"
    )

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
    print("  Window 'inference':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Keyboard Publisher")
    print("    Pane 2 (top-right):    VLA Inference  <-- you are here")
    if config.data_exporter:
        print("    Pane 3 (bottom-right): Data Exporter")
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Keyboard controls (type in pane 1):")
    print("    p        - Pause / resume inference")
    print("    k        - Start / stop C++ control loop")
    print("    i        - Send initial pose")
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
