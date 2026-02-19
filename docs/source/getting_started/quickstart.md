# Quick Start

Get started with SONIC in minutes!

 First, ensure you've completed the [installation guide](installation_deploy).

```{admonition} Safety Warning
:class: danger
Robots can be dangerous. Ensure a clear safety zone, keep a safety operator ready to trigger an emergency stop in front of the keyboard, and use this software at your own risk. The authors and contributors are not responsible for any damage, injury, or loss caused by use or misuse of this project.
```

## IsaacLab Eval 

Coming soon! 


## Sim2Sim in MuJoCo

<video width="100%" autoplay loop muted playsinline style="border-radius: 8px; margin: 1em 0;">
  <source src="../_static/sim2sim.mp4" type="video/mp4">
</video>

For testing in a MuJoCo simulator, run the simulation loop and deployment script in separate terminals.

```{note}
The MuJoCo simulator (Terminal 1) runs on the **host** in a Python virtual environment — it is **not** inside the Docker container. The deployment binary (Terminal 2) can run either natively on the host or inside the Docker container. If you are using Docker, run Terminal 1 on the host and Terminal 2 inside the container.
```

### One-time setup: install the MuJoCo sim environment

On the **host** (outside Docker), from the **repo root** (`GR00T-WholeBodyControl/`), run:

```sh
bash install_scripts/install_mujoco_sim.sh
```

This creates a lightweight `.venv_sim` virtual environment with only the packages needed for the simulator (MuJoCo, Pinocchio, Unitree SDK2, etc.).

### Running the sim2sim loop

We highly recommend running through this process and getting familiar with the controls in simulation before deploying on real hardware.

**Terminal 1 — MuJoCo simulator** (host, from repo root):

```sh
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 — Deployment** (host or Docker, from `gear_sonic_deploy/`):

```sh
bash deploy.sh sim
```

**Starting Control:**

1. In Terminal 2 (deploy.sh), press **`]`** to start the policy.
2. Click on the MuJoCo viewer window, press **`9`** to drop the robot to the ground.
3. Go back to Terminal 2. Press **`T`** to play the current reference motion — the robot will execute it to completion.
4. Press **`N`** or **`P`** to switch to the next or previous motion sequence.
5. Press **`T`** again to play the new motion.
6. You can press **`T`** again to replay the same motion once it has finished. If you want to stop and go back to the first frame of the current motion, press **`R`** to restart it from the beginning. This can be used to stop the motion without terminating the policy.
7. When you are done or need an **emergency stop**, press **`O`** to stop control and exit.

For more controls, see the tutorials for [Keyboard](../tutorials/keyboard.md), [Gamepad](../tutorials/gamepad.md), [ZMQ Streaming](../tutorials/zmq.md), and [Interface Manager](../tutorials/manager.md).

## Real Robot

To deploy on the real G1 robot, run:

```sh
./deploy.sh real
```

## Online Visualization

Start the visualizer and connect to a running `g1_deploy` executable:

```sh
python visualize_motion.py --realtime_debug_url tcp://localhost:5557
```

Notes:
- Default port: 5557 (change with `--zmq-out-port <port>`)
- Default topic: `g1_debug` (change with `--zmq-out-topic <topic>` on executable, `--realtime_debug_topic <topic>` on visualizer)
- For physical robots, replace `localhost` with the robot's IP address

For offline motion CSV visualization and logging details, see [Deployment Code & Program Flow](../references/deployment_code.md).

For more advanced usage, see the tutorials for [Keyboard](../tutorials/keyboard.md), [Gamepad](../tutorials/gamepad.md), [ZMQ Streaming](../tutorials/zmq.md), [VR Whole-Body Teleop](../tutorials/vr_wholebody_teleop.md), and [Interface Manager](../tutorials/manager.md).
