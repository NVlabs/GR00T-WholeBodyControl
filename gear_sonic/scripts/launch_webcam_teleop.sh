#!/bin/bash
# Launch the full webcam -> GEM -> SONIC teleoperation stack in a tmux session.
#
# Panes:
#   0 (top-left):     MuJoCo sim loop            (.venv_sim)          [sim mode only]
#   1 (top-right):    C++ SONIC controller       (deploy.sh)
#   2 (bottom-left):  GEM->SONIC bridge          (.venv_teleop)       <- keyboard here!
#   3 (bottom-right): GEM webcam pose estimation (GENMO/.venv, GPU)
#
# Usage (from repo root):
#   bash gear_sonic/scripts/launch_webcam_teleop.sh sim      # simulation (default)
#   bash gear_sonic/scripts/launch_webcam_teleop.sh real     # real G1 robot
#
# Controls once running:
#   - Bridge pane (bottom-left):  s=start imitation, p=pause/resume, o/q=stop
#   - C++ pane:                   O=emergency stop
#   - GEM pane:                   q on the OpenCV window closes the demo
#
# Extra args are forwarded to the GEM publisher, e.g.:
#   bash gear_sonic/scripts/launch_webcam_teleop.sh sim --camera_id 2

set -e

MODE="${1:-sim}"
shift || true
GEM_EXTRA_ARGS="$*"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="webcam_teleop"

if [ "$MODE" != "sim" ] && [ "$MODE" != "real" ]; then
    echo "Usage: $0 [sim|real] [extra GEM args]"
    exit 1
fi

# --- Sanity checks ---
err=0
[ -d "$REPO_ROOT/.venv_teleop" ] || { echo "MISSING: .venv_teleop (run: bash install_scripts/install_pico.sh)"; err=1; }
[ -d "$REPO_ROOT/external_dependencies/GENMO/.venv" ] || { echo "MISSING: GENMO venv (run: bash install_scripts/install_gem_webcam.sh)"; err=1; }
[ -f "$REPO_ROOT/external_dependencies/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz" ] || \
    echo "WARNING: SMPLX_NEUTRAL.npz not found (only needed for --render); see install_gem_webcam.sh output."
if [ "$MODE" = "sim" ]; then
    [ -d "$REPO_ROOT/.venv_sim" ] || { echo "MISSING: .venv_sim (run: bash install_scripts/install_mujoco_sim.sh)"; err=1; }
fi
[ -f "$REPO_ROOT/gear_sonic_deploy/deploy.sh" ] || { echo "MISSING: gear_sonic_deploy/deploy.sh"; err=1; }
[ $err -ne 0 ] && exit 1

command -v tmux &> /dev/null || { echo "tmux not installed (sudo apt install tmux)"; exit 1; }

# Kill stale session
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT"

# Pane 0: sim loop (or placeholder in real mode)
if [ "$MODE" = "sim" ]; then
    tmux send-keys -t "$SESSION" \
        "source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py" C-m
else
    tmux send-keys -t "$SESSION" \
        "echo '[real mode] no sim needed — make sure the G1 is up (192.168.123.x)'" C-m
fi

# Pane 1: C++ SONIC controller
tmux split-window -h -t "$SESSION" -c "$REPO_ROOT/gear_sonic_deploy"
tmux send-keys -t "$SESSION" \
    "sleep 3 && ./deploy.sh --input-type zmq_manager $MODE" C-m

# Pane 2: bridge (keyboard control lives here)
tmux split-window -v -t "$SESSION:0.0" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION" \
    "source .venv_teleop/bin/activate && python gear_sonic/scripts/webcam_smpl_streamer.py" C-m

# Pane 3: GEM webcam publisher (GPU)
tmux split-window -v -t "$SESSION:0.1" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION" \
    "source external_dependencies/GENMO/.venv/bin/activate && python gear_sonic/scripts/gem_webcam_zmq_publisher.py --no_imgfeat --render --render_mode opencv $GEM_EXTRA_ARGS" C-m

tmux select-pane -t "$SESSION:0.2"   # focus the bridge pane (keyboard control)

echo ""
echo "tmux session '$SESSION' started (mode: $MODE)."
echo "Attach with:   tmux attach -t $SESSION"
echo ""
echo "Then: wait for GEM warmup (~120 frames), stand in front of the camera,"
echo "and press 's' in the bridge pane (bottom-left) to start imitation."
tmux attach -t "$SESSION"
