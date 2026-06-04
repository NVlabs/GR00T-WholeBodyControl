#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[Real Robot Preflight]${NC}"

matches="$(pgrep -af 'run_sim_loop.py|mujoco|mujoco.viewer|simulate' || true)"
if [[ -n "$matches" ]]; then
    echo -e "${RED}Refusing real deploy: MuJoCo/simulation process appears to be running.${NC}"
    echo "$matches"
    echo "Stop every matched process before launching real robot deploy."
    exit 1
fi

confirm_item() {
    local prompt="$1"
    local answer
    read -r -p "$prompt [y/N]: " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Preflight item not confirmed; aborting real deploy.${NC}"
        exit 1
    fi
}

echo ""
echo "Confirm before real low-level GR00T deploy:"
echo ""

confirm_item "Robot is on a safety harness/protective frame"
confirm_item "A 3 m clear zone is established and the designated spotter is assigned"
confirm_item "Safety operator is at the deploy keyboard and has drilled keyboard O three times"
confirm_item "VR operator has drilled PICO A+B+X+Y three times"
confirm_item "Unitree remote L1+A / L2+B is NOT treated as an E-stop during lowcmd deploy"
confirm_item "Floor is flat, dry, unobstructed, and robot remains in line of sight"
confirm_item "Robot, headset, and controller batteries are charged"
confirm_item "Operator clothing is tight-fitting enough for VR tracking"
confirm_item "Network latency is checked and is not above 30 ms"
confirm_item "PICO controller-only mode risk is understood because ankle trackers are absent"
confirm_item "Operator is ready to match the robot zero/CALIB_FULL pose before teleop"
confirm_item "Pose/wrist mismatch policy is clear: stop and recalibrate in sim before retrying"

echo -e "${GREEN}Real-robot preflight confirmed.${NC}"
