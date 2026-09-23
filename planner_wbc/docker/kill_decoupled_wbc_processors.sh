#!/bin/bash

# kill_decoupled_wbc_processors.sh
# Kill decoupled_wbc processes in current container to prevent message passing conflicts

# Note: Don't use 'set -e' as tmux/pgrep commands may return non-zero exit codes

# Configuration
DRY_RUN=false
FORCE=false
QUIET=false
declare -A FOUND_PROCESSES

# Default to verbose mode if no arguments
[[ $# -eq 0 ]] && { QUIET=false; DRY_RUN=false; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --force) FORCE=true ;;
        --verbose|-v) VERBOSE=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--force] [--verbose] [--help]"
            echo "Kill decoupled_wbc processes to prevent message passing conflicts"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Colors (only if not quiet)
if [[ "$QUIET" != true ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

# Show processes by pattern (for preview)
show_processes_by_pattern() {
    local pattern="$1" desc="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    [[ -z "$pids" ]] && return 0
    
    echo -e "${YELLOW}$desc processes:${NC}"
    
    for pid in $pids; do
        local cmd=$(ps -p $pid -o cmd= 2>/dev/null || echo "Process not found")
        echo "  PID $pid: $cmd"
    done
}

# Kill processes by pattern (silent killing)
kill_by_pattern() {
    local pattern="$1" desc="$2" signal="${3:-TERM}"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    [[ -z "$pids" ]] && return 0
    
    for pid in $pids; do
        # Kill if not dry run
        [[ "$DRY_RUN" != true ]] && kill -$signal $pid 2>/dev/null
    done
}

# Show tmux sessions (for preview)
show_tmux() {
    local pattern="$1"
    local sessions=$(tmux list-sessions 2>/dev/null | grep "$pattern" | cut -d: -f1 || true)
    
    [[ -z "$sessions" ]] && return 0
    
    echo -e "${YELLOW}Tmux sessions:${NC}"
    
    for session in $sessions; do
        echo "  Session: $session"
    done
}

# Kill tmux sessions (silent killing)
kill_tmux() {
    local pattern="$1"
    local sessions=$(tmux list-sessions 2>/dev/null | grep "$pattern" | cut -d: -f1 || true)
    
    [[ -z "$sessions" ]] && return 0
    
    for session in $sessions; do
        [[ "$DRY_RUN" != true ]] && tmux kill-session -t "$session" 2>/dev/null
    done
}

# Show processes by port (for preview)
show_processes_by_port() {
    local port="$1" desc="$2"
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    [[ -z "$pids" ]] && return 0
    
    echo -e "${YELLOW}$desc (port $port):${NC}"
    
    for pid in $pids; do
        local cmd=$(ps -p $pid -o cmd= 2>/dev/null || echo "Process not found")
        echo "  PID $pid: $cmd"
    done
}

# Kill processes by port (silent killing)
kill_by_port() {
    local port="$1" desc="$2"
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    [[ -z "$pids" ]] && return 0
    
    for pid in $pids; do
        [[ "$DRY_RUN" != true ]] && kill -TERM $pid 2>/dev/null
    done
}

# Check if any processes exist
has_processes() {
    # Check for processes
    local has_tmux=$(tmux list-sessions 2>/dev/null | grep "g1_deployment" || true)
    local has_control=$(pgrep -f "run_g1_control_loop.py" 2>/dev/null || true)
    local has_teleop=$(pgrep -f "run_teleop_policy_loop.py" 2>/dev/null || true)
    local has_camera=$(pgrep -f "camera_forwarder.py" 2>/dev/null || true)
    local has_rqt=$(pgrep -f "rqt.*image_view" 2>/dev/null || true)
    local has_port=$(lsof -ti:5555 2>/dev/null || true)
    
    [[ -n "$has_tmux" || -n "$has_control" || -n "$has_teleop" || -n "$has_camera" || -n "$has_rqt" || -n "$has_port" ]]
}

# Main execution
main() {
    # Check if any processes exist first
    if ! has_processes; then
        # No processes to kill, exit silently
        exit 0
    fi
    
    # Show header and processes to be killed
    if [[ "$QUIET" != true ]]; then
        echo -e "${BLUE}=== decoupled_wbc Process Killer ===${NC}"
        [[ "$DRY_RUN" == true ]] && echo -e "${BLUE}=== DRY RUN MODE ===${NC}"
        
        # Show what will be killed
        show_tmux "g1_deployment"
        show_processes_by_pattern "run_g1_control_loop.py" "G1 control loop"
        show_processes_by_pattern "run_teleop_policy_loop.py" "Teleop policy"
        show_processes_by_pattern "camera_forwarder.py" "Camera forwarder"
        show_processes_by_pattern "rqt.*image_view" "RQT viewer"
        show_processes_by_port "5555" "Inference server"
        
        # Ask for confirmation
        if [[ "$FORCE" != true && "$DRY_RUN" != true ]]; then
            echo
            echo -e "${RED}WARNING: This will terminate the above decoupled_wbc processes!${NC}"
            read -p "Continue? [Y/n]: " -n 1 -r
            echo
            # Default to Y - only abort if user explicitly types 'n' or 'N'
            [[ $REPLY =~ ^[Nn]$ ]] && { echo "Aborted."; exit 0; }
        fi
        echo
    fi
    
    # Kill processes (silently)
    kill_tmux "g1_deployment"
    kill_by_pattern "run_g1_control_loop.py" "G1 control loop"
    kill_by_pattern "run_teleop_policy_loop.py" "Teleop policy"
    kill_by_pattern "camera_forwarder.py" "Camera forwarder"
    kill_by_pattern "rqt.*image_view" "RQT viewer"
    kill_by_port "5555" "Inference server"
    
    # Force kill remaining (SIGKILL)
    [[ "$DRY_RUN" != true ]] && {
        sleep 1
        kill_by_pattern "run_g1_control_loop.py" "G1 control loop" "KILL"
        kill_by_pattern "run_teleop_policy_loop.py" "Teleop policy" "KILL"
        kill_by_pattern "camera_forwarder.py" "Camera forwarder" "KILL"
    }
    
    # Summary (unless quiet)
    [[ "$QUIET" != true ]] && {
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "${BLUE}=== DRY RUN COMPLETE ===${NC}"
        else
            echo -e "${GREEN}All decoupled_wbc processes terminated${NC}"
        fi
    }
}

main "$@"
