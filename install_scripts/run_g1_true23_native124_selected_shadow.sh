#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

# Read-only PICO -> SOMA -> selected native 124-to-23 policy -> LowState shadow.
# This script never starts a Unitree command publisher.

network="eth0"
frames=100
timeout_seconds=120
check_only=false
continue_rejected_diagnostic=false
experimental_solver_iterations=""

usage() {
  printf '%s\n' \
    "Usage: $0 [--network eth0] [--frames 100] [--timeout-seconds 120] [--experimental-solver-iterations 12|16] [--continue-rejected-diagnostic] [--check-only]"
}

while (($#)); do
  case "$1" in
    --network) network="${2:?--network requires a value}"; shift 2 ;;
    --frames) frames="${2:?--frames requires a value}"; shift 2 ;;
    --timeout-seconds) timeout_seconds="${2:?--timeout-seconds requires a value}"; shift 2 ;;
    --experimental-solver-iterations)
      experimental_solver_iterations="${2:?--experimental-solver-iterations requires a value}"
      shift 2
      ;;
    --continue-rejected-diagnostic)
      continue_rejected_diagnostic=true
      shift
      ;;
    --check-only) check_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$frames" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  printf 'frames and timeout must be positive integers\n' >&2
  exit 2
fi
if [[ -n "$experimental_solver_iterations" &&
      "$experimental_solver_iterations" != "12" &&
      "$experimental_solver_iterations" != "16" ]]; then
  printf 'experimental solver iterations must be 12 or 16\n' >&2
  exit 2
fi
if "$continue_rejected_diagnostic" && ((frames != 100)); then
  printf '%s\n' \
    '--continue-rejected-diagnostic requires exactly --frames 100' >&2
  exit 2
fi
solver_args=()
if [[ -n "$experimental_solver_iterations" ]]; then
  solver_args=(--experimental-solver-iterations "$experimental_solver_iterations")
fi
publisher_diagnostic_args=()
if "$continue_rejected_diagnostic"; then
  publisher_diagnostic_args=(--read-only-diagnostic-100ms-max-age)
fi
shadow_diagnostic_args=()
if "$continue_rejected_diagnostic"; then
  shadow_diagnostic_args=(--continue-rejected-diagnostic)
fi

repo="/mnt/z/codex/GR00T-WholeBodyControl"
binary="$repo/gear_sonic_deploy/target/release/g1_true23_live_shadow"
policy="$repo/artifacts/unitree23_candidates/assets/models/g1/beyondmimic/23dof_50fps/fightAndSports1_subject1.onnx"
publisher="$repo/gear_sonic/scripts/stream_g1_23dof_pico_causal_zmq.py"
worker="$repo/gear_sonic/scripts/stream_g1_23dof_pico_raw_worker.py"
xrt_module_dir="$repo/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
xrt_module="$xrt_module_dir/xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so"
python="/root/.venvs/g1_true23_soma/bin/python"
evidence_root="/root/g1_true23_runs/live"
apk_sha256="e4ac5adb615bf26b49ab4725a9d6b28b1c290682682c010922e2cc66a5f669ca"

declare -A expected_hash=(
  ["$binary"]="20f1b6f0c410327d201270e5a43097f485b237ff239fda52e270ce69f555afbb"
  ["$policy"]="cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9"
  ["$publisher"]="87abe5f2296008300f3d5b82b91aaf655b6002cb4eec947ecb7bb434f7044dc0"
  ["$worker"]="f10a5d2325d100d7b2f548bf41a028b448ca90bf153cf960d1167ca91983fce3"
  ["$xrt_module"]="34eeb4484fb68e860ef4c7a1617022e020084a041b98226aea80f6eb93483de1"
)

for path in "${!expected_hash[@]}"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    printf '[BLOCKED] Missing regular non-symlink input: %s\n' "$path" >&2
    exit 1
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "${expected_hash[$path]}" ]]; then
    printf '[BLOCKED] Frozen bytes changed: %s\n' "$path" >&2
    exit 1
  fi
done
if [[ ! -x "$python" ]]; then
  printf '[BLOCKED] Missing SOMA Python: %s\n' "$python" >&2
  exit 1
fi

"$binary" --validate-only --native124-policy "$policy"
if "$check_only"; then
  printf '[PASS] Selected native23 ONNX and runtime validated; no channels opened.\n'
  exit 0
fi

if pgrep -af '(^|/)(g1_true23_live_shadow|g1_true23_active_gantry|g1_true23_gantry_hold_smoke|stream_g1_23dof_pico_causal_zmq.py|stream_g1_23dof_pico_raw_worker.py)( |$)' >/dev/null; then
  printf '[BLOCKED] Competing robot/PICO process exists.\n' >&2
  exit 1
fi

mapfile -t robotics_service_pids < <(
  pgrep -f '(^|/)RoboticsServiceProcess( |$)' || true
)
if ((${#robotics_service_pids[@]} != 1)); then
  printf '[BLOCKED] Expected one RoboticsServiceProcess; found %s.\n' \
    "${#robotics_service_pids[@]}" >&2
  exit 1
fi
robotics_service_pid="${robotics_service_pids[0]}"
taskset -apc 4,5 "$robotics_service_pid" >/dev/null

mkdir -p "$evidence_root"
run_id="$(date -u +%Y%m%d_%H%M%S)"
shadow_evidence="$evidence_root/native124_selected_shadow_${run_id}.jsonl"
publisher_evidence="$evidence_root/native124_selected_pico_${run_id}.jsonl"
shadow_pid=""
publisher_pid=""
keeper_pid=""
capture_pid=""

stop_group() {
  local pid="$1" signal="$2"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -"$signal" -- "-$pid" 2>/dev/null || true
    timeout 10s tail --pid="$pid" -f /dev/null 2>/dev/null || true
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
}
cleanup() {
  trap - EXIT INT TERM
  stop_group "$publisher_pid" TERM
  stop_group "$shadow_pid" INT
  if [[ -n "$capture_pid" ]]; then
    kill -TERM "$capture_pid" 2>/dev/null || true
    wait "$capture_pid" 2>/dev/null || true
  fi
  if [[ -n "$keeper_pid" ]]; then
    kill -TERM "$keeper_pid" 2>/dev/null || true
    wait "$keeper_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# WSL mirrored networking needs an explicit multicast membership plus a
# read-only capture holder or Unitree DDS packets disappear between probes.
local_ip="$(ip -4 -o addr show dev "$network" | awk '{split($4, a, "/"); print a[1]; exit}')"
if [[ -z "$local_ip" ]]; then
  printf '[BLOCKED] No IPv4 address on robot network: %s\n' "$network" >&2
  exit 1
fi
/usr/bin/python3 - "$local_ip" "$((timeout_seconds + 30))" <<'PY' &
import socket
import sys
import time

local_ip = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", 0))
membership = socket.inet_aton("239.255.0.1") + socket.inet_aton(local_ip)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
print(f"[PASS] Unitree DDS multicast held on {local_ip}", flush=True)
time.sleep(float(sys.argv[2]))
PY
keeper_pid=$!
taskset -c 4,5 tcpdump -ni "$network" -w /dev/null \
  'udp and dst host 239.255.0.1 and dst port 7401' \
  >/dev/null 2>&1 &
capture_pid=$!
sleep 1

# Start costly SOMA/Warp initialization before opening LowState monitor. This
# prevents GPU startup stalls from being mistaken for live robot-state loss.
PYTHONPATH="/root/.cache/g1_true23_soma/source:$repo" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  setsid taskset -c 6-15 \
  timeout --foreground --preserve-status --signal=TERM --kill-after=5s \
  "$((timeout_seconds + 10))s" \
  "$python" "$publisher" \
    --bind tcp://127.0.0.1:5557 \
    --xrt-module-dir "$xrt_module_dir" \
    --packets "$((frames + 25))" \
    --timeout-seconds "$timeout_seconds" \
    --subscriber-warmup-s 1 \
    --evidence "$publisher_evidence" \
    --pico-client-apk-sha256 "$apk_sha256" \
    "${publisher_diagnostic_args[@]}" \
    "${solver_args[@]}" &
publisher_pid=$!

publisher_ready=false
publisher_startup_attempts=$((timeout_seconds * 10))
for ((attempt = 0; attempt < publisher_startup_attempts; attempt++)); do
  if [[ -n "$(ss -ltnH 'sport = :5557')" ]]; then
    publisher_ready=true
    break
  fi
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    wait "$publisher_pid"
  fi
  sleep 0.1
done
if ! "$publisher_ready"; then
  printf '[BLOCKED] PICO publisher did not bind within %s seconds.\n' \
    "$timeout_seconds" >&2
  exit 1
fi

setsid taskset -c 0-3 \
timeout --foreground --preserve-status --signal=INT --kill-after=5s \
  "$((timeout_seconds + 10))s" \
  "$binary" \
    --mode shadow \
    --native124-policy "$policy" \
    --network "$network" \
    --pico-endpoint tcp://127.0.0.1:5557 \
    --frames "$frames" \
    --timeout-seconds "$timeout_seconds" \
    --evidence "$shadow_evidence" \
    "${shadow_diagnostic_args[@]}" &
shadow_pid=$!

shadow_status=0
wait "$shadow_pid" || shadow_status=$?
shadow_pid=""
stop_group "$publisher_pid" TERM
wait "$publisher_pid" 2>/dev/null || true
publisher_pid=""

if "$continue_rejected_diagnostic"; then
  if ((shadow_status != 2)); then
    printf '[FAIL] Rejected-frame diagnostic returned %s, expected 2. Evidence: %s\n' \
      "$shadow_status" "$shadow_evidence" >&2
    exit 1
  fi
  if ! /usr/bin/python3 - "$shadow_evidence" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]
frames = [record for record in records
          if record.get("event") == "native124_action_frame"]
terminal = records[-1] if records else {}
valid = (
    terminal.get("event") == "diagnostic_session_complete"
    and terminal.get("passed") is False
    and terminal.get("diagnostic_only") is True
    and terminal.get("active_runtime_eligible") is False
    and terminal.get("robot_mutation_authorized") is False
    and terminal.get("action_frames") == 100
    and terminal.get("accepted_frames", -1)
        + terminal.get("rejected_frames", -1) == 100
    and terminal.get("rejected_frames", 0) > 0
    and len(frames) == 100
    and [frame.get("action_frame_index") for frame in frames]
        == list(range(100))
    and all(frame.get("diagnostic_only") is True for frame in frames)
    and all(frame.get("active_runtime_eligible") is False for frame in frames)
    and all(frame.get("robot_mutation_authorized") is False for frame in frames)
)
raise SystemExit(0 if valid else 1)
PY
  then
    printf '[FAIL] Rejected-frame diagnostic terminal contract failed. Evidence: %s\n' \
      "$shadow_evidence" >&2
    exit 1
  fi
  printf '[DIAGNOSTIC] Captured 100 frames; passed=false, active-runtime-ineligible, no command APIs.\n'
  printf 'SHADOW_EVIDENCE=%s\nPICO_EVIDENCE=%s\n' \
    "$shadow_evidence" "$publisher_evidence"
  exit 0
fi

if ((shadow_status != 0)) ||
   ! grep -Fq '"event":"session_complete"' "$shadow_evidence" ||
   ! grep -Fq '"passed":true' "$shadow_evidence"; then
  printf '[FAIL] Native23 shadow failed. Evidence: %s\n' "$shadow_evidence" >&2
  exit 1
fi

printf '[PASS] Native23 PICO shadow complete; robot command APIs absent.\n'
printf 'SHADOW_EVIDENCE=%s\nPICO_EVIDENCE=%s\n' \
  "$shadow_evidence" "$publisher_evidence"
