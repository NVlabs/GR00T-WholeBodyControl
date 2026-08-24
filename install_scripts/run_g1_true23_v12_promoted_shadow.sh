#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

# One-shot, read-only V12 integration run. This script has no Unitree command
# publisher and never invokes the active gantry executable.

network="eth0"
frames=100
timeout_seconds=120
check_only=false

# Reviewed workstation isolation contract. Keep the DDS/ONNX reader on two
# physical cores, reserve one physical core for WSL/kernel/XRT service, and keep
# the causal publisher and its inherited capture worker on five disjoint cores.
# Do not use SCHED_RR here: whole-process RT scheduling lets solver/native
# helper threads preempt the normal-priority LowState delivery path.
control_cpu_set="0-3"
service_cpu_set="4,5"
publisher_cpu_set="6-15"
expected_online_cpus="0-15"
robotics_service_pid=""
robotics_service_binary="/opt/apps/roboticsservice/RoboticsServiceProcess"
expected_robotics_service_sha256="8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc"

usage() {
  printf '%s\n' \
    "Usage: $0 [--network eth0] [--frames 100] [--timeout-seconds 120] [--check-only]"
}

while (($#)); do
  case "$1" in
    --network)
      network="${2:?--network requires a value}"
      shift 2
      ;;
    --frames)
      frames="${2:?--frames requires a value}"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="${2:?--timeout-seconds requires a value}"
      shift 2
      ;;
    --check-only)
      check_only=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$frames" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  printf 'frames and timeout must be positive integers\n' >&2
  exit 2
fi

repo="/mnt/z/codex/GR00T-WholeBodyControl"
binary="$repo/gear_sonic_deploy/target/release/g1_true23_live_shadow"
expected_binary_sha256="5afbb65dbce81520441b79b384afa0ccfeb50f6236c1b425a22142672918bc67"
publisher="$repo/gear_sonic/scripts/stream_g1_23dof_pico_causal_zmq.py"
expected_publisher_sha256="fd387e3f3af9893167a15a23b1c8a67ed80f812a968983b1755f1a594274e163"
worker="$repo/gear_sonic/scripts/stream_g1_23dof_pico_raw_worker.py"
expected_worker_sha256="f10a5d2325d100d7b2f548bf41a028b448ca90bf153cf960d1167ca91983fce3"
validator="$repo/gear_sonic/scripts/validate_g1_true23_stage1_evidence.py"
expected_validator_sha256="2a6a68266481dc296d80beee2e5e4196c5ab430346e9055a112de12aad422f48"
xrt_module_dir="$repo/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
xrt_module="$xrt_module_dir/xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so"
expected_xrt_module_sha256="34eeb4484fb68e860ef4c7a1617022e020084a041b98226aea80f6eb93483de1"
python="/root/.venvs/g1_true23_soma/bin/python"
encoder="/root/g1_true23_eval/v12/causal_model_50_diagnostic.encoder.onnx"
decoder="/root/g1_true23_eval/v12/causal_model_50_diagnostic.decoder.onnx"
metadata="/root/g1_true23_eval/v12/causal_model_50_diagnostic.diagnostic.json"
promotion="/root/g1_true23_eval/v12/causal_model_50_causal_mujoco_deployment_bytes_promotion.json"
evidence_root="/root/g1_true23_runs/live"
apk_sha256="e4ac5adb615bf26b49ab4725a9d6b28b1c290682682c010922e2cc66a5f669ca"

for required in \
  "$binary" "$publisher" "$worker" "$validator" "$xrt_module" \
  "$encoder" "$decoder" "$metadata" "$promotion"; do
  if [[ ! -f "$required" || -L "$required" ]]; then
    printf 'Required input must be a regular non-symlink file: %s\n' "$required" >&2
    exit 1
  fi
done
python_real="$(readlink -f -- "$python")"
if [[ ! -x "$python" || ! -f "$python_real" || -L "$python_real" ]]; then
  printf 'Python must resolve to an executable regular file: %s\n' "$python" >&2
  exit 1
fi
if [[ ! -d "$xrt_module_dir" ]]; then
  printf 'Missing required XRT module directory: %s\n' "$xrt_module_dir" >&2
  exit 1
fi
for hash_binding in \
  "shadow binary|$binary|$expected_binary_sha256" \
  "publisher|$publisher|$expected_publisher_sha256" \
  "capture worker|$worker|$expected_worker_sha256" \
  "XRT module|$xrt_module|$expected_xrt_module_sha256" \
  "evidence validator|$validator|$expected_validator_sha256"; do
  IFS='|' read -r hash_role hash_path expected_hash <<<"$hash_binding"
  actual_hash="$(sha256sum "$hash_path" | awk '{print $1}')"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    printf '[BLOCKED] Frozen %s bytes changed: expected %s, got %s\n' \
      "$hash_role" "$expected_hash" "$actual_hash" >&2
    exit 1
  fi
done

validate_cpu_isolation() {
  local actual_siblings affinity affinity_line cls cpu expected_siblings
  local online_cpus scheduler_class service_exe service_sha256 tid
  local -a service_pids=()
  for required_command in taskset setsid ps pgrep; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
      printf '[BLOCKED] %s is required for owned scheduling isolation.\n' \
        "$required_command" >&2
      return 1
    fi
  done
  scheduler_class="$(ps -o cls= -p "$$" | tr -d '[:space:]')"
  if [[ "$scheduler_class" != "TS" ]]; then
    printf '[BLOCKED] Wrapper must run under SCHED_OTHER/TS; found %s.\n' \
      "$scheduler_class" >&2
    return 1
  fi
  if [[ ! -r /sys/devices/system/cpu/online ]]; then
    printf '[BLOCKED] Cannot verify the reviewed CPU topology.\n' >&2
    return 1
  fi
  online_cpus="$(< /sys/devices/system/cpu/online)"
  if [[ "$online_cpus" != "$expected_online_cpus" ]]; then
    printf '[BLOCKED] Reviewed CPU topology requires %s online; found %s.\n' \
      "$expected_online_cpus" "$online_cpus" >&2
    return 1
  fi
  for cpu in {0..15}; do
    expected_siblings="$((cpu / 2 * 2))-$((cpu / 2 * 2 + 1))"
    if [[ ! -r "/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list" ]]; then
      printf '[BLOCKED] Cannot verify SMT siblings for CPU %s.\n' "$cpu" >&2
      return 1
    fi
    actual_siblings="$(< "/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list")"
    if [[ "$actual_siblings" != "$expected_siblings" ]]; then
      printf '[BLOCKED] CPU %s sibling contract requires %s; found %s.\n' \
        "$cpu" "$expected_siblings" "$actual_siblings" >&2
      return 1
    fi
  done
  if ! taskset -c "$control_cpu_set" true ||
     ! taskset -c "$service_cpu_set" true ||
     ! taskset -c "$publisher_cpu_set" true; then
    printf '[BLOCKED] Reviewed control/service/publisher CPU sets are unavailable.\n' >&2
    return 1
  fi
  mapfile -t service_pids < <(
    pgrep -f '(^|/)RoboticsServiceProcess( |$)' || true
  )
  if ((${#service_pids[@]} != 1)); then
    printf '[BLOCKED] Expected exactly one RoboticsServiceProcess; found %s.\n' \
      "${#service_pids[@]}" >&2
    return 1
  fi
  robotics_service_pid="${service_pids[0]}"
  service_exe="$(readlink -f -- "/proc/$robotics_service_pid/exe")"
  if [[ "$service_exe" != "$robotics_service_binary" ]]; then
    printf '[BLOCKED] RoboticsService executable is %s; expected %s.\n' \
      "$service_exe" "$robotics_service_binary" >&2
    return 1
  fi
  service_sha256="$(sha256sum "$service_exe" | awk '{print $1}')"
  if [[ "$service_sha256" != "$expected_robotics_service_sha256" ]]; then
    printf '[BLOCKED] RoboticsService hash changed: expected %s, found %s.\n' \
      "$expected_robotics_service_sha256" "$service_sha256" >&2
    return 1
  fi
  while read -r tid cls; do
    [[ -n "$tid" ]] || continue
    affinity_line="$(taskset -pc "$tid" 2>&1)" || return 1
    affinity="${affinity_line##*: }"
    if [[ "$cls" != "TS" || "$affinity" != "$service_cpu_set" ]]; then
      printf '[BLOCKED] RoboticsService TID %s is %s/%s; expected TS/%s.\n' \
        "$tid" "$cls" "$affinity" "$service_cpu_set" >&2
      return 1
    fi
  done < <(ps -Lo tid=,cls= -p "$robotics_service_pid")
}
validate_cpu_isolation

"$binary" \
  --validate-only \
  --encoder "$encoder" \
  --decoder "$decoder" \
  --metadata "$metadata" \
  --promotion "$promotion"

if "$check_only"; then
  printf '[PASS] Check-only complete; CPU isolation verified; no robot or PICO channels opened.\n'
  exit 0
fi

competing_processes="$(
  pgrep -af \
    '(^|/)(g1_deploy_onnx_ref|g1_true23_gantry_hold_smoke|g1_true23_active_gantry|g1_true23_live_shadow|stream_g1_23dof_pico_causal_zmq.py|stream_g1_23dof_pico_raw_worker.py)( |$)' \
    || true
)"
if [[ -n "$competing_processes" ]]; then
  printf '[BLOCKED] Competing robot/controller or PICO process detected:\n%s\n' \
    "$competing_processes" >&2
  exit 1
fi

mkdir -p "$evidence_root"
run_id="$(date +%Y%m%d_%H%M%S)"
shadow_evidence="$evidence_root/v12_promoted_shadow_${run_id}.jsonl"
publisher_evidence="$evidence_root/v12_pico_causal_publisher_${run_id}.jsonl"
scheduler_log="$evidence_root/v12_promoted_scheduler_${run_id}.log"
if [[ -e "$shadow_evidence" || -e "$publisher_evidence" ||
      -e "$scheduler_log" ]]; then
  printf 'Evidence destination collision\n' >&2
  exit 1
fi

shadow_pid=""
shadow_pgid=""
publisher_pid=""
publisher_pgid=""
owned_group_pgid() {
  local actual_pgid="" leader_pid="$1" shell_pgid
  shell_pgid="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"
  for _ in {1..50}; do
    actual_pgid="$(ps -o pgid= -p "$leader_pid" 2>/dev/null | tr -d '[:space:]')"
    if [[ "$actual_pgid" == "$leader_pid" ]]; then
      break
    fi
    kill -0 "$leader_pid" 2>/dev/null || return 1
    sleep 0.02
  done
  if [[ ! "$actual_pgid" =~ ^[1-9][0-9]*$ ]] ||
     [[ "$actual_pgid" != "$leader_pid" ]] ||
     [[ "$actual_pgid" == "$shell_pgid" ]]; then
    printf '[BLOCKED] Refusing unowned process group for PID %s (PGID %s).\n' \
      "$leader_pid" "${actual_pgid:-missing}" >&2
    return 1
  fi
  printf '%s' "$actual_pgid"
}
owned_group_alive() {
  [[ -n "$1" ]] && pgrep -g "$1" >/dev/null 2>&1
}
wait_owned_group_clean() {
  local child_status=0 group_pid="$1" leader_pid="$2"
  if wait "$leader_pid"; then
    child_status=0
  else
    child_status=$?
  fi
  for _ in {1..150}; do
    owned_group_alive "$group_pid" || break
    sleep 0.1
  done
  if owned_group_alive "$group_pid"; then
    kill -KILL -- "-$group_pid" 2>/dev/null || true
    for _ in {1..50}; do
      owned_group_alive "$group_pid" || break
      sleep 0.1
    done
    printf '[BLOCKED] Descendant survived leader exit in owned PGID %s.\n' \
      "$group_pid" >&2
    return 125
  fi
  return "$child_status"
}
stop_owned_group_bounded() {
  local child_status=0 forced=false group_pid="$1" leader_pid="$2"
  local stop_signal="$3"
  if kill -0 "$leader_pid" 2>/dev/null; then
    kill -"$stop_signal" "$leader_pid" 2>/dev/null || true
    if ! timeout 20s tail --pid="$leader_pid" -f /dev/null; then
      forced=true
      kill -KILL -- "-$group_pid" 2>/dev/null || true
      timeout 5s tail --pid="$leader_pid" -f /dev/null || true
    fi
  fi
  if wait "$leader_pid" 2>/dev/null; then
    child_status=0
  else
    child_status=$?
  fi
  for _ in {1..150}; do
    owned_group_alive "$group_pid" || break
    sleep 0.1
  done
  if owned_group_alive "$group_pid"; then
    forced=true
    kill -KILL -- "-$group_pid" 2>/dev/null || true
    for _ in {1..50}; do
      owned_group_alive "$group_pid" || break
      sleep 0.1
    done
  fi
  if owned_group_alive "$group_pid"; then
    printf '[BLOCKED] Owned PGID %s survived bounded KILL.\n' "$group_pid" >&2
    return 125
  fi
  if "$forced"; then
    return 124
  fi
  return "$child_status"
}
abort_unregistered_launch() {
  local actual_pgid leader_pid="$1" shell_pgid stop_signal="$2"
  actual_pgid="$(ps -o pgid= -p "$leader_pid" 2>/dev/null | tr -d '[:space:]')"
  shell_pgid="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"
  if [[ "$actual_pgid" =~ ^[1-9][0-9]*$ ]] &&
     [[ "$actual_pgid" == "$leader_pid" ]] &&
     [[ "$actual_pgid" != "$shell_pgid" ]]; then
    stop_owned_group_bounded "$actual_pgid" "$leader_pid" "$stop_signal" || true
  else
    kill -"$stop_signal" "$leader_pid" 2>/dev/null || true
    timeout 20s tail --pid="$leader_pid" -f /dev/null || true
    kill -KILL "$leader_pid" 2>/dev/null || true
    wait "$leader_pid" 2>/dev/null || true
  fi
}
verify_and_log_group_scheduler() {
  local affinity affinity_line cls comm expected_cpu_set="$3" found=false
  local group_pid="$2" pid psr role="$1" rtprio tid
  while read -r pid tid cls rtprio psr comm; do
    [[ -n "$tid" ]] || continue
    found=true
    affinity_line="$(taskset -pc "$tid" 2>&1)" || return 1
    affinity="${affinity_line##*: }"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$role" "$pid" "$tid" "$group_pid" "$cls" "$rtprio" "$psr" \
      "$affinity" "$comm" >>"$scheduler_log"
    if [[ "$cls" != "TS" || "$affinity" != "$expected_cpu_set" ]]; then
      printf '[BLOCKED] %s TID %s scheduler/affinity is %s/%s, expected TS/%s.\n' \
        "$role" "$tid" "$cls" "$affinity" "$expected_cpu_set" >&2
      return 1
    fi
  done < <(
    ps -eLo pid=,tid=,pgid=,cls=,rtprio=,psr=,comm= |
      awk -v target="$group_pid" '$3 == target {print $1, $2, $4, $5, $6, $7}'
  )
  "$found"
}
verify_and_log_process_scheduler() {
  local affinity affinity_line cls comm expected_cpu_set="$3" found=false
  local group_pid pid="$2" psr role="$1" rtprio tid
  while read -r tid group_pid cls rtprio psr comm; do
    [[ -n "$tid" ]] || continue
    found=true
    affinity_line="$(taskset -pc "$tid" 2>&1)" || return 1
    affinity="${affinity_line##*: }"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$role" "$pid" "$tid" "$group_pid" "$cls" "$rtprio" "$psr" \
      "$affinity" "$comm" >>"$scheduler_log"
    if [[ "$cls" != "TS" || "$affinity" != "$expected_cpu_set" ]]; then
      printf '[BLOCKED] %s TID %s scheduler/affinity is %s/%s, expected TS/%s.\n' \
        "$role" "$tid" "$cls" "$affinity" "$expected_cpu_set" >&2
      return 1
    fi
  done < <(ps -Lo tid=,pgid=,cls=,rtprio=,psr=,comm= -p "$pid")
  "$found"
}
assert_no_runtime_tree() {
  local survivors
  survivors="$(
    pgrep -af \
      '(^|/)(g1_deploy_onnx_ref|g1_true23_gantry_hold_smoke|g1_true23_active_gantry|g1_true23_live_shadow|stream_g1_23dof_pico_causal_zmq.py|stream_g1_23dof_pico_raw_worker.py)( |$)' \
      || true
  )"
  if [[ -n "$survivors" ]]; then
    printf '[BLOCKED] Controller/publisher process survived cleanup:\n%s\n' \
      "$survivors" >&2
    return 1
  fi
}
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$publisher_pid" ]]; then
    stop_owned_group_bounded "$publisher_pgid" "$publisher_pid" TERM || true
    publisher_pid=""
    publisher_pgid=""
  fi
  if [[ -n "$shadow_pid" ]]; then
    stop_owned_group_bounded "$shadow_pgid" "$shadow_pid" INT || true
    shadow_pid=""
    shadow_pgid=""
  fi
  if ! assert_no_runtime_tree; then
    exit 125
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

setsid taskset -c "$control_cpu_set" \
  timeout --foreground --preserve-status --signal=INT --kill-after=5s \
  "$((timeout_seconds + 10))s" "$binary" \
  --mode shadow \
  --network "$network" \
  --pico-endpoint tcp://127.0.0.1:5557 \
  --frames "$frames" \
  --timeout-seconds "$timeout_seconds" \
  --evidence "$shadow_evidence" \
  --encoder "$encoder" \
  --decoder "$decoder" \
  --metadata "$metadata" \
  --promotion "$promotion" &
shadow_pid=$!
if ! shadow_pgid="$(owned_group_pgid "$shadow_pid")"; then
  abort_unregistered_launch "$shadow_pid" INT
  shadow_pid=""
  exit 1
fi

# Let the read-only subscriber finish artifact validation and bind first.
sleep 2
PYTHONPATH="/root/.cache/g1_true23_soma/source:$repo" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  setsid taskset -c "$publisher_cpu_set" \
  timeout --foreground --preserve-status --signal=TERM --kill-after=5s \
  "$((timeout_seconds + 10))s" \
  "$python" "$publisher" \
    --bind tcp://127.0.0.1:5557 \
    --xrt-module-dir "$xrt_module_dir" \
    --packets "$((frames + 25))" \
    --timeout-seconds "$timeout_seconds" \
    --subscriber-warmup-s 1 \
    --evidence "$publisher_evidence" \
    --pico-client-apk-sha256 "$apk_sha256" &
publisher_pid=$!
if ! publisher_pgid="$(owned_group_pgid "$publisher_pid")"; then
  abort_unregistered_launch "$publisher_pid" TERM
  publisher_pid=""
  assert_no_runtime_tree
  exit 1
fi

printf 'role\tpid\ttid\tpgid\tcls\trtprio\tpsr\taffinity\tcomm\n' >"$scheduler_log"
for _ in {1..50}; do
  if [[ -f "$publisher_evidence" ]] &&
     grep -Fq '"event":"xrt_binding_verified"' "$publisher_evidence"; then
    break
  fi
  kill -0 "$publisher_pid" 2>/dev/null || break
  sleep 0.1
done
verify_and_log_group_scheduler shadow "$shadow_pgid" "$control_cpu_set"
verify_and_log_group_scheduler publisher "$publisher_pgid" "$publisher_cpu_set"
verify_and_log_process_scheduler service "$robotics_service_pid" "$service_cpu_set"

set +e
wait_owned_group_clean "$shadow_pgid" "$shadow_pid"
shadow_status=$?
shadow_pid=""
shadow_pgid=""
if ((shadow_status != 0)); then
  stop_owned_group_bounded "$publisher_pgid" "$publisher_pid" TERM
  publisher_status=$?
  publisher_pid=""
  publisher_pgid=""
else
  wait_owned_group_clean "$publisher_pgid" "$publisher_pid"
  publisher_status=$?
  publisher_pid=""
  publisher_pgid=""
fi
set -e
assert_no_runtime_tree

if ((shadow_status != 0 || publisher_status != 0)); then
  printf '[BLOCKED] promoted shadow=%d publisher=%d\n' \
    "$shadow_status" "$publisher_status" >&2
  printf 'Shadow evidence: %s\nPublisher evidence: %s\n' \
    "$shadow_evidence" "$publisher_evidence" >&2
  exit 1
fi

"$python" "$validator" publisher \
  --publisher-evidence "$publisher_evidence" \
  --shadow-evidence "$shadow_evidence" \
  --publisher-source "$publisher" \
  --worker-source "$worker" \
  --xrt-module "$xrt_module" \
  --apk-sha256 "$apk_sha256" \
  --minimum-packets "$frames" \
  --max-age-seconds 300

printf '[PASS] Promoted read-only integration complete.\n'
sha256sum "$shadow_evidence" "$publisher_evidence" "$scheduler_log"
