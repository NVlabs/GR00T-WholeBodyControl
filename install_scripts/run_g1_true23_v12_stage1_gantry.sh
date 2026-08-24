#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

# Bounded stage-one launcher. It creates no authorization and does not contain
# the gantry phrase; the operator must type that phrase into a real TTY.

active_promotion=""
live_shadow_evidence=""
reviewed_publisher_evidence=""
authorization_id=""
network=""
pico_endpoint=""
duration_seconds=""
execute_stage_one=false
validate_only=false

# Same reviewed scheduling contract as promoted shadow: two physical cores for
# control/DDS, one reserved for WSL/kernel/XRT service, and five for the causal
# publisher plus inherited capture worker. Whole-process SCHED_RR is forbidden
# because inherited solver/native threads can starve LowState.
control_cpu_set="0-3"
service_cpu_set="4,5"
publisher_cpu_set="6-15"
expected_online_cpus="0-15"
robotics_service_pid=""
robotics_service_binary="/opt/apps/roboticsservice/RoboticsServiceProcess"
expected_robotics_service_sha256="8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc"

usage() {
  printf '%s\n' \
    "Usage: $0 --active-promotion <sidecar.json> --live-shadow-evidence <pass.jsonl>" \
    "          --publisher-evidence <paired-pass.jsonl>" \
    "          --authorization-id <id> --network eth0" \
    "          --pico-endpoint tcp://127.0.0.1:5557" \
    "          [--validate-only | --duration-seconds <20..30> --execute-stage-one]"
}

while (($#)); do
  case "$1" in
    --active-promotion)
      active_promotion="${2:?--active-promotion requires a value}"
      shift 2
      ;;
    --live-shadow-evidence)
      live_shadow_evidence="${2:?--live-shadow-evidence requires a value}"
      shift 2
      ;;
    --publisher-evidence)
      reviewed_publisher_evidence="${2:?--publisher-evidence requires a value}"
      shift 2
      ;;
    --authorization-id)
      authorization_id="${2:?--authorization-id requires a value}"
      shift 2
      ;;
    --network)
      network="${2:?--network requires a value}"
      shift 2
      ;;
    --pico-endpoint)
      pico_endpoint="${2:?--pico-endpoint requires a value}"
      shift 2
      ;;
    --duration-seconds)
      duration_seconds="${2:?--duration-seconds requires a value}"
      shift 2
      ;;
    --execute-stage-one)
      execute_stage_one=true
      shift
      ;;
    --validate-only)
      validate_only=true
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

if [[ -z "$active_promotion" || -z "$live_shadow_evidence" ||
      -z "$reviewed_publisher_evidence" ||
      -z "$authorization_id" || -z "$network" || -z "$pico_endpoint" ]]; then
  printf 'All sidecar, evidence, identity, network, and endpoint arguments are required.\n' >&2
  exit 2
fi
if [[ ! "$authorization_id" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$ ]]; then
  printf 'authorization-id must contain 8-128 safe characters.\n' >&2
  exit 2
fi
if [[ "$network" != "eth0" ||
      "$pico_endpoint" != "tcp://127.0.0.1:5557" ]]; then
  printf 'This reviewed V12 stage-one wrapper requires eth0 and loopback port 5557.\n' >&2
  exit 2
fi
if "$validate_only"; then
  if "$execute_stage_one" || [[ -n "$duration_seconds" ]]; then
    printf -- '--validate-only rejects execution/duration arguments.\n' >&2
    exit 2
  fi
else
  if ! "$execute_stage_one" ||
     [[ ! "$duration_seconds" =~ ^[1-9][0-9]*$ ]] ||
     ((duration_seconds < 20 || duration_seconds > 30)); then
    printf 'Execution requires --execute-stage-one and duration 20..30 seconds.\n' >&2
    exit 2
  fi
  if [[ ! -t 0 || ! -t 1 ]]; then
    printf 'Stage-one execution requires an interactive operator TTY.\n' >&2
    exit 2
  fi
fi

repo="/mnt/z/codex/GR00T-WholeBodyControl"
binary="$repo/gear_sonic_deploy/target/release/g1_true23_active_gantry"
expected_binary_sha256="78d71371d74bb64fa88463753fdc2b71c711143a8c58823592707e8a78466530"
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
apk_sha256="e4ac5adb615bf26b49ab4725a9d6b28b1c290682682c010922e2cc66a5f669ca"
evidence_root="/root/g1_true23_runs/live"

for required in \
  "$binary" "$publisher" "$worker" "$validator" "$xrt_module" \
  "$encoder" "$decoder" "$metadata" "$promotion" "$active_promotion" \
  "$live_shadow_evidence" "$reviewed_publisher_evidence"; do
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

actual_binary_sha256="$(sha256sum "$binary" | awk '{print $1}')"
if [[ "$actual_binary_sha256" != "$expected_binary_sha256" ]]; then
  printf '[BLOCKED] Active controller byte hash changed: expected %s, got %s\n' \
    "$expected_binary_sha256" "$actual_binary_sha256" >&2
  exit 1
fi
for hash_binding in \
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

active_promotion="$(realpath -e -- "$active_promotion")"
live_shadow_evidence="$(realpath -e -- "$live_shadow_evidence")"
reviewed_publisher_evidence="$(realpath -e -- "$reviewed_publisher_evidence")"
case "$active_promotion" in
  /root/g1_true23_eval/v12/*promotion*.json) ;;
  *)
    printf 'Active promotion path/name is outside the reviewed V12 location.\n' >&2
    exit 1
    ;;
esac
case "$reviewed_publisher_evidence" in
  /root/g1_true23_runs/live/v12_pico_causal_publisher_*.jsonl) ;;
  *)
    printf 'Publisher evidence is outside the reviewed paired-run location.\n' >&2
    exit 1
    ;;
esac

"$python" "$validator" publisher \
  --publisher-evidence "$reviewed_publisher_evidence" \
  --shadow-evidence "$live_shadow_evidence" \
  --publisher-source "$publisher" \
  --worker-source "$worker" \
  --xrt-module "$xrt_module" \
  --apk-sha256 "$apk_sha256" \
  --minimum-packets 100 \
  --max-age-seconds 300
case "$live_shadow_evidence" in
  /root/g1_true23_runs/live/v12_promoted_shadow_*.jsonl) ;;
  *)
    printf 'Live-shadow evidence is outside the reviewed promoted-shadow location.\n' >&2
    exit 1
    ;;
esac

preflight=(
  "$binary"
  --network "$network"
  --pico-endpoint "$pico_endpoint"
  --authorization-id "$authorization_id"
  --encoder "$encoder"
  --decoder "$decoder"
  --metadata "$metadata"
  --promotion "$promotion"
  --active-promotion "$active_promotion"
  --live-shadow-evidence "$live_shadow_evidence"
  --validate-only
)
"${preflight[@]}"
if "$validate_only"; then
  printf '[PASS] Stage-one bytes, evidence, and CPU isolation validated; no channels opened.\n'
  exit 0
fi

competing_processes="$(
  pgrep -af \
    '(^|/)(g1_deploy_onnx_ref|g1_true23_gantry_hold_smoke|g1_true23_active_gantry|g1_true23_live_shadow|stream_g1_23dof_pico_causal_zmq.py|stream_g1_23dof_pico_raw_worker.py)( |$)' \
    || true
)"
if [[ -n "$competing_processes" ]]; then
  printf '[BLOCKED] Competing robot command or PICO publisher process detected:\n%s\n' \
    "$competing_processes" >&2
  exit 1
fi
printf '%s' 'Type the exact gantry authorization phrase, then press Enter: '
IFS= read -r -s gantry_authorization
printf '\n'
if [[ -z "$gantry_authorization" ]]; then
  printf 'Empty gantry authorization rejected.\n' >&2
  exit 2
fi

# The prompt is intentionally unbounded; repeat every mutable host/process
# preflight after it so an operator pause cannot invalidate the launch checks.
validate_cpu_isolation
competing_processes="$(
  pgrep -af \
    '(^|/)(g1_deploy_onnx_ref|g1_true23_gantry_hold_smoke|g1_true23_active_gantry|g1_true23_live_shadow|stream_g1_23dof_pico_causal_zmq.py|stream_g1_23dof_pico_raw_worker.py)( |$)' \
    || true
)"
if [[ -n "$competing_processes" ]]; then
  gantry_authorization=""
  printf '[BLOCKED] Process state changed during authorization:\n%s\n' \
    "$competing_processes" >&2
  exit 1
fi

mkdir -p "$evidence_root"
run_id="$(date +%Y%m%d_%H%M%S)"
publisher_evidence="$evidence_root/v12_stage1_pico_publisher_${run_id}.jsonl"
active_evidence="$evidence_root/v12_stage1_active_${run_id}.jsonl"
scheduler_log="$evidence_root/v12_stage1_scheduler_${run_id}.log"
if [[ -e "$publisher_evidence" || -e "$active_evidence" ||
      -e "$scheduler_log" ]]; then
  printf 'Stage-one evidence destination collision.\n' >&2
  exit 1
fi

publisher_pid=""
publisher_pgid=""
active_pid=""
active_pgid=""
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
emergency_cleanup() {
  trap - EXIT INT TERM
  gantry_authorization=""
  if [[ -n "$active_pid" ]]; then
    stop_owned_group_bounded "$active_pgid" "$active_pid" INT || true
    active_pid=""
    active_pgid=""
  fi
  if [[ -n "$publisher_pid" ]]; then
    stop_owned_group_bounded "$publisher_pgid" "$publisher_pid" TERM || true
    publisher_pid=""
    publisher_pgid=""
  fi
  if ! assert_no_runtime_tree; then
    exit 125
  fi
}
stop_publisher_gracefully() {
  local publisher_status=0
  if [[ -z "$publisher_pid" || -z "$publisher_pgid" ]]; then
    return 1
  fi
  if stop_owned_group_bounded "$publisher_pgid" "$publisher_pid" TERM; then
    publisher_status=0
  else
    publisher_status=$?
  fi
  publisher_pid=""
  publisher_pgid=""
  assert_no_runtime_tree || return 1
  ((publisher_status == 0))
}
trap emergency_cleanup EXIT
trap 'exit 130' INT TERM

# Keep the finite publisher target beyond the controller's entire outer timeout
# so successful Stage-1 always terminates it through the reviewed SIGTERM path.
packet_budget=$(((duration_seconds + 120) * 50))
PYTHONPATH="/root/.cache/g1_true23_soma/source:$repo" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  setsid taskset -c "$publisher_cpu_set" "$python" "$publisher" \
    --bind "$pico_endpoint" \
    --xrt-module-dir "$xrt_module_dir" \
    --packets "$packet_budget" \
    --timeout-seconds "$((duration_seconds + 150))" \
    --subscriber-warmup-s 5 \
    --evidence "$publisher_evidence" \
    --pico-client-apk-sha256 "$apk_sha256" &
publisher_pid=$!
if ! publisher_pgid="$(owned_group_pgid "$publisher_pid")"; then
  abort_unregistered_launch "$publisher_pid" TERM
  publisher_pid=""
  assert_no_runtime_tree
  exit 1
fi

publisher_live=false
for _ in $(seq 1 300); do
  if ! kill -0 "$publisher_pid" 2>/dev/null; then
    break
  fi
  if [[ -f "$publisher_evidence" ]] &&
     grep -Fq '"event":"reference_packet_published"' \
       "$publisher_evidence"; then
    publisher_live=true
    break
  fi
  sleep 0.2
done
if ! "$publisher_live"; then
  printf '[BLOCKED] PICO publisher did not prove a fresh reference packet.\n' >&2
  exit 1
fi

printf 'role\tpid\ttid\tpgid\tcls\trtprio\tpsr\taffinity\tcomm\n' >"$scheduler_log"
verify_and_log_group_scheduler publisher "$publisher_pgid" "$publisher_cpu_set"
verify_and_log_process_scheduler service "$robotics_service_pid" "$service_cpu_set"

setsid taskset -c "$control_cpu_set" \
  timeout --foreground --preserve-status --signal=INT --kill-after=5s \
  "$((duration_seconds + 90))s" \
  "$binary" \
    --network "$network" \
    --pico-endpoint "$pico_endpoint" \
    --authorization-id "$authorization_id" \
    --encoder "$encoder" \
    --decoder "$decoder" \
    --metadata "$metadata" \
    --promotion "$promotion" \
    --active-promotion "$active_promotion" \
    --live-shadow-evidence "$live_shadow_evidence" \
    --evidence "$active_evidence" \
    --post-arm-duration-seconds "$duration_seconds" \
    --execute-stage-one \
    --gantry-authorize "$gantry_authorization" &
active_pid=$!
if ! active_pgid="$(owned_group_pgid "$active_pid")"; then
  abort_unregistered_launch "$active_pid" INT
  active_pid=""
  exit 1
fi
gantry_authorization=""
for _ in {1..50}; do
  if [[ -f "$active_evidence" ]] &&
     grep -Fq '"event":"session_start"' "$active_evidence"; then
    break
  fi
  kill -0 "$active_pid" 2>/dev/null || break
  sleep 0.1
done
verify_and_log_group_scheduler active "$active_pgid" "$control_cpu_set"
set +e
wait_owned_group_clean "$active_pgid" "$active_pid"
active_status=$?
set -e
active_pid=""
active_pgid=""
if ! stop_publisher_gracefully; then
  printf '[BLOCKED] Live PICO publisher did not stop with reviewed terminal evidence. Evidence: %s\n' \
    "$publisher_evidence" >&2
  exit 1
fi
trap - EXIT INT TERM

if ((active_status != 0)); then
  printf '[BLOCKED] Stage-one controller exited %d. Active evidence: %s Publisher evidence: %s\n' \
    "$active_status" "$active_evidence" "$publisher_evidence" >&2
  exit "$active_status"
fi
"$python" "$validator" publisher-runtime \
  --publisher-evidence "$publisher_evidence" \
  --active-evidence "$active_evidence" \
  --publisher-source "$publisher" \
  --worker-source "$worker" \
  --xrt-module "$xrt_module" \
  --apk-sha256 "$apk_sha256" \
  --binary "$binary" \
  --shadow-evidence "$live_shadow_evidence" \
  --active-promotion "$active_promotion" \
  --authorization-id "$authorization_id" \
  --minimum-packets "$((duration_seconds * 50 + 10))" \
  --max-age-seconds 300
printf '[COMPLETE] Stage-one policy actuation and deterministic damping proved.\n'
sha256sum "$active_evidence" "$publisher_evidence" "$scheduler_log"
