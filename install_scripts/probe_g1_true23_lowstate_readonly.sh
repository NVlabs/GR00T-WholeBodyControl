#!/usr/bin/env bash
set -euo pipefail

interface="${1:-eth0}"
duration_s="${2:-4.0}"
local_ip="$(ip -4 -o addr show dev "${interface}" | awk '{split($4, a, "/"); print a[1]; exit}')"
if [[ -z "${local_ip}" ]]; then
  echo "[FAIL] no IPv4 address on ${interface}" >&2
  exit 1
fi

keeper_pid=""
capture_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${capture_pid}" ]]; then
    kill -TERM "${capture_pid}" 2>/dev/null || true
    wait "${capture_pid}" 2>/dev/null || true
  fi
  if [[ -n "${keeper_pid}" ]]; then
    kill -TERM "${keeper_pid}" 2>/dev/null || true
    wait "${keeper_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

/usr/bin/python3 - "${local_ip}" <<'PY' &
import socket
import sys
import time

local_ip = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", 0))
membership = socket.inet_aton("239.255.0.1") + socket.inet_aton(local_ip)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
print(f"[PASS] IGMP membership held on {local_ip}", flush=True)
time.sleep(30)
PY
keeper_pid=$!

tcpdump -ni "${interface}" -w /dev/null \
  'udp and dst host 239.255.0.1 and dst port 7401' \
  >/dev/null 2>&1 &
capture_pid=$!
sleep 1

PYTHONPATH="/mnt/z/codex/GR00T-WholeBodyControl:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python" \
  /usr/bin/python3 - "${interface}" "${duration_s}" <<'PY'
import sys

from gear_sonic.scripts.pico_g1_preflight import Checks, _probe_lowstate

checks = Checks()
_probe_lowstate(
    checks,
    sys.argv[1],
    float(sys.argv[2]),
    policy_profile="true23",
)
result = checks.results[0]
print(f"{result.status}: {result.detail}", flush=True)
raise SystemExit(0 if result.status == "PASS" else 1)
PY
