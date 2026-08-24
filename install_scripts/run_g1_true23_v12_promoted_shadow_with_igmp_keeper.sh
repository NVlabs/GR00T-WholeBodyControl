#!/usr/bin/env bash
set -euo pipefail

# WSL mirrored networking only forwards the Unitree DDS multicast reliably
# while an explicit IGMP membership is held. This helper opens no robot API;
# it only keeps 239.255.0.1 joined while the frozen read-only wrapper runs.
keeper_pid=""
capture_pid=""
cleanup() {
  trap - EXIT INT TERM
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

/usr/bin/python3 -c \
  "import socket,time; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('',0)); s.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,socket.inet_aton('239.255.0.1')+socket.inet_aton('192.168.123.200')); print('KEEPER_JOINED',flush=True); time.sleep(150)" &
keeper_pid=$!

# Hyper-V mirrored networking can revoke multicast delivery when a DDS
# subscriber closes even though a second IGMP membership remains. Holding a
# read-only capture open keeps the mirrored packet path active for the frozen
# subscriber. Payload is discarded and no packet is transmitted.
taskset -c 4-5 tcpdump -ni eth0 -w /dev/null \
  'udp and dst host 239.255.0.1 and dst port 7401' \
  >/dev/null 2>&1 &
capture_pid=$!
sleep 1

PYTHONPATH="/mnt/z/codex/GR00T-WholeBodyControl:/root/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python" \
  /usr/bin/python3 -c \
  "from gear_sonic.scripts.pico_g1_preflight import Checks,_probe_lowstate; c=Checks(); _probe_lowstate(c,'eth0',4.0,policy_profile='true23'); r=c.results[0]; print(r.status+': '+r.detail,flush=True); raise SystemExit(0 if r.status=='PASS' else 1)"

bash /mnt/z/codex/GR00T-WholeBodyControl/install_scripts/run_g1_true23_v12_promoted_shadow.sh "$@"
