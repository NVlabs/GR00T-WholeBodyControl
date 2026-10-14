#!/usr/bin/env bash
set -euo pipefail
bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    --help|-h) echo 'Usage: ./run.sh [--check]'; echo 'Configuration: bridge.env beside this script. --check never opens serial devices.'; exit 0 ;;
    ''|--check) ;;
    *) echo 'Usage: ./run.sh [--check]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
# Parse data only: no shell sourcing, interpolation or eval.
LEFT_DEVICE='' RIGHT_DEVICE='' COMMAND_ENDPOINT='' STATE_ENDPOINT=''
BAUD=57600 LEFT_ID=1 RIGHT_ID=1 RATE_HZ=50 RUNTIME_SPEED=1000
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || { echo 'Expected KEY=value in bridge.env' >&2; exit 2; }
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in
        LEFT_DEVICE|RIGHT_DEVICE|COMMAND_ENDPOINT|STATE_ENDPOINT|BAUD|LEFT_ID|RIGHT_ID|RATE_HZ|RUNTIME_SPEED)
            printf -v "$key" '%s' "$value" ;;
        *) echo "Unknown bridge.env key: $key" >&2; exit 2 ;;
    esac
done < "$bridge_dir/bridge.env"
for device in "$LEFT_DEVICE" "$RIGHT_DEVICE"; do
    [[ "$device" == /* && -c "$device" && -r "$device" && -w "$device" ]] || {
        echo "Serial device must be an accessible character device: $device" >&2; exit 2;
    }
done
[[ "$(readlink -f -- "$LEFT_DEVICE")" != "$(readlink -f -- "$RIGHT_DEVICE")" ]] || {
    echo 'Left and right devices must differ' >&2; exit 2;
}
for endpoint in "$COMMAND_ENDPOINT" "$STATE_ENDPOINT"; do
    [[ "$endpoint" =~ ^tcp://127\.0\.0\.1:([0-9]{1,5})$ ]] || {
        echo 'Use tcp://127.0.0.1:PORT endpoints with SSH forwarding' >&2; exit 2;
    }
    port="${BASH_REMATCH[1]}"
    (( 10#$port >= 1 && 10#$port <= 65535 )) || { echo 'Invalid port' >&2; exit 2; }
done
[[ "$COMMAND_ENDPOINT" != "$STATE_ENDPOINT" ]] || { echo 'Endpoints must differ' >&2; exit 2; }
for key in LEFT_ID RIGHT_ID RATE_HZ RUNTIME_SPEED; do
    value="${!key}"
    [[ "$value" =~ ^[1-9][0-9]{0,3}$ ]] || { echo "Invalid $key" >&2; exit 2; }
    case "$key" in LEFT_ID|RIGHT_ID) max=254 ;; RATE_HZ) max=100 ;; RUNTIME_SPEED) max=1000 ;; esac
    (( value <= max )) || { echo "Invalid $key" >&2; exit 2; }
done
case "$BAUD" in 9600|19200|38400|57600|115200) ;; *) echo 'Unsupported BAUD' >&2; exit 2 ;; esac
binary="$bridge_dir/inspire_hand_bridge"
[[ -x "$binary" ]] || { echo "Build and install $binary first" >&2; exit 2; }
if [[ "${1:-}" == --check ]]; then
    echo "Configuration OK: left=$LEFT_DEVICE right=$RIGHT_DEVICE rate=$RATE_HZ Hz baud=$BAUD"
    echo 'No serial devices opened; no commands sent.'
    exit 0
fi
export G1_RL_SESSION_CONTROL=1
exec "$binary" "$RIGHT_DEVICE" "$LEFT_DEVICE" "$COMMAND_ENDPOINT" "$STATE_ENDPOINT" \
    "$RUNTIME_SPEED" "$RATE_HZ" "$bridge_dir/writer.lock" "$BAUD" "$RIGHT_ID" "$LEFT_ID"
