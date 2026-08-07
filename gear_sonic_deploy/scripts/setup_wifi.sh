#!/bin/bash

echo "=== [WiFi Setup Script] Starting... ==="

# WiFi network configuration (set via env or flags)
# Usage:
#   WIFI_SSID="MyWifi" WIFI_PASSWORD="secret" WIFI_GATEWAY="192.168.x.x" ./setup_wifi.sh
#   ./setup_wifi.sh --ssid "MyWifi" --password "secret" --interface "wlan0" --gateway "192.168.x.x"
WIFI_SSID="${WIFI_SSID:-}"
WIFI_PASSWORD="${WIFI_PASSWORD:-}"
CONNECTION_NAME="${CONNECTION_NAME:-}"
WIFI_INTERFACE="${WIFI_INTERFACE:-}"
FALLBACK_SSID="${FALLBACK_SSID:-}"

usage() {
    cat << 'USAGE_EOF'
Usage: setup_wifi.sh [options]

Options:
  --ssid <ssid>            WiFi SSID (required)
  --password <password>    WiFi password (required for secured networks)
  --interface <ifname>     WiFi interface (optional; auto-detect if omitted)
  --connection <name>      Connection profile name (optional; defaults to SSID)
  --gateway <ip>           Gateway IP (optional; sets route/DNS)
  --fallback-ssid <ssid>   Fallback SSID (optional)
  -h, --help               Show this help

You can also set environment variables:
  WIFI_SSID, WIFI_PASSWORD, WIFI_INTERFACE, CONNECTION_NAME, FALLBACK_SSID, WIFI_GATEWAY,
  ROUTE_CLEANUP_GW
USAGE_EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ssid) WIFI_SSID="$2"; shift 2 ;;
        --password) WIFI_PASSWORD="$2"; shift 2 ;;
        --interface) WIFI_INTERFACE="$2"; shift 2 ;;
        --connection) CONNECTION_NAME="$2"; shift 2 ;;
        --fallback-ssid) FALLBACK_SSID="$2"; shift 2 ;;
        --gateway) WIFI_GATEWAY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[!] Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

if [ -z "$WIFI_SSID" ]; then
    echo "[✗] Missing WiFi SSID. Provide --ssid or WIFI_SSID."
    usage
    exit 2
fi

if [ -z "$CONNECTION_NAME" ]; then
    CONNECTION_NAME="$WIFI_SSID"
fi

if [ -z "$WIFI_INTERFACE" ]; then
    WIFI_INTERFACE=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi"{print $1; exit}')
fi

if [ -z "$WIFI_INTERFACE" ]; then
    echo "[✗] No WiFi interface detected. Provide --interface or WIFI_INTERFACE."
    exit 2
fi

# Network configuration
# If WIFI_GATEWAY is set, routing and DNS will be configured to use it.
WIFI_GATEWAY="${WIFI_GATEWAY:-}"
# Optional default gateway to remove from routing table.
ROUTE_CLEANUP_GW="${ROUTE_CLEANUP_GW:-}"
WIRED_CONNECTION="Wired connection 1"

# Step 1: Unblock and enable WiFi
echo "[1] Unblocking WiFi and enabling interface..."
sudo rfkill unblock wifi
sudo ip link set $WIFI_INTERFACE up

# Step 2: Scan for networks to verify NetworkManager is working
echo "[2] Scanning for WiFi networks..."
sudo nmcli device wifi rescan
sleep 3

# Step 3: Check if the target network is visible
echo "[3] Checking for '$WIFI_SSID' network..."
if sudo nmcli device wifi list | grep -q "$WIFI_SSID"; then
    echo "[✓] Network '$WIFI_SSID' found in scan"
    # Try direct connection
    echo "[4] Attempting direct connection to '$WIFI_SSID'..."
    if [ -n "$WIFI_PASSWORD" ]; then
        CONNECT_CMD=(sudo nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PASSWORD")
    else
        CONNECT_CMD=(sudo nmcli device wifi connect "$WIFI_SSID")
    fi
    if "${CONNECT_CMD[@]}"; then
        echo "[✓] Successfully connected directly"
        CONNECTION_SUCCESS=true
    else
        echo "[!] Direct connection failed, trying manual profile method..."
        CONNECTION_SUCCESS=false
    fi
else
    echo "[!] Network '$WIFI_SSID' not found in scan (possibly hidden)"
    CONNECTION_SUCCESS=false
fi

# Step 4: If direct connection failed or network not found, create manual connection profile
if [ "$CONNECTION_SUCCESS" != "true" ]; then
    echo "[4] Creating manual connection profile for '$WIFI_SSID'..."
    
    # Remove existing connection profile if it exists
    sudo nmcli connection delete "$CONNECTION_NAME" 2>/dev/null || true
    
    # Create new connection profile
    if [ -n "$WIFI_PASSWORD" ]; then
        ADD_ARGS=(wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$WIFI_PASSWORD")
    else
        ADD_ARGS=()
    fi
    if sudo nmcli connection add type wifi con-name "$CONNECTION_NAME" ifname "$WIFI_INTERFACE" ssid "$WIFI_SSID" "${ADD_ARGS[@]}"; then
        echo "[✓] Connection profile created successfully"
        
        echo "[5] Activating connection profile..."
        if sudo nmcli connection up "$CONNECTION_NAME"; then
            echo "[✓] Connection profile activated successfully"
            CONNECTION_SUCCESS=true
        else
            echo "[✗] Failed to activate connection profile"
            CONNECTION_SUCCESS=false
        fi
    else
        echo "[✗] Failed to create connection profile"
        CONNECTION_SUCCESS=false
    fi
fi

# Step 5: Configure network routing priorities if connection successful
if [ "$CONNECTION_SUCCESS" = "true" ]; then
    echo "[5] Configuring network routing priorities..."
    
    # Set WiFi connection to high priority (low metric)
    echo "    Setting WiFi connection priority to 100..."
    sudo nmcli connection modify "$CONNECTION_NAME" ipv4.route-metric 100
    
    # Set wired connection to very low priority (high metric) 
    echo "    Setting wired connection priority to 30000..."
    sudo nmcli connection modify "$WIRED_CONNECTION" ipv4.route-metric 30000 2>/dev/null || true
    
    # Remove any problematic default routes
    echo "    Cleaning up routing table..."
    if [ -n "$ROUTE_CLEANUP_GW" ]; then
        sudo ip route del default via "$ROUTE_CLEANUP_GW" 2>/dev/null || true
    fi
    
    # Ensure WiFi route has highest priority
    if [ -n "$WIFI_GATEWAY" ]; then
        sudo ip route add default via "$WIFI_GATEWAY" dev "$WIFI_INTERFACE" metric 50 2>/dev/null || true
        
        # Configure DNS to use router and public DNS
        echo "    Configuring DNS..."
        sudo tee /etc/resolv.conf > /dev/null << DNS_EOF
nameserver $WIFI_GATEWAY
nameserver 8.8.8.8
nameserver 1.1.1.1
DNS_EOF
    fi
    
    echo "[✓] Network routing configured for WiFi priority"
fi

# Step 6: Verify connection status
echo "[6] Checking connection status..."
if [ "$CONNECTION_SUCCESS" = "true" ]; then
    CURRENT_CONNECTION=$(nmcli device status | grep "$WIFI_INTERFACE" | awk '{print $4}')
    if [[ "$CURRENT_CONNECTION" == "$CONNECTION_NAME" ]] || [[ "$CURRENT_CONNECTION" == "$WIFI_SSID" ]]; then
        echo "[✓] Successfully connected to '$WIFI_SSID'"
        echo "    Interface: $WIFI_INTERFACE"
        echo "    Connection: $CURRENT_CONNECTION"
    else
        echo "[!] Connection status unclear"
        CONNECTION_SUCCESS=false
    fi
fi

# Step 7: Test internet connectivity
if [ "$CONNECTION_SUCCESS" = "true" ]; then
    echo "[7] Testing internet connectivity..."
    MAX_RETRIES=3
    COUNT=0
    INTERNET_OK=false
    
    while [ "$COUNT" -lt "$MAX_RETRIES" ]; do
        if ping -c 2 -W 5 8.8.8.8 >/dev/null 2>&1; then
            echo "[✓] Internet connectivity verified (ping test)"
            
            # Also test DNS resolution
            if curl -s --connect-timeout 5 -I http://google.com >/dev/null 2>&1; then
                echo "[✓] DNS resolution and HTTP connectivity verified"
                INTERNET_OK=true
                break
            else
                echo "[!] DNS/HTTP test failed, but ping works. Retrying... ($((COUNT+1))/$MAX_RETRIES)"
            fi
        else
            echo "[!] Internet test failed. Retrying... ($((COUNT+1))/$MAX_RETRIES)"
        fi
        
        sleep 3
        ((COUNT++))
    done
    
    if [ "$INTERNET_OK" != "true" ]; then
        echo "[!] Internet connectivity test failed after $MAX_RETRIES attempts"
        CONNECTION_SUCCESS=false
    fi
fi

# Step 8: Fallback to previous network if connection failed
if [ "$CONNECTION_SUCCESS" != "true" ] && [ -n "$FALLBACK_SSID" ]; then
    echo "[8] Connection to '$WIFI_SSID' failed. Attempting fallback to '$FALLBACK_SSID'..."
    
    # Try to connect to fallback network
    if sudo nmcli device wifi list | grep -q "$FALLBACK_SSID"; then
        echo "[!] Attempting to connect to fallback network '$FALLBACK_SSID'..."
        if sudo nmcli device wifi connect "$FALLBACK_SSID"; then
            echo "[✓] Connected to fallback network '$FALLBACK_SSID'"
            
            # Apply routing fixes for fallback network too
            echo "    Applying routing fixes for fallback network..."
            if [ -n "$ROUTE_CLEANUP_GW" ]; then
                sudo ip route del default via "$ROUTE_CLEANUP_GW" 2>/dev/null || true
            fi
            sudo nmcli connection modify "$WIRED_CONNECTION" ipv4.route-metric 30000 2>/dev/null || true
        else
            echo "[✗] Failed to connect to fallback network"
        fi
    else
        echo "[!] Fallback network '$FALLBACK_SSID' not available"
    fi
fi

# Step 9: Final status report
echo "[9] Final WiFi status:"
nmcli device status | grep wifi
echo ""
echo "Active connections:"
nmcli connection show --active | head -5
echo ""
echo "Current routing table:"
ip route show | head -5
echo ""
echo "DNS configuration:"
cat /etc/resolv.conf | grep nameserver

echo "=== [WiFi Setup Script] Completed ==="

# Exit with appropriate code
if [ "$CONNECTION_SUCCESS" = "true" ]; then
    exit 0
else
    exit 1
fi
