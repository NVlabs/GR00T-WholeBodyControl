#!/bin/bash

# Script to kill all running Docker containers
# Usage: ./kill_all_containers.sh [--force]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! sudo docker info >/dev/null 2>&1; then
    print_error "Docker is not running or not accessible. Please start Docker first."
    exit 1
fi

# Get list of running containers
RUNNING_CONTAINERS=$(sudo docker ps -q)

if [ -z "$RUNNING_CONTAINERS" ]; then
    print_info "No running containers found."
    exit 0
fi

# Count running containers
CONTAINER_COUNT=$(echo "$RUNNING_CONTAINERS" | wc -l | tr -d ' ')

print_info "Found $CONTAINER_COUNT running container(s):"
sudo docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}"

# Check for --force flag
FORCE_KILL=false
if [ "$1" = "--force" ]; then
    FORCE_KILL=true
    print_warning "Force mode enabled. Containers will be killed without confirmation."
fi

# Ask for confirmation unless --force is used
if [ "$FORCE_KILL" = false ]; then
    echo
    read -p "Are you sure you want to kill all running containers? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled."
        exit 0
    fi
fi

# Kill all running containers
print_info "Killing all running containers..."
if sudo docker kill $RUNNING_CONTAINERS; then
    print_info "Successfully killed all running containers."
else
    print_error "Failed to kill some containers. You may need to run with sudo or check Docker permissions."
    exit 1
fi

# Optional: Remove stopped containers (commented out by default)
# Uncomment the following lines if you also want to remove the stopped containers
# print_info "Removing stopped containers..."
# sudo docker container prune -f

print_info "Done!" 