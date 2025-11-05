#!/bin/bash
set -e # Exit on error

# Run the deployment script
# Check for script existence before running
DEPLOY_SCRIPT="scripts/deploy_g1.py"
if [ -f "$DEPLOY_SCRIPT" ]; then
    echo "Running deployment script at $DEPLOY_SCRIPT"
    echo "Using python from $(which python)"
    echo "Deploy args: $@"
    exec python "$DEPLOY_SCRIPT" "$@"
else
    echo "ERROR: Deployment script not found at $DEPLOY_SCRIPT"
    echo "Current directory structure:"
    find . -type f -name "*.py" | grep -i deploy
    echo "Available script options:"
    find . -type f -name "*.py" | sort
    echo "Starting a bash shell for troubleshooting..."
    exec /bin/bash
fi