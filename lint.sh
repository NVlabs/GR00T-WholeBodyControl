#!/bin/bash
# Script to run linters the same way as in the GitLab CI pipeline

# Default mode is check only
FIX_MODE=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fix) FIX_MODE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Install required packages if not already installed
echo "Checking for required linting tools..."
pip install black ruff

# Set the mode message
if [ "$FIX_MODE" = true ]; then
    echo "Running in FIX mode - will automatically correct issues"
else
    echo "Running in CHECK mode - will only report issues"
fi

# Run Ruff lint checks
echo "Running Ruff linting checks..."
if [ "$FIX_MODE" = true ]; then
    python -m ruff check --fix .
else
    python -m ruff check .
fi

# Run Ruff import sorting and Black
echo "Running style checks..."
if [ "$FIX_MODE" = true ]; then
    python -m ruff check --select I --fix .
    python -m black .
else
    python -m ruff check --select I .
    python -m black --check .
fi

echo "Linting completed!" 