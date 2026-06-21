#!/bin/bash
# install.sh - Prepare environment using uv

set -e

echo "=== Installing and preparing environment using uv ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command is not installed or not in PATH." >&2
    exit 1
fi

# Initialize project if pyproject.toml is missing
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing new Python project with uv..."
    uv init --no-workspace
fi

# Synchronize python dependencies and setup virtualenv
echo "Syncing dependencies..."
uv sync

echo "=== Environment preparation successful ==="
exit 0
