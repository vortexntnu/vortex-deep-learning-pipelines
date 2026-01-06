#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Image Configuration
# ------------------------------------------------------------------------------
IMAGE="vortex-deep-learning-pipelines:latest"  # Default Docker image name/tag

# ------------------------------------------------------------------------------
# Locate Script Directory and Workspace Root
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKSPACE="$(realpath "$SCRIPT_DIR/..")"

echo "======================================================================"
echo " Running Container"
echo "   • IMAGE:          $IMAGE"
echo "   • Volume mount:   $WORKSPACE -> /ros2_ws"
echo "======================================================================"
echo ""

# ------------------------------------------------------------------------------
# Run Docker Container
# ------------------------------------------------------------------------------
docker run -it --rm \
    --privileged \
    --network host \
    --ipc=host \
    -v "$WORKSPACE":/ros2_ws \
    -w /ros2_ws \
    "$IMAGE" \
    /bin/bash
