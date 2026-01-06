#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------------------
IMAGE="vortex-deep-learning-pipelines:latest"   # Docker image name/tag
BASE_IMAGE="ros:humble"                    # Base image for Docker builds

# ------------------------------------------------------------------------------
# Platform Detection
# ------------------------------------------------------------------------------
ARCHITECTURE="$(uname -m)"
if [[ "$ARCHITECTURE" == "arm64" || "$ARCHITECTURE" == "aarch64" ]]; then
    PLATFORM="linux/arm64"
elif [[ "$ARCHITECTURE" == "x86_64" ]]; then
    PLATFORM="linux/amd64"
else
    echo "Unsupported architecture: $ARCHITECTURE" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Locate Script and Workspace Root
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKSPACE="$(realpath "$SCRIPT_DIR/..")"

# ------------------------------------------------------------------------------
# Build Start Info
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " Building Docker Image"
echo "   • PLATFORM:       $PLATFORM"
echo "   • BASE_IMAGE:     $BASE_IMAGE"
echo "   • IMAGE:          $IMAGE"
echo "   • BUILD CONTEXT:  $WORKSPACE"
echo "======================================================================"
echo ""

# ------------------------------------------------------------------------------
# Build Docker Image with Buildx
# ------------------------------------------------------------------------------
docker buildx build \
    --platform "$PLATFORM" \
    --network=host \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --tag "$IMAGE" \
    --file "$SCRIPT_DIR/Dockerfile" \
    --load \
    "$WORKSPACE"

echo ""
echo "======================================================================"
echo " Successfully built image '$IMAGE' for platform '$PLATFORM'"
echo "======================================================================"
