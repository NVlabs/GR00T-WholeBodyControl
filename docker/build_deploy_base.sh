#!/bin/bash

# Multi-architecture Docker build script
# Supports linux/amd64

set -e

# Configuration
IMAGE_NAME="nvgear/ros-2"
TAG="${1:-latest}"
DOCKERFILE="docker/Dockerfile.deploy.base"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building multi-architecture Docker image: ${IMAGE_NAME}:${TAG}${NC}"

# Ensure we're using the multiarch builder
echo -e "${YELLOW}Setting up multiarch builder...${NC}"
sudo docker buildx use multiarch-builder 2>/dev/null || {
    echo -e "${YELLOW}Creating multiarch builder...${NC}"
    sudo docker buildx create --name multiarch-builder --use --bootstrap
}

# Show supported platforms
echo -e "${YELLOW}Supported platforms:${NC}"
sudo docker buildx inspect --bootstrap | grep Platforms

# Build for multiple architectures
echo -e "${GREEN}Starting multi-arch build...${NC}"
sudo docker buildx build \
    --platform linux/amd64 \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${TAG}" \
    --push \
    .

# Alternative: Build and load locally (only works for single platform)
# docker buildx build \
#     --platform linux/amd64 \
#     --file "${DOCKERFILE}" \
#     --tag "${IMAGE_NAME}:${TAG}" \
#     --load \
#     .

echo -e "${GREEN}Multi-arch build completed successfully!${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${TAG}${NC}"
echo -e "${GREEN}Platforms: linux/amd64${NC}"

# Verify the manifest
echo -e "${YELLOW}Verifying multi-arch manifest...${NC}"
sudo docker buildx imagetools inspect "${IMAGE_NAME}:${TAG}" 