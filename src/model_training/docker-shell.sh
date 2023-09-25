#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="model_training"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app -v "$SECRETS_DIR":/secrets $IMAGE_NAME