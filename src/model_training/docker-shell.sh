#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="model_training"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_URI="gs://ac215-sciencetutor-trainer"
export GCP_PROJECT="ac215project-398401"
export WANDB_API_KEY=$(cat $(pwd)/../../../secrets/wandb.txt)

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=../secrets/model-trainer.json \
-e GCP_PROJECT="$GCP_PROJECT" \
-e GCS_BUCKET_URI="$GCS_BUCKET_URI" \
-e WANDB_API_KEY="$WANDB_API_KEY" \
$IMAGE_NAME
