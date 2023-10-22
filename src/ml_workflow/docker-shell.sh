#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="ml_workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCS_BUCKET_NAME="sciencetutor-app-models-demo2"
export GCS_BUCKET_URI="gs://sciencetutor-app-models-demo2"
export GCP_PROJECT="ac215project-398401"
export WANDB_API_KEY=$(cat $(pwd)/../../../secrets/wandb.txt)
export GCS_SERVICE_ACCOUNT="ml-workflow@ac215project-398401.iam.gserviceaccount.com"
export GCP_REGION="us-central1" # Adjust region based on you approved quotas for GPUs
export GCS_PACKAGE_URI="gs://ac215-sciencetutor-trainer2"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=../secrets/ml-workflow-service-account.json \
-e GCP_PROJECT="$GCP_PROJECT" \
-e GCS_BUCKET_NAME="$GCS_BUCKET_NAME" \
-e GCS_BUCKET_URI="$GCS_BUCKET_URI" \
-e WANDB_API_KEY="$WANDB_API_KEY" \
-e GCS_SERVICE_ACCOUNT="$GCS_SERVICE_ACCOUNT" \
-e GCP_REGION="$GCP_REGION" \
-e GCS_PACKAGE_URI="$GCS_PACKAGE_URI" \
$IMAGE_NAME
