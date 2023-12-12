#!/bin/bash

set -e

# build container
docker buildx build -t ui --platform=linux/amd64 -f Dockerfile .

# run container
docker run --gpus all \
  -p 7860:7860 \
  -p 5000:5000 \
  -p 5005:5005 \
  -t ui