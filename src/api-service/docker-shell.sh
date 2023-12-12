#!/bin/bash

set -e

# build docker
# docker build . -t backend
docker buildx build -t backend --platform=linux/amd64 -f Dockerfile .

# use all your GPUs 
# it will hang, until you manually terminate the container
# access the backend endpoint at http://localhost:5000/chat
docker run --gpus all -p 5000:5000 -t backend