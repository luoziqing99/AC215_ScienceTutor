#!/bin/bash

set -e

# build container
docker buildx build -t data_processing --platform=linux/amd64 -f Dockerfile .

# run container
docker run -it data_processing

# push container
#docker tag data_processing jenniferz99/data_processing
#docker push jenniferz99/data_processing