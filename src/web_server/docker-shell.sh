#!/bin/bash

set -e

# build container
docker build -t web_server -f Dockerfile .

# run container
docker run -it web_server