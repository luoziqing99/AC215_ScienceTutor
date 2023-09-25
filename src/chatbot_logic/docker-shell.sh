#!/bin/bash

set -e

# build container
docker build -t chatbot_logic -f Dockerfile .

# run container
docker run -it chatbot_logic