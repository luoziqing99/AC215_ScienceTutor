FROM node:14.18.0-buster-slim

RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    npm install -g http-server && \
    mkdir -p /app

EXPOSE 8080

WORKDIR /app

ENTRYPOINT ["/bin/bash"]