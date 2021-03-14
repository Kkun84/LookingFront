#!/bin/bash
docker build \
    --pull \
    --rm \
    -f "Dockerfile" \
    --build-arg UID=$(id -u) --build-arg GID=1000 --build-arg USER=hoge --build-arg PASSWORD=fuga \
    -t \
    looking_front:latest "."
