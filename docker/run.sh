#!/bin/bash
docker run \
    -d \
    --gpus=all \
    --init \
    --rm \
    -p 5000:5000 \
    -p 6006:6006 \
    -p 8501:8501 \
    -p 8888:8888 \
    -it \
    --ipc=host \
    --name=LookingFront \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$DATASET:/dataset \
    looking_front:latest \
    ${@-fish}

