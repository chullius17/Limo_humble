#!/bin/bash

docker run \
-it \
--rm \
--gpus all \
--network host \
--ipc host \
-e DISPLAY=$DISPLAY \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $(pwd)/workspace:/workspace \
limo_humble_dev
