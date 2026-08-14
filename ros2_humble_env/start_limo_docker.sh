#!/bin/bash

xhost +local:docker

sudo docker start limo_dev 2>/dev/null || \
sudo docker run -it \
--name limo_dev \
--gpus all \
--net=host \
-e DISPLAY=$DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/limo_humble:/workspace \
limo_humble_dev bash
