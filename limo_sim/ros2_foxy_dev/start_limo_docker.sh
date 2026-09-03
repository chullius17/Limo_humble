#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

xhost +local:docker

if docker container inspect limo_sim >/dev/null 2>&1; then
  CONTAINER_IMAGE="$(docker container inspect limo_sim --format '{{.Image}}')"
  LATEST_IMAGE="$(docker image inspect limo_sim:latest --format '{{.Id}}')"

  if [ "${CONTAINER_IMAGE}" != "${LATEST_IMAGE}" ]; then
    echo "The limo_sim image changed; recreating the container..."
    docker rm -f limo_sim
  fi
fi

if docker container inspect limo_sim >/dev/null 2>&1; then
  docker start -ai limo_sim
else
  docker run -it \
    --name limo_sim \
    --gpus all \
    --network host \
    --ipc host \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${SCRIPT_DIR}/../workspace:/workspace" \
    limo_sim bash
fi
