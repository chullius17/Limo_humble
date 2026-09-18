#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export DISPLAY="${DISPLAY:-:0}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker non accessibile. Eseguire ./setup-host.sh e poi riconnettersi." >&2
  exit 1
fi

allow_x11() {
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || \
      echo "Avviso: X11 non autorizzato; RViz/Gazebo potrebbero non aprirsi." >&2
  fi
}

case "${1:-shell}" in
  build)
    docker compose build
    ;;
  up)
    allow_x11
    docker compose up -d --build
    ;;
  shell)
    allow_x11
    docker compose up -d
    docker compose exec foxy bash
    ;;
  build-workspace)
    docker compose up -d
    docker compose exec foxy bash -lc \
      'sudo apt-get update && rosdep install --from-paths src --ignore-src -r -y --rosdistro foxy && colcon build --symlink-install'
    ;;
  gpu)
    docker compose up -d
    docker compose exec -T foxy nvidia-smi
    ;;
  down)
    docker compose down
    ;;
  *)
    echo "Uso: $0 {build|up|shell|build-workspace|gpu|down}" >&2
    exit 2
    ;;
esac
