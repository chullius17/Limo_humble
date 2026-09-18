#!/usr/bin/env bash
set -euo pipefail

if [[ "$(. /etc/os-release; printf '%s' "${ID}")" != "ubuntu" ]]; then
  echo "Questo script supporta solo Ubuntu." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Driver NVIDIA non rilevato: installarlo e riavviare prima di continuare." >&2
  exit 1
fi

# Repair permissions left by an interrupted/older toolkit setup before apt
# reads an already configured NVIDIA repository.
if [[ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]]; then
  sudo chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Docker Engine official apt repository.
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
printf '%s\n' \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "${USER}"

# NVIDIA Container Toolkit official apt repository.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo docker run --rm hello-world
sudo docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

echo
echo "Host pronto. Disconnettersi e riconnettersi per usare Docker senza sudo."
