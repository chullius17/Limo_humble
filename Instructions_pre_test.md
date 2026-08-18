# DOCKER BRING-UP

## 1. PRIMO AVVIO (Se il container non esiste ancora)
cd /home/giulio/limo_humble/ros2_humble_env
./start_limo_docker.sh

## 2. AVVII SUCCESSIVI (Se il container è stato fermato)
docker start -ai limo_dev

## 3. APRIRE NUOVI TERMINALI NEL CONTAINER GIÀ ATTIVO
docker exec -it limo_dev bash

# PERMESSI GRAFICI (Se RViz / Gazebo non si aprono)
xhost +local:docker

# PER I PERMESSI 
sudo chown -R $USER:$USER ~/limo_humble
