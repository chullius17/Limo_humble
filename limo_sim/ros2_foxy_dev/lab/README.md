# Configurazione ROS 2 Foxy per `lab`

Profilo dedicato al PC `bi-fdd-f-l001` (`ssh lab`): Ubuntu 24.04 x86_64,
RTX 5060 8 GB e ROS 2 Foxy isolato in Ubuntu 20.04.

## Prima installazione sull'host

```bash
cd ~/limo_foxy/limo_sim/ros2_foxy_dev/lab
./setup-host.sh
```

Al termine, disconnettersi e riconnettersi per applicare il gruppo `docker`.

## Uso

```bash
./dev.sh build            # crea l'immagine
./dev.sh up               # avvia il container persistente
./dev.sh build-workspace  # rosdep + colcon build del workspace montato
./dev.sh shell            # apre una shell ROS
./dev.sh gpu              # verifica la GPU dal container
./dev.sh down             # arresta/rimuove il container
```

Il workspace `../../workspace` è montato in `/workspace`. Il container usa la
rete e l'IPC dell'host per ROS 2, la GPU NVIDIA, X11 e i dispositivi USB/seriali.
La modalità `privileged` è intenzionale per lo sviluppo con lidar e camera;
non usare questa configurazione per workload non fidati.

Per GUI avviate da una sessione SSH, connettersi con forwarding X11 (`ssh -X lab`)
oppure eseguire `dev.sh` dal desktop del PC. Con il solo `ssh lab`, `DISPLAY`
potrebbe non essere disponibile.
