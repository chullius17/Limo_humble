# GUI sul computer, nodi sul robot

Il container del PC `limo_sim` usa ROS Foxy, rete host e il display X del PC.
Nel setup verificato monta `~/limo_foxy/limo_sim/workspace` in `/workspace`:
è una copia distinta da questo checkout. Il robot e il container si vedono
nel dominio ROS 0. Il comando seguente copia il launch principale e il profilo
YAML in `/tmp` del container, senza sostituire il workspace installato.

Dal terminale grafico del **PC**, nella directory `workspace/src` di questo
checkout (il profilo predefinito è `mapping_real.yaml`):

```bash
bash scripts/limo_gui.sh
```

Apre RViz e il pannello **LIMO Map Saver** sul PC. Non avvia sensori, SLAM,
mapper o publisher TF. Usa il tempo reale (`use_sim_time:=false`). Il pulsante
Save Map chiama il servizio del mapper: i file vengono salvati sulla macchina
dove gira quel nodo, normalmente il robot.

Lo script esegue `map.launch.py` con `mode:=desktop`, copiando temporaneamente
il launch principale e il YAML selezionato nel container. Per la simulazione:
`bash scripts/limo_gui.sh --profile sim`.

Per guardare lidar e TF senza una mappa ancora disponibile:

```bash
bash scripts/limo_gui.sh fixed_frame:=odom start_gui:=false
```

Per altre GUI ROS installate nel container:

```bash
bash scripts/limo_gui.sh --exec ros2 run rqt_gui rqt_gui
bash scripts/limo_gui.sh --exec ros2 run rqt_image_view rqt_image_view
```

Chiudere con Ctrl-C nel terminale che ha avviato il comando. Non lanciare
più copie della stessa GUI se non necessario.

Sul **robot**, avviare la mappatura con le finestre disabilitate:

```bash
ros2 launch offline_map_package map_real.launch.py
```

Il profilo reale disabilita le GUI sul robot e usa `base_link` con il tempo reale.
Prima di usare i nuovi launch, compilare il pacchetto aggiornato e fare source:

```bash
colcon build --symlink-install --packages-select offline_map_package
source install/setup.bash
```

Con il pacchetto aggiornato anche nel container, le GUI si possono avviare
direttamente con `ros2 launch offline_map_package map_real.launch.py mode:=desktop`.
Per l'avvio hardware usare `custom_start limo_real.launch.py open_rviz:=false`.

Le finestre non vengono trasferite automaticamente da una normale sessione
`ssh limo`: un'applicazione eseguita sul robot usa il display di quella
sessione. Avviare le GUI dal PC con questo script e i nodi di elaborazione
sul robot. Le immagini e le mappe passano attraverso i topic ROS.

Se RViz non trova `map`, verificare che SLAM sia attivo; per il solo lidar
usare `fixed_frame:=odom`. Se i topic del robot non compaiono, verificare
connessione di rete e `ROS_DOMAIN_ID` uguale su entrambe le macchine.
Se il display X rifiuta l'accesso, verificare le autorizzazioni grafiche del
container; lo script non disabilita il controllo degli accessi X.
