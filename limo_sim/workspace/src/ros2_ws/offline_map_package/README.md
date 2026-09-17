# Mapping semantico da PointCloud2

Il launch `map.launch.py` consuma direttamente
`/limo/cv_package/visual_ptcld/points`: campi `x,y,z` FLOAT32 e
`class_id` UINT8, coordinate metriche in `base_link`, timestamp del sensore.
Consuma inoltre `/scan`: ogni endpoint lidar valido viene trasformato nel frame
`map`, accumulato e salvato con costo 100.

| class_id | Classe | Costo |
| --- | --- | --- |
| 1 | blu di bordo / strada osservata | 0 |
| 2 | turchese | 60 |
| 3 | bianco | 30 |
| 4 | boardwalk | 90 |
| 5 | blu interno / strada osservata | 0 |

`visual_ptcld` separa i blu interni come `blu_originale & ~blu_di_bordo`,
prima di applicare la stessa ROI. Il bordo è l'intersezione del blu con la
dilatazione del bianco. I blu interni vengono voxelizzati nell'immagine con
`point_voxel_size` (5x5 pixel nel launch), proiettati nella BEV con profondità
valida e mantenuti fuori dalle query KD-tree del boardwalk. Dopo la
classificazione, tutte le classi vengono voxelizzate separatamente nella griglia
metrica da `pointcloud_voxel_size_m` (2 cm nel launch) e pubblicate nella cloud.

Il nodo non usa immagini, OpenCV o proiezioni BEV. Legge il buffer della cloud
con NumPy, trasforma i punti al timestamp del sensore e aggiorna tile sparse.
Le mappe dense vengono generate a 1 Hz, solo quando cambiano i dati o le pose.
I parametri sono nei due profili `config/mapping_sim.yaml` e
`config/mapping_real.yaml`. Ciascun file contiene le sezioni `launch`,
`slam_toolbox`, `semantic_mapper` e `map_save_gui`. Sono profili letti dal
launch principale, non file da passare direttamente a ROS con `--params-file`.

## Evidenza e correzioni

Per ciascuna cella e classe si conservano punteggi log-odds separati dai costi.
Un'osservazione incrementa la propria classe e riduce quelle incompatibili nella
stessa cella; i punteggi sono saturati per poter correggere errori con osservazioni
successive. Più punti nella stessa cella si dividono un solo aggiornamento per cloud.
Una singola osservazione discordante non cancella una classe consolidata.
La classe con evidenza maggiore, sopra soglia, determina il costo esatto: la
confidenza non moltiplica 0, 30, 60 o 90. Parità e confidenza insufficiente danno unknown.
Blu di bordo e blu interno alimentano un unico punteggio di strada nel mapper,
con un solo aggiornamento normalizzato per cella e cloud. Usano gli stessi
incrementi e decrementi delle altre classi:
osservazioni ripetute di strada riducono l'evidenza degli ostacoli nella stessa
cella. Quando prevale la strada, la combinata e tutti e tre i layer diventano 0.
Un ostacolo osservato successivamente può riprendere il sopravvento.

Questi punti rappresentano superfici classificate: non si cancellano celle lungo
raggi 2D né nell'intero campo visivo. Un punto bianco indica costo 30; non dimostra
che tutto il segmento dal robot al punto sia privo delle altre classi.
Celle mai osservate e punti invalidi non generano evidenza negativa.
Una vecchia classificazione isolata resta finché non viene riosservata o spostata
da una correzione della posa; non si applica decadimento temporale indiscriminato.

## Avvio in Foxy

Nel container `limo_sim`, dopo la build e il source di `/workspace/install/setup.bash`:

```bash
ros2 launch offline_map_package map_sim.launch.py
```

Il wrapper seleziona `mapping_sim.yaml` e include il launch principale
`map.launch.py`: avvia SLAM Toolbox, mapping semantico, RViz e GUI di salvataggio
con il tempo simulato. I valori di tuning sono quelli del precedente launch.
La computer vision deve essere già attiva; la cloud pubblicata da `visual_ptcld`
include sempre i punti blu con `class_id=1`.
Con lo SLAM già attivo:

```bash
ros2 launch offline_map_package map.launch.py start_slam:=false
```

Per un replay/headless aggiungere `mode:=backend`.

Sul robot reale (sensori, EKF e computer vision già attivi):

```bash
ros2 launch offline_map_package map_real.launch.py
```

Questo launch usa `mapping_real.yaml` e forza `mode:=backend`: avvia SLAM e
mapper con tempo reale, `base_link` e nessuna finestra sul robot. Il tuning
semantico/SLAM resta quello esistente; i due YAML permettono di modificarlo
indipendentemente dopo le prove sul robot.

Sul PC, nel container Foxy con i pacchetti aggiornati, le sole interfacce:

```bash
ros2 launch offline_map_package desktop_offline.launch.py
```

`desktop_offline.launch.py` apre solo RViz e Save Map, collegati ai topic e al servizio
del robot; non avvia SLAM o mapper. I file salvati restano sulla LIMO.
Il container deve avere accesso al display del PC e alla rete del robot
(nel setup attuale `limo_sim` usa la rete host). Usare lo stesso `ROS_DOMAIN_ID`
su entrambe le macchine, normalmente `0`, e `ROS_LOCALHOST_ONLY=0`.

Prima del primo avvio, aggiornare questo pacchetto nel workspace di entrambe
le macchine, poi dalla radice di ciascun workspace eseguire:

```bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select offline_map_package
source install/setup.bash
```

Il workspace montato nel Docker del PC può essere una copia diversa da questo
checkout: verificare che contenga i launch aggiornati. Per fermare la mappatura
e le finestre, premere `Ctrl-C` nei rispettivi terminali.

`map.launch.py` senza argomenti mantiene il profilo simulato. Per un file proprio:

```bash
ros2 launch offline_map_package map.launch.py config_file:=/percorso/mapping.yaml
```

Gli override CLI `start_slam`, `start_mapper`, `start_rviz`, `start_gui`,
`use_sim_time`, `rviz_config`, `fixed_frame`, `pose_source`, `trajectory_id`,
`resolution` e `save_directory` restano disponibili. Il valore vuoto usa il YAML;
`resolution` sovrascrive sia SLAM sia mapper, `use_sim_time` tutti i nodi.
`mode:=backend` forza le GUI spente; `mode:=desktop` forza SLAM/mapper spenti
e permette di disabilitare singole finestre con `start_rviz`/`start_gui`.
I percorsi RViz relativi si riferiscono a `limo_rviz/config`.

SLAM viene avviato direttamente con i parametri del profilo, quindi non serve
più il workaround Foxy `params_file:=...` e non si usano i default con
`base_footprint`. I due ingressi per il robot reale sono `map_real.launch.py`
sulla LIMO e `desktop_offline.launch.py` nel Docker del PC; `map.launch.py` contiene la
logica condivisa e `map_sim.launch.py` resta l'ingresso per la simulazione.

Il nodo non pubblica TF. Un TF mancante viene atteso fino a `tf_wait_sec`, poi la
cloud viene scartata; non si ripiega sulla posa più recente. Timestamp duplicati
o fuori ordine vengono ignorati. Prima di riavvolgere un bag usare `reset_map`.
Al salvataggio, `save_median_kernel: 3` rimuove le celle boardwalk nere isolate
con una mediana 3x3; la mappa pubblicata live non viene modificata.

Se `/map` è disponibile, le quattro uscite hanno esattamente la sua geometria
(risoluzione, dimensioni, origine e rotazione); i punti esterni a questa vista
rimangono nelle tile e ricompaiono se la mappa si espande. `/map` fornisce solo
la geometria; i costi laser provengono dagli endpoint accumulati da `/scan`.
Senza `/map`, il salvataggio calcola i limiti dall'unione delle osservazioni
laser e semantiche.
Scegliere `resolution` vicina a quella di `/map` per limitare il ricampionamento.

## Cartographer e loop closure

`pose_source:=tf` accumula nel frame `map` e corregge le classi sulle celle
riosservate. Non corregge retroattivamente la geometria storica dopo loop closure.
Applicare un solo `map -> odom` alla storia non risolve le correzioni diverse dei
singoli tratti della traiettoria.

Con Cartographer esterno e `cartographer_ros_msgs` installato:

```bash
ros2 launch offline_map_package map.launch.py start_slam:=false pose_source:=cartographer
```

Cartographer deve fornire `/submap_list`, TF e preferibilmente `/map` dal proprio
occupancy-grid node. Configurare `trajectory_id`, `map_frame`, `odom_frame` e
`submap_topic` secondo il suo setup; `odom_frame` deve essere continuo.
Il messaggio ufficiale pubblica pose e identificativi delle submap:
[API Cartographer](https://google-cartographer-ros.readthedocs.io/en/latest/ros_api.html).

Il mapper associa le osservazioni alla submap più recente non congelata della
traiettoria selezionata e conserva evidenza nelle sue coordinate locali. Risolve
la posa della cloud e la posa della submap allo stesso istante del frame globale
usando `lookup_transform_full` e il frame continuo. Quando arrivano pose di
submap aggiornate, rigenera la vista globale dalle tile locali: le posizioni
precedenti non restano impresse nella mappa. Submap rimosse dalla lista non
contribuiscono alla vista. Nelle sovrapposizioni vince la cella osservata più di
recente, per permettere alle rivisite di correggere evidenze più vecchie.

Questo è un livello semantico agganciato alle submap: non modifica lo scan
matching di Cartographer e non usa i costi come misure lidar. L'associazione
esterna e il ricampionamento ai centri cella sono approssimazioni; deformazioni
interne a una submap richiedono il replay delle osservazioni originali con pose
ottimizzate. Per un export finale attendere le pose dopo l'ottimizzazione finale.
Il percorso Cartographer richiede una verifica end-to-end su bag/SLAM reale;
i test inclusi controllano il riposizionamento tramite pose sintetiche.

## Topic e salvataggio

Prefisso: `/limo/map_package/offline/map/`.

- `turquoise_map`, `white_map`, `boardwalk_map`: costo della classe selezionata,
  0 nelle celle classificate diversamente, -1 nelle celle sconosciute/incerte.
- `combined_grid`: mappa live fusa laser + CV, valori 0 / 30 / 60 / 90 / 100;
  gli ostacoli laser hanno precedenza a 100, i costi semantici CV sono
  sovrapposti allo spazio libero laser.

Tutti sono `nav_msgs/OccupancyGrid`, QoS reliable/transient-local. Il contenuto
è un **costo semantico**, non probabilità di occupazione fisica: eventuali
consumatori AMCL/Nav2 devono interpretarlo esplicitamente. I vecchi topic binari
`cv_map`/`street_map` non sono prodotti da questa pipeline.

Servizi disponibili:

```bash
ros2 service call /limo/map_package/offline/map_saver/save_map std_srvs/srv/Trigger '{}'
ros2 service call /limo/map_package/offline/reset_map std_srvs/srv/Trigger '{}'
```

Il salvataggio richiede almeno un endpoint valido ricevuto su `/scan` e genera
tre coppie in
`/workspace/ros2_maps/semantic` (o `save_directory`):

- `limo_map_laser.pgm/.yaml`: mappa `trinary`, con celle lidar occupate a 100
  e spazio conosciuto libero a 0;
- `limo_map_complete.pgm/.yaml`: costi semantici 0/30/60/90 con le celle lidar
  occupate sovrapposte a 100, salvata in modalità `scale`;
- `limo_map_cv_obstacle.pgm/.yaml`: mappa `trinary` derivata dalla completa;
  i costi da 10 a 95 inclusi sono occupati, quelli da 0 a 9 sono liberi e i
  costi da 96 a 100 sono sconosciuti. Le celle lidar a 100 risultano quindi
  sconosciute in questa mappa.

Il nome base `limo_map` si cambia con `save_map_name`. La mappa completa usa le
soglie `free_thresh: 0.0` e `occupied_thresh: 1.0`, così Nav2 ricostruisce
esattamente i costi intermedi in modalità `scale`. Nella mappa completa la
precedenza è: laser 100, classe semantica osservata, quindi stato libero del laser.
L'orientamento delle righe segue la convenzione del map saver Nav2 e ciascuna
coppia può essere caricata da `nav2_map_server`. Un salvataggio successivo
sostituisce tutte e tre le coppie precedenti.
Non viene più prodotto alcun `.npz`.

Questa è una fotografia della mappa combinata, non un `.pbstream` o un checkpoint
per riprendere l'accumulo. Salvare separatamente stato Cartographer e bag se serve
un successivo riallineamento/replay.

## Verifica e risorse

```bash
colcon build --symlink-install --packages-select offline_map_package
python3 -m pytest src/ros2_ws/offline_map_package/test/test_semantic_grid.py src/ros2_ws/offline_map_package/test/test_semantic_mapper.py
python3 -m pytest src/ros2_ws/offline_map_package/test/test_mapping_launch.py
```

`max_cells` limita le tile allocate (circa 20 byte/cella più overhead);
`max_output_cells` limita la vista densa. Superato un limite viene segnalato un
errore, senza cancellare la mappa esistente. Le allocazioni temporanee durante
pubblicazione/salvataggio richiedono ulteriore memoria. L'efficienza effettiva
va misurata sulla Jetson Nano con il flusso camera reale.
