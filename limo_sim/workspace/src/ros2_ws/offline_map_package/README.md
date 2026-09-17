# Mapping semantico da PointCloud2

Il launch `map.launch.py` consuma direttamente
`/limo/cv_package/visual_ptcld/points`: campi `x,y,z` FLOAT32 e
`class_id` UINT8, coordinate metriche in `base_link`, timestamp del sensore.
Consuma inoltre `/map` di SLAM Toolbox come layer laser completo: ogni messaggio
sostituisce il precedente, incluse le celle libere e sconosciute. Il mapper non
accumula endpoint da `/scan`; lo scan viene usato direttamente da SLAM Toolbox.

| class_id | Classe | Costo |
| --- | --- | --- |
| 1 | exterior road | 0 |
| 2 | yellow lines | 60 |
| 3 | soft obstacle | 30 |
| 4 | boardwalk | 90 |
| 5 | interior road | 0 |
| 6 | interior boardwalk | 90 |

`visual_ptcld` separa `interior road` come
`road_originale & ~exterior_road`, prima di applicare la stessa ROI. Il bordo
è l'intersezione della strada con la dilatazione di `soft obstacle`.
I punti `interior road` vengono voxelizzati nell'immagine con
`point_voxel_size` (5x5 pixel nel launch), proiettati nella BEV con profondità
valida e mantenuti fuori dalle query KD-tree del boardwalk. Dopo la
classificazione, tutte le classi vengono voxelizzate separatamente nella griglia
metrica da `pointcloud_voxel_size_m` (2 cm nel launch) e pubblicate nella cloud.
Nel primo passaggio KD-tree i punti `soft obstacle` oltre
`blue_radius_max_m` diventano
`interior boardwalk` (`class_id=6`). Il mapper conserva l'ID distinto in ingresso
e lo accumula nello stesso layer e con lo stesso costo del boardwalk.

Il nodo non usa immagini, OpenCV o proiezioni BEV. Legge il buffer della cloud
con NumPy, trasforma i punti al timestamp del sensore e aggiorna tile sparse.
Le mappe dense vengono generate fino a 4 Hz quando cambiano dati, pose o mappa laser.
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
Exterior road e interior road alimentano un unico punteggio di strada nel mapper,
con un solo aggiornamento normalizzato per cella e cloud. Usano gli stessi
incrementi e decrementi delle altre classi:
osservazioni ripetute di strada riducono l'evidenza degli ostacoli nella stessa
cella. Quando prevale la strada, la combinata e tutti e tre i layer diventano 0.
Un ostacolo osservato successivamente può riprendere il sopravvento.

Questi punti rappresentano superfici classificate: non si cancellano celle lungo
raggi 2D né nell'intero campo visivo. Un punto soft obstacle indica costo 30;
non dimostra
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
Il profilo simulato include anche la computer vision; la cloud di `visual_ptcld`
include i punti exterior road con `class_id=1`.
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

Le uscite hanno esattamente la geometria dell'ultima `/map` valida
(risoluzione, dimensioni, origine e rotazione); i punti CV esterni a questa vista
rimangono nelle tile e ricompaiono se la mappa si espande. `/map` fornisce anche
l'intero layer laser. Aggiornamenti a geometria invariata sostituiscono comunque
il contenuto: gli ostacoli cancellati da SLAM spariscono dalla combinata e
dal successivo salvataggio. I messaggi non validi non sostituiscono la mappa.
Prima della prima `/map`, il nodo accumula la CV ma attende per pubblicare e salvare.
`reset_map` cancella solo l'evidenza CV; il layer laser resta quello di SLAM.
Scegliere `resolution` vicina a quella di `/map` per limitare il ricampionamento.

## Cartographer e loop closure

`pose_source:=tf` accumula nel frame `map` e corregge le classi sulle celle
riosservate. Non corregge retroattivamente la geometria storica della CV dopo
loop closure. Il layer laser segue invece la mappa aggiornata da SLAM.
Applicare un solo `map -> odom` alla storia non risolve le correzioni diverse dei
singoli tratti della traiettoria.

Con Cartographer esterno e `cartographer_ros_msgs` installato:

```bash
ros2 launch offline_map_package map.launch.py start_slam:=false pose_source:=cartographer
```

Cartographer deve fornire `/submap_list`, TF e `/map` dal proprio
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

- `turquoise_map` (yellow lines), `white_map` (soft obstacle) e
  `boardwalk_map`: costo della classe selezionata,
  0 nelle celle classificate diversamente, -1 nelle celle sconosciute/incerte.
- `combined_grid`: mappa live fusa laser + CV; gli ostacoli laser a 100 hanno
  precedenza, poi i costi semantici osservati. Le altre celle conservano il
  valore laser, incluso -1 quando sconosciuto.

Tutti sono `nav_msgs/OccupancyGrid`, QoS reliable/transient-local. Il contenuto
è un **costo semantico**, non probabilità di occupazione fisica: eventuali
consumatori AMCL/Nav2 devono interpretarlo esplicitamente. I vecchi topic binari
`cv_map`/`street_map` non sono prodotti da questa pipeline.

Servizi disponibili:

```bash
ros2 service call /limo/map_package/offline/map_saver/save_map std_srvs/srv/Trigger '{}'
ros2 service call /limo/map_package/offline/reset_map std_srvs/srv/Trigger '{}'
```

Il salvataggio richiede una mappa laser valida ricevuta su
`reference_map_topic` (default `/map`) e genera tre coppie in
`/workspace/ros2_maps/semantic` (o `save_directory`):

- `limo_map_laser.pgm/.yaml`: copia del solo layer laser SLAM in modalità
  `raw`, con gli stessi valori di occupazione e celle sconosciute.
- `limo_map_cv_obstacle.pgm/.yaml`: mappa binaria `trinary` ottenuta dal
  layer CV (dopo il filtro di salvataggio). I costi 0..39 sono liberi, 40..95
  occupati, gli altri sconosciuti. Le celle CV non osservate (-1) diventano
  libere solo se il laser vale esattamente 0; con laser sconosciuto o positivo
  restano sconosciute. I valori CV osservati non vengono sovrascritti dal laser.
  Questo riempimento viene ricalcolato a ogni salvataggio e non modifica
  l'evidenza CV accumulata.
- `limo_map_complete.pgm/.yaml`: fusione dei layer in modalità `raw`.
  La precedenza è laser a 100, classe CV osservata, quindi stato del laser.
  Una cella sconosciuta in entrambi resta sconosciuta.

Il nome base `limo_map` si cambia con `save_map_name`. In modalità `raw`
il PGM contiene direttamente valori 0..100 e 255 per unknown: Nav2 ricostruisce
esattamente i costi e gli sconosciuti. La resa del PGM in un visualizzatore di
immagini generico è diversa dalla scala di grigi tradizionale; usare il YAML
con il map server per visualizzarlo come mappa.
I nomi dei file YAML restano invariati e i launch online li caricano normalmente.
Le vecchie mappe `scale` continuano a essere caricabili; la nuova modalità si
applica ai salvataggi successivi.

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
