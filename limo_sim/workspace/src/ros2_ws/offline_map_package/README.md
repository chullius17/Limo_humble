# Mapping semantico da PointCloud2

Il launch `map.launch.py` consuma direttamente
`/limo/cv_package/visual_ptcld/points`: campi `x,y,z` FLOAT32 e
`class_id` UINT8, coordinate metriche in `base_link`, timestamp del sensore.
La pipeline precedente è disponibile in `legacy_map.launch.py`.

| class_id | Classe | Costo |
| --- | --- | --- |
| 1 | blu | ignorato, nessuna evidenza positiva o negativa |
| 2 | turchese | 60 |
| 3 | bianco | 30 |
| 4 | boardwalk | 90 |

Il nodo non usa immagini, OpenCV o proiezioni BEV. Legge il buffer della cloud
con NumPy, trasforma i punti al timestamp del sensore e aggiorna tile sparse.
Le mappe dense vengono generate a 1 Hz, solo quando cambiano i dati o le pose.
I parametri sono in `config/semantic_mapping.yaml`.

## Evidenza e correzioni

Per ciascuna cella e classe si conservano punteggi log-odds separati dai costi.
Un'osservazione incrementa la propria classe e riduce quelle incompatibili nella
stessa cella; i punteggi sono saturati per poter correggere errori con osservazioni
successive. Più punti nella stessa cella si dividono un solo aggiornamento per cloud.
Una singola osservazione discordante non cancella una classe consolidata.
La classe con evidenza maggiore, sopra soglia, determina il costo esatto: la
confidenza non moltiplica 30, 60 o 90. Parità e confidenza insufficiente danno unknown.

Questi punti rappresentano superfici classificate: non si cancellano celle lungo
raggi 2D né nell'intero campo visivo. Un punto bianco indica costo 30; non dimostra
che tutto il segmento dal robot al punto sia privo delle altre classi.
Celle mai osservate, blu e punti invalidi non generano evidenza negativa.
Una vecchia classificazione isolata resta finché non viene riosservata o spostata
da una correzione della posa; non si applica decadimento temporale indiscriminato.

## Avvio in Foxy

Nel container `limo_sim`, dopo la build e il source di `/workspace/install/setup.bash`:

```bash
ros2 launch offline_map_package map.launch.py
```

Avvia CV, lo SLAM Toolbox esistente, mapping semantico, RViz e GUI di salvataggio.
La cloud CV non include i punti blu per default; usare
`publish_blue_points:=true` per pubblicarli mantenendoli comunque disponibili
internamente alla classificazione boardwalk.
Con CV e SLAM già attivi:

```bash
ros2 launch offline_map_package map.launch.py start_cv:=false start_slam:=false
```

Per un replay/headless aggiungere `start_gui:=false start_rviz:=false`.
Usare `use_sim_time:=false` con un robot reale.
Il nodo non pubblica TF. Un TF mancante viene atteso fino a `tf_wait_sec`, poi la
cloud viene scartata; non si ripiega sulla posa più recente. Timestamp duplicati
o fuori ordine vengono ignorati. Prima di riavvolgere un bag usare `reset_map`.

Se `/map` è disponibile, le quattro uscite hanno esattamente la sua geometria
(risoluzione, dimensioni, origine e rotazione); i punti esterni a questa vista
rimangono nelle tile e ricompaiono se la mappa si espande. Il contenuto lidar non
viene sovrapposto ai costi semantici. Senza `/map` i limiti sono ricavati dai punti.
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
- `combined_grid`: classe selezionata, valori -1 / 30 / 60 / 90.

Tutti sono `nav_msgs/OccupancyGrid`, QoS reliable/transient-local. Il contenuto
è un **costo semantico**, non probabilità di occupazione fisica: eventuali
consumatori AMCL/Nav2 devono interpretarlo esplicitamente. I vecchi topic binari
`cv_map`/`street_map` non sono prodotti da questa pipeline.

Servizi disponibili:

```bash
ros2 service call /limo/map_package/offline/map_saver/save_map std_srvs/srv/Trigger '{}'
ros2 service call /limo/map_package/offline/reset_map std_srvs/srv/Trigger '{}'
```

Il salvataggio genera un nuovo `semantic_*.npz` in
`/workspace/ros2_maps/semantic` (o `save_directory`). Contiene i tre livelli,
`combined`, costi, class_id, risoluzione, origine, frame e timestamp. Preserva i
numeri esatti e unknown; non usa le soglie del vecchio map saver di Nav2.
È una fotografia numerica della mappa, non un file YAML/PGM Nav2, un `.pbstream`
o un checkpoint per riprendere l'accumulo. Salvare separatamente stato Cartographer
e bag se serve un successivo riallineamento/replay.

## Verifica e risorse

```bash
colcon build --symlink-install --packages-select offline_map_package
python3 -m pytest src/ros2_ws/offline_map_package/test/test_semantic_grid.py src/ros2_ws/offline_map_package/test/test_semantic_mapper.py
```

`max_cells` limita le tile allocate (circa 16 byte/cella più overhead);
`max_output_cells` limita la vista densa. Superato un limite viene segnalato un
errore, senza cancellare la mappa esistente. Le allocazioni temporanee durante
pubblicazione/salvataggio richiedono ulteriore memoria. L'efficienza effettiva
va misurata sulla Jetson Nano con il flusso camera reale.
