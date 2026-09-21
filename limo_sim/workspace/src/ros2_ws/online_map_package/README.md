# Localizzazione AMCL con laser e pointcloud semantica

I profili `config/mapping_sim.yaml` e `config/mapping_real.yaml` configurano
avvio, map server e AMCL. `online_map.launch.py` contiene la logica condivisa e
usa la pipeline CV ottimizzata, il planner di traiettoria e le tre mappe esportate
in `ros2_maps/semantic` dal mapper offline:

| File | Topic | Uso |
| --- | --- | --- |
| `limo_map_laser.yaml` | `/limo/map_package/online/maps/laser_map` | confronto laser AMCL |
| `limo_map_cv_obstacle.yaml` | `/limo/map_package/online/maps/cv_obstacle` | confronto CV AMCL |
| `limo_map_complete.yaml` | `/map` | sorgente completa per planner e costmap globale |

`trajectory.launch.py` riceve `/map` dal profilo online. Il relativo
`BorderFollowLayer` costruisce `/global_costmap/costmap`, visualizzata in RViz
come **Global Costmap (Inflation)** con lo schema colori `costmap` e alpha 0.35,
come nel ramo `humble-navigation`.

Su Foxy `always_send_full_costmap: true` evita un bug di RViz negli aggiornamenti
`OccupancyGridUpdate`: la prima riga viene ripetuta su tutte le righe, facendo
comparire strisce dopo la prima mappa corretta. La costmap completa viene
pubblicata a 1 Hz; la palette non causa questo problema.

La cloud `/limo/cv_package/visual_ptcld/points` contiene `x,y,z` FLOAT32 e
`class_id` UINT8. AMCL seleziona **yellow lines=2, boardwalk=4, interior
boardwalk=6** come ostacoli e **exterior road=1, interior road=5** come strada.
**Soft obstacle=3**, unknown e le altre classi non votano. Nessuna immagine
o griglia BEV locale entra in AMCL.

Per ogni aggiornamento laser:

1. AMCL aggiorna le particelle con il modello laser sulla sola mappa laser.
2. Sceglie la cloud col timestamp più vicino, entro `cv_sync_tolerance`.
3. Trasforma la cloud nel frame base al timestamp laser usando `odom` come
   frame fisso, senza dipendere dalla posa globale stimata da AMCL.
4. Dopo il filtro delle classi, aggrega in voxel **XY** da `cv_voxel_size`
   (default 0.075 m), separando strada e ostacoli. Usa il centroide dei punti
   di ciascun gruppo nel voxel, con peso 1.
   La riduzione da 2 cm in `visual_ptcld` resta il primo stadio. Non si
   moltiplica il peso per il numero di punti o classi della stessa polarità.
   Se strada e ostacolo condividono un voxel, mantengono due voti distinti.
5. Trasforma questo insieme con **ogni posa candidata** e calcola la frazione
   di voti non corrispondenti: ostacoli su celle occupate, strada su celle
   esattamente a costo 0 della stessa mappa CV. Il mismatch è normalizzato
   sul totale dei voti strada + ostacoli, senza pesi aggiuntivi tra i due gruppi.
6. Applica la formula del riferimento Humble e normalizza prima del resampling:
   `w_final ∝ w_laser^laser_weight_factor × exp(-cv_weight_factor × cv_sad_gain × mismatch)`.

Il mismatch conserva la regola SAD positiva di Humble: libero, sconosciuto e
fuori mappa danno disaccordo rispetto a un ostacolo osservato. La nuova componente
"negative SAD" usa solo le classi strada esplicitamente osservate: qualsiasi
cella diversa da 0, sconosciuto e fuori mappa danno disaccordo. L'assenza di punti
non prova spazio libero. Non viene usata una mappa street separata e i soft
obstacles non partecipano al confronto. La componente negativa estende il
modello positivo del riferimento Humble disponibile nel repository.

Cloud assente, troppo distante nel tempo, malformata, TF indisponibile o meno di
`cv_min_points` voxel (default 5) impediscono l'aggiornamento CV.
Come nel launch online Humble, `cv_sync_tolerance` vale 0.20 s: la stessa cloud
può essere riutilizzata entro questo limite **solo se il lidar non contiene
ritorni validi oppure `laser_weight_factor` è zero**. Un ritorno valido è una
distanza finita strettamente interna ai limiti utili del sensore e configurati.
Con lidar valido e peso positivo ogni frame CV viene usato una sola volta.
Il limite temporale parte sempre dal timestamp originale della cloud; il
riutilizzo non lo rinnova. Ogni riutilizzo ricompensa il movimento tramite TF
odom e conserva il guadagno CV configurato, come nel riferimento Humble.

I profili attuali usano `laser_weight_factor: 1.0` e `cv_weight_factor: 1.0`.
Il peso laser zero salta il modello laser anche quando manca la CV. Il topic
`/scan` resta necessario per scandire gli aggiornamenti e valutarne la validità:
se smette completamente di arrivare, questo meccanismo non avvia aggiornamenti
autonomi. `cv_enabled:=false` o `cv_weight_factor:=0.0` disabilitano la fusione CV.
I parametri CV si leggono alla configurazione del nodo: per cambiarli riavviare.

Prima della fusione, `cv_quality_gate_enabled: true` valuta la probabilità
prodotta dalla sola CV sulle pose delle particelle correnti. Un confronto quasi
uniforme viene scartato (`cv_min_information: 0.02` nats di divergenza KL dalla
distribuzione uniforme). Si scartano anche ipotesi CV con deviazione standard
superiore a `cv_max_position_stddev: 0.5` m lungo l'asse XY più disperso oppure
`cv_max_yaw_stddev: 0.5` rad per l'orientamento circolare. Le strade restano attive.
Il rifiuto avviene prima di modificare qualsiasi peso: con laser attivo viene
conservato il suo aggiornamento. Il log `CV update rejected` mostra motivo e
misure. Questi valori descrivono l'ambiguità CV sulle particelle disponibili,
non la covarianza del sensore; le soglie iniziali richiedono verifica sul circuito.
Una distribuzione globale molto dispersa può restare esclusa dalla fusione CV
finché il lidar o una posa iniziale non restringono le ipotesi.

## Avvio

Dopo la build di `nav2_amcl`, `limo_inflation`, `traj_package`, `limo_rviz`,
`cv_package`, `online_map_package` e il source del workspace:

```bash
cd /workspace
colcon build --packages-select nav2_amcl limo_inflation traj_package \
  limo_rviz cv_package online_map_package --symlink-install
source install/setup.bash
ros2 launch online_map_package online_map_sim.launch.py
```

Con sensori, odometria e CV già attivi, sulla LIMO:

```bash
ros2 launch online_map_package online_map_real.launch.py
```

Sul PC, per aprire solo RViz collegato ai topic della LIMO:

```bash
ros2 launch online_map_package desktop_online.launch.py
```

Per override temporanei di mappe o voxel in simulazione:

```bash
ros2 launch online_map_package online_map_sim.launch.py \
  map_directory:=/workspace/ros2_maps/semantic map_name:=limo_map \
  cv_voxel_size:=0.10 cv_min_points:=5.0
```

I valori persistenti si modificano nei due YAML; gli argomenti della riga di
comando servono per prove temporanee. Il profilo reale non riavvia la CV e non
apre finestre. `desktop_online.launch.py` usa lo stesso profilo reale in modalità
desktop e avvia esclusivamente RViz. Il planner e la costmap vengono avviati
dal profilo online sul backend; si possono disabilitare temporaneamente con
`start_trajectory:=false`.

AMCL pubblica `map -> odom`; fornire una posa iniziale tramite RViz oppure il
servizio AMCL di localizzazione globale. Non avviare contemporaneamente SLAM
che pubblichi lo stesso TF. Il launch avvia localizzazione, map server, CV,
planner di traiettoria e RViz opzionali: i vecchi nodi `online_metric_bev`,
`cv_2_ptcld`, `cv_amcl_debug`,
`online_map` e `local_ptcld`, basati sulle vecchie griglie, non vengono avviati.
I relativi sorgenti restano disponibili, ma non sono stati convertiti in questa
modifica alla localizzazione.

`local_map_final` viene invece avviato dal profilo online e pubblica su
`/limo/map_package/online/local_map_final/markers` i limiti di lavoro in
`base_link`: il rettangolo persistente verde da 2,50 x 2,66 m e il trapezio ROI
giallo alto 1,95 m, largo da 0,60 a 2,66 m. La base maggiore del trapezio ha
sempre la stessa larghezza del rettangolo e coincide con il suo lato anteriore,
a 2,50 m da `base_link`. Il display **Local Map Regions** è già abilitato in
`online_map.rviz`. In modalità desktop il nodo resta sul backend e RViz
visualizza il topic ricevuto dalla LIMO.

Un terzo contorno rosso mostra il trapezio interno: il lato vicino al robot e
i due lati obliqui sono arretrati di 20 cm verso l'interno, misurati
perpendicolarmente al lato (`inner_trapezoid_inset: 0.20`). La base larga rossa
rimane invece allineata al bordo anteriore del trapezio giallo e del rettangolo
verde. È un riferimento visivo in `base_link`; la memoria dei punti usa il
trapezio giallo.
Vertici e messaggi dei tre contorni sono precalcolati una sola volta durante
l'inizializzazione; la pubblicazione aggiorna soltanto i timestamp.

`local_map_final` fonde la cloud CV corrente e la memoria dei punti riproiettati
e pubblica direttamente `/limo/map_package/online/local_costmap` come
`nav_msgs/OccupancyGrid`. La griglia coincide con il rettangolo verde: misura
2,50 x 2,66 m, ha origine `(0, -1.33)` in `base_link` e, con risoluzione 2 cm,
contiene esattamente 125 x 133 celle. Le celle libere valgono 0; yellow line
(classe 2) vale 60 e boardwalk (classe 4) vale 90, come nella costruzione della
mappa semantica offline. Se più punti cadono nella stessa cella viene conservato
il costo maggiore. I costi sono fissi: la confidenza regola la persistenza dei
punti, ma non riduce il loro costo.

I punti della memoria, riproiettati nel frame corrente e già sottoposti al cap
di 300 elementi, sono pubblicati a 10 Hz anche come `sensor_msgs/PointCloud2` su
`/limo/map_package/online/local_map_final/points`. La cloud è in `base_link`,
contiene i campi `x`, `y`, `z` e `class_id`. Il display RViz
**Local Reprojected Semantic Points** è abilitato e colora le classi 2 e 4
tramite `class_id`; la cloud CV live resta visibile nel display separato.

A ogni frame CV i punti delle due classi vengono conservati come sorgente live
per 0,50 s. In parallelo, i punti situati **dentro il trapezio giallo ma fuori
da quello rosso** vengono convertiti subito in coordinate `odom` usando la TF
dello stesso timestamp e inseriti nella memoria. Non si attende che escano dal
trapezio: la fascia fra i due contorni è la zona di ammissione. Un frame senza
TF viene ignorato senza alterare né la sorgente live né la memoria esistente.

I punti persistenti sono riproiettati a 10 Hz anche senza nuovi frame CV.
Sono cancellati quando entrano nel trapezio rosso, escono dal rettangolo oppure
scendono sotto `minimum_confidence` (0,30). Alla nascita la confidenza vale 1:
il decadimento viene integrato a ogni riproiezione in base alla regione e al
comando `/cmd_vel` attuali. Come in `local_ptcld`, il maggiore fra i rapporti
di velocità lineare e angolare scala il decadimento da zero a uno: sotto le
soglie di quiete (0,01 m/s e 0,02 rad/s) la memoria non decade; raggiunge il
tasso massimo rispettivamente a 0,50 m/s o 1,00 rad/s. Nel rettangolo verde,
fuori dal giallo, il massimo è `confidence_decay_per_sec` (0,10/s). Dentro il
giallo e fuori dal rosso viene inoltre applicato `yellow_decay_multiplier: 3.0`.
Dentro il rosso il punto viene cancellato immediatamente, indipendentemente
dalla velocità. Con soglia 0,30 e velocità al rapporto massimo, un punto mai
riosservato dura circa 12 secondi nella regione rossa e 4 secondi in quella
gialla; da fermo non scade. `cmd_vel_timeout_sec: 0.0` conserva l'ultimo
comando senza timeout, come il vecchio nodo. Impostare un valore positivo
arresta il decadimento quando il comando diventa obsoleto. Questa confidenza
descrive la memoria, non la certezza del classificatore CV.
Un filtro voxel da 3 cm separato per classe evita duplicati. Il limite
`maximum_points: 300` è condiviso dalle due classi: quando viene superato sono
rimossi prima i punti a confidenza più bassa; a parità di confidenza i punti
rimasti sono distribuiti spazialmente. Parametri nella sezione
`local_map_final` dei profili; override
del limite dal launch: `local_map_maximum_points:=300` (riavviare per applicare).
RViz mostra la griglia risultante nel display **Local Semantic Costmap**, oltre
alla cloud CV originale e ai tre contorni. Il launch del controller non avvia
più un convertitore separato: questa griglia è già pronta per il controller.

I log `CV cloud fusion` mostrano differenza temporale, punti in ingresso,
voxel, particelle ed effettivo numero di confronti voxel × particelle.

## Verifica

`nav2_amcl/test/test_cv_cloud.cpp` verifica selezione e fusione delle classi,
voxelizzazione, layout/endian, trasformazioni e pesi. `test_cv_sync.cpp` verifica
la compensazione temporale e il comportamento senza dati utilizzabili; i test Python in
`test/test_localization_launch.py` verificano mappe, topic e parametri del launch.
Il tuning e la convergenza vanno poi valutati con sensori reali o rosbag.
