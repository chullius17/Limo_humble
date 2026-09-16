# Localizzazione AMCL con laser e pointcloud semantica

I profili `config/mapping_sim.yaml` e `config/mapping_real.yaml` configurano
avvio, map server e AMCL. `online_map.launch.py` contiene la logica condivisa e
usa la pipeline CV ottimizzata e le tre mappe esportate
in `ros2_maps/semantic` dal mapper offline:

| File | Topic | Uso |
| --- | --- | --- |
| `limo_map_laser.yaml` | `/limo/map_package/online/maps/laser_map` | confronto laser AMCL |
| `limo_map_cv_obstacle.yaml` | `/limo/map_package/online/maps/cv_obstacle` | confronto CV AMCL |
| `limo_map_complete.yaml` | `/map` | riferimento completo per visualizzazione/navigation |

La cloud `/limo/cv_package/visual_ptcld/points` contiene `x,y,z` FLOAT32 e
`class_id` UINT8. AMCL seleziona **turquoise=2, white=3, boardwalk=4**; i blu
1/5 e le altre classi non votano. Le tre classi vengono unite perché la mappa
CV di riferimento è binaria. Nessuna immagine o griglia BEV locale entra in AMCL.

Per ogni aggiornamento laser:

1. AMCL aggiorna le particelle con il modello laser sulla sola mappa laser.
2. Sceglie la cloud col timestamp più vicino, entro `cv_sync_tolerance`.
3. Trasforma la cloud nel frame base al timestamp laser usando `odom` come
   frame fisso, senza dipendere dalla posa globale stimata da AMCL.
4. Dopo il filtro delle classi, aggrega in voxel **XY** da `cv_voxel_size`
   (default 0.075 m). Usa il centroide dei punti di ciascun voxel, con peso 1.
   La riduzione da 2 cm in `visual_ptcld` resta il primo stadio. Non si
   moltiplica il peso per il numero di punti o classi nello stesso voxel.
5. Trasforma questo insieme con **ogni posa candidata** e calcola la frazione
   di voxel che non ricadono su celle occupate della mappa CV.
6. Applica la formula del riferimento Humble e normalizza prima del resampling:
   `w_final ∝ w_laser^laser_weight_factor × exp(-cv_weight_factor × cv_sad_gain × mismatch)`.

Il mismatch conserva la regola SAD positiva di Humble: libero, sconosciuto e
fuori mappa danno disaccordo rispetto a un ostacolo osservato. L'assenza di punti
non prova spazio libero. White è una classe positiva, non lo sfondo bianco di
un'immagine. Non viene più usata una mappa street separata.

Cloud assente, troppo distante nel tempo, malformata, TF indisponibile o meno di
`cv_min_points` voxel (default 5) lasciano i pesi laser invariati. Lo stesso
frame CV non viene riapplicato in aggiornamenti consecutivi. `cv_enabled:=false`
o `cv_weight_factor:=0.0` disabilitano la fusione. Come in Humble,
`laser_weight_factor` si applica soltanto quando avviene la fusione CV.
I parametri CV si leggono alla configurazione del nodo: per cambiarli riavviare.

## Avvio

Dopo la build di `nav2_amcl`, `limo_rviz`, `cv_package`, `online_map_package`
e il source del workspace:

```bash
cd /workspace
colcon build --packages-select nav2_amcl limo_rviz cv_package online_map_package --symlink-install
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
desktop e avvia esclusivamente RViz.

AMCL pubblica `map -> odom`; fornire una posa iniziale tramite RViz oppure il
servizio AMCL di localizzazione globale. Non avviare contemporaneamente SLAM
che pubblichi lo stesso TF. Il launch avvia localizzazione, map server, CV e RViz
opzionali: i vecchi nodi `online_metric_bev`, `cv_2_ptcld`, `cv_amcl_debug`,
`online_map` e `local_ptcld`, basati sulle vecchie griglie, non vengono avviati.
I relativi sorgenti restano disponibili, ma non sono stati convertiti in questa
modifica alla localizzazione.

I log `CV cloud fusion` mostrano differenza temporale, punti in ingresso,
voxel, particelle ed effettivo numero di confronti voxel × particelle.

## Verifica

`nav2_amcl/test/test_cv_cloud.cpp` verifica selezione e fusione delle classi,
voxelizzazione, layout/endian, trasformazioni e pesi. `test_cv_sync.cpp` verifica
la compensazione temporale e il comportamento senza dati utilizzabili; i test Python in
`test/test_localization_launch.py` verificano mappe, topic e parametri del launch.
Il tuning e la convergenza vanno poi valutati con sensori reali o rosbag.
