# Diagnosi del mapping sul LIMO reale

24 settembre 2026 — robot Ackermann, ROS 2 Foxy.

## Riscontro sul robot

Con la sessione minima (driver, lidar, EKF, SLAM e successivamente RViz),
il laser resta allineato nel rettilineo lento con Fixed Frame `odom`.
Ripetendo il controllo con Fixed Frame `map`, l'utente osserva spostamenti
marcati. Questo localizza il movimento visibile nella correzione
`map -> odom`, senza dimostrare che gli ingressi alla SLAM siano perfetti.

La CPU era circa al 99% nella sessione completa e circa al 40% nella
sessione minima senza RViz. Alleggerire il carico migliora i tempi, ma il
difetto del mapping persiste. I controlli da fermo e la precedente modifica
EKF sono nel [report odometrico](README_odometria_reale.md).

## Confronti sullo stesso rosbag

Rosbag `~/prova_tf_mapping_fix`: 540 scansioni e 4713 messaggi TF odometrici.
Riproduzione a metà velocità, con orologio simulato, nel dominio ROS 187
limitato a localhost. Sono ripubblicate solo scansioni, TF statiche e
`odom -> base_link`: la `map -> odom` registrata è esclusa, così viene
misurata esclusivamente quella prodotta dalla nuova istanza di SLAM.
Non vengono riprodotti comandi al robot.

Ogni prova parte con una nuova SLAM e modifica soltanto la voce indicata.

| Variante | Correzioni > 5° | Correzioni > 25 cm | Massima correzione angolare | Massima correzione traslazionale |
| --- | ---: | ---: | ---: | ---: |
| Configurazione attuale | 22 | 18 | 21,79° | 0,833 m |
| `do_loop_closing: false` | 2 | 3 | 10,70° | 0,416 m |
| Solo TF lidar X = 0,19 m | 13 | 10 | 27,81° | 1,169 m |
| Solo `loop_match_minimum_response_fine: 0.8` | 14 | 14 | 27,30° | 2,243 m |

Le massime correzioni angolari e traslazionali possono appartenere a istanti
diversi. Sono variazioni di `map -> odom`, non misure dello spostamento fisico
del robot. Le riproduzioni asincrone hanno scartato poche scansioni; i
risultati non sono una riproduzione identica campione per campione della
registrazione originale. Il numero di salti non misura da solo la qualità
globale della mappa: manca un riferimento esterno della traiettoria reale.

Il difetto si riproduce anche rallentando i dati. Le chiusure degli anelli
aggravano i salti in questo rosbag, ma disabilitarle non elimina tutti gli
errori. Né l'offset stimato del lidar né la sola soglia più severa sono
stati validati come soluzione e non sono stati applicati al robot.

## Configurazione diagnostica attiva

Per il prossimo controllo fisico è stata scelta la variante con
`do_loop_closing: false`, salvata sul robot in
`/tmp/limo_mapping_no_loop.yaml`. È una copia del profilo reale con questa
sola modifica SLAM; CV, mapper semantico e finestre del launch sono disabilitati.
Il file `mapping_real.yaml` del progetto mantiene i suoi valori precedenti.

Il confronto tra scansioni rimane attivo. Si rinuncia temporaneamente alle
chiusure degli anelli e alla loro correzione globale della deriva: questa
configurazione serve a isolare il problema, non costituisce ancora una
soluzione definitiva per costruire mappe estese.

Prima del riavvio della sola SLAM è stato salvato il grafo precedente in
`/tmp/limo_mapping_before_no_loop.posegraph` e nel relativo file `.data`.
Driver, lidar, EKF e loro TF non sono stati riavviati per questo confronto.

Sul robot:

- Log della prova attiva: `/tmp/limo_minimal_mapping_no_loop_retry.log`.
- Processi e comando di avvio: `/tmp/limo_minimal_session.json`.
- Risultati delle riproduzioni: `/tmp/limo_replay_{baseline,no_loop,laser_x,strict_loop}.json`.
- Script della riproduzione isolata: `/tmp/limo_slam_replay.py`.

Il primo riavvio non ha prodotto una mappa, pur con ingressi e parametri
corretti. Dopo un ulteriore riavvio della sola SLAM è stata verificata la
pubblicazione di una nuova mappa di 149 × 232 celle. La causa di quel
problema di avvio non è stata determinata.

Resta da ripetere il breve rettilineo lento in RViz con Fixed Frame `map`
sulla nuova mappa. La conferma visiva e il comportamento in curva sono
ancora da verificare.


## Causa individuata: raggi invalidi pubblicati a zero

Nella registrazione reale `prova_tf_mapping_fix`, il 42,12% dei raggi di
`/scan` vale zero. Il driver leggeva `invalid_range_is_inf`, ma non usava
mai il parametro: sia i campioni invalidi dello SDK sia i bin angolari
senza un campione rimanevano a zero.

Nella versione installata di SLAM Toolbox (Foxy 2.4.1), Karto usa anche
punti non filtrati nel matcher. `GridIndexLookup::ComputeOffsets` scarta
NaN e infinito, ma non i valori finiti a zero: questi rappresentano punti
all'origine del sensore. La loro presenza altera il confronto tra scansioni,
anche quando RViz esclude gli stessi raggi perché inferiori a `range_min`.
Questo spiega perché la visualizzazione in `odom` può apparire corretta
mentre lo scan matching produce correzioni errate.

### Confronto isolato con i dati del robot

Sul PC, con SLAM Toolbox Foxy 2.4.1 e dominio ROS 187 limitato a localhost,
sono state riprodotte le stesse 540 scansioni e 4713 TF odometriche a
velocità reale. Sono esclusi comandi al robot e `map -> odom` registrata.
La variante corretta cambia solo i raggi invalidi in `+inf`.

| Variante | Correzioni > 25 cm | Correzioni > 5° | Massima correzione traslazionale | Massima correzione angolare |
| --- | ---: | ---: | ---: | ---: |
| Zeri originali, loop closure disabilitata | 5 | 3 | 0,409 m | 10,70° |
| Raggi invalidi a +inf, loop closure disabilitata | 0 | 0 | 0,143 m | 4,00° |
| Raggi invalidi a +inf, loop closure abilitata | 0 | 0 | 0,156 m | 4,20° |

Anche le mappe risultanti mostrano una riduzione netta delle pareti
sdoppiate e deformate. Le prove asincrone possono scartare alcune scansioni:
non costituiscono un confronto deterministico campione per campione né una
misura di accuratezza rispetto a una traiettoria di riferimento. Rimane
necessaria la conferma in movimento sul robot dopo il riavvio.

### Correzione nel workspace

- Il driver applica ora `invalid_range_is_inf` dopo il riempimento dei bin,
  includendo bin vuoti, zeri, valori negativi/non finiti e fuori intervallo.
  Le distanze valide, i loro indici e la geometria della scansione restano
  inalterati. Con il parametro falso conserva il comportamento precedente.
- Il profilo lidar reale `limo_ros2/limo_bringup/param/ydlidar.yaml` abilita
  `invalid_range_is_inf: true`.
- Nessuna modifica ai parametri SLAM del progetto, alla simulazione o
  alla calibrazione delle TF. Le modifiche precedenti a EKF e
  `fixed_resolution` sono preservate.
- Build del solo driver riuscita sul Jetson; tre test GTest passati
  (raggi invalidi, compatibilità con parametro falso, bin non riempiti).

Artefatti diagnostici sul PC: `/tmp/limo_slam_diagnosis/`, inclusi script,
risultati JSON e `comparison.png`. Nessun commit o push eseguito.
