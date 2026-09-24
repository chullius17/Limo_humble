# Report: correzione odometria del LIMO reale

Data: 24 settembre 2026. Robot in modalità meccanica Ackermann, ROS 2 Foxy.

## Problema e diagnosi

Durante le curve e le frenate, le scansioni lidar apparivano ruotate e il
mapping produceva pareti duplicate. Nei rosbag, dopo l'arresto, lo yaw di
`/odom` e dell'IMU restava stabile mentre quello di `/odometry/filtered`
continuava a correggersi di diversi gradi.

La velocità angolare `/odom.twist.twist.angular.z` era sempre zero, anche in
curva, mentre il giroscopio misurava una rotazione. L'EKF fondeva
entrambi gli ingressi, in conflitto tra loro. Le covarianze delle velocità di
`/odom` erano inoltre tutte zero: non esprimevano un'incertezza adeguata.

## Modifica

Nel solo [launch reale](launch/limo_real.launch.py), il parametro
`odom0_config` viene sovrascritto impostando a `False` il dodicesimo elemento
(`vyaw`, indice 11). La velocità angolare continua a essere fornita da
`/limo/imu`; lo yaw della posa di `/odom` e gli altri ingressi restano abilitati
come prima. Il [file EKF condiviso](config/ekf.yaml) e il launch della
simulazione non sono stati modificati.

Al nodo EKF è stato aggiunto anche `namespace='/'`. Su Foxy, senza namespace
esplicito, il dizionario dei parametri del launch viene applicato a `/**` e
può essere scavalcato dalla configurazione YAML specifica del nodo. Questo
impediva alla prima modifica di diventare effettiva.
[Riferimento al comportamento di Foxy](https://github.com/ros2/launch_ros/blob/foxy/launch_ros/launch_ros/actions/node.py#L119-L126).

## Verifica sperimentale

Confronto dello yaw filtrato durante le fermate individuate dalla velocità
odometrica, escludendo la sosta iniziale:

| Registrazione | Stato della modifica | Massima variazione dello yaw durante la sosta |
| --- | --- | ---: |
| `~/prova_frenate` | Configurazione originale | 5,83° |
| `~/prova_frenate_fix` | Override non ancora effettivo | 3,56° |
| `~/prova_frenate_fix2` | Modifica attiva e parametro verificato | 0,21° |

Nell'ultima prova, le cinque fermate mostrano variazioni tra 0,02° e 0,21°.
Sono scomparsi anche gli azzeramenti anomali della velocità angolare filtrata
nei campioni confrontati durante le curve. Le velocità angolari massime erano
simili: circa 63,5°/s nella prima prova e 62°/s nell'ultima.

## Controllo dopo l'avvio

Dopo aver riavviato il bringup reale, verificare:

```bash
ros2 param get /ekf_filter_node odom0_config
```

Il valore atteso è:

```text
[True, True, False, False, False, True, True, True, False, False, False, False, False, False, False]
```

La correzione è verificata sui rosbag per il comportamento dell'EKF. La qualità
del mapping va verificata con una nuova mappa: quelle precedenti possono
conservare le pareti duplicate. Il dato angolare errato del driver e le sue
covarianze non sono stati corretti da questa modifica.


## Indagine successiva: instabilità durante il mapping

La correzione EKF sopra non ha risolto il mapping. In RViz il problema è stato
osservato con Fixed Frame `map`: in questo riferimento conta anche la
trasformazione `map -> odom` calcolata da SLAM.

- `prova_tf_mapping`: 6 correzioni di `map -> odom`, fino a circa 0,61 m.
- `prova_tf_mapping_fix`: 169 correzioni, con un salto di circa 1,75 m e 20,6°.
  Non si osservano salti metrici corrispondenti in `odom -> base_link`, ma
  sono presenti intervalli senza campioni fino a circa 0,4 s.
- Nel YAML YDLidar, `resolution_fixed` è stato corretto in `fixed_resolution`,
  il nome letto dal driver. Le scansioni sono passate da 456–465 a 410 raggi
  costanti. Questo elimina la variazione della dimensione, ma **non dimostra
  un miglioramento del mapping**: la seconda registrazione è peggiore.

Un confronto indipendente fra scansioni successive (ICP 2D, coppie distanti
circa 0,4 s) trova sul rettilineo una discrepanza mediana rispetto
all'odometria di circa 1,3–1,4 cm nelle coppie analizzate. È un controllo
locale su un campione delle scansioni, non una validazione globale della mappa.

La TF attuale `base_link -> laser_frame` è `(0, 0, 0.02)` con rotazione
identità. Un adattamento dei movimenti nei due rosbag suggerisce un offset
longitudinale di circa 0,19 m rispetto all'origine odometrica. È un indizio
per verificare il riferimento del driver e la posizione fisica del lidar,
non una calibrazione da applicare direttamente: il fit può assorbire anche
errori odometrici e distorsioni delle scansioni. La TF non è stata modificata.

Sul Jetson acceso è stata misurata anche saturazione della CPU: circa 99%
sui quattro core, con 8–9 task eseguibili. Erano attivi camera, elaborazione
visiva, mappa semantica e RViz oltre ai nodi di base. Questo motiva un confronto
con carico ridotto, ma non prova da solo la causa dei salti. Le frequenze di
una sonda Python sotto carico possono sottostimare quelle dei publisher.

### Avvio minimo per isolare il problema

Da eseguire solo dopo aver terminato i launch precedenti, con robot fermo,
e aver caricato l'ambiente ROS e il workspace. Due terminali sul Jetson:

```bash
ros2 launch custom_start limo_real.launch.py use_camera:=false
```

```bash
ros2 launch offline_map_package map.launch.py mode:=backend start_cv:=false start_mapper:=false
```

Questo avvia driver, lidar, EKF e SLAM senza camera, CV, mappa semantica o
finestre locali. Per visualizzare, usare RViz sul PC collegato al medesimo
grafo ROS; confrontare Fixed Frame `odom` e `map`. Evitare launch separati
che riaccendano CV o una seconda istanza di SLAM. Il controllo da fermo di questo avvio è riportato sotto; resta da
verificarne il comportamento in movimento.


### Verifica della sessione minima da fermo

Il confronto previsto con sospensione di 15 secondi non è utilizzabile:
i nodi sono stati chiusi manualmente durante la misura iniziale. Non si
attribuisce quindi valore causale ai suoi risultati prima/dopo.

È stato poi avviato sul Jetson il bringup senza camera e il mapping in
modalità `backend`, senza CV né mapper semantico, con i comandi sopra.
Nella nuova sessione il carico CPU misurato è circa 38–41%, contro circa
99% osservato nella precedente sessione completa. È un confronto fra due
sessioni, non il confronto controllato inizialmente previsto.

Una sonda dedicata alle TF, con coda sufficiente per non scartare i messaggi
fra i due collegamenti, ha misurato circa 10 secondi utili a robot fermo:

| Misura | `odom -> base_link` | `map -> odom` |
| --- | ---: | ---: |
| Frequenza osservata | 99,7 Hz | 20,0 Hz |
| Massimo intervallo fra timestamp | 22,1 ms | 55,5 ms |
| Massimo spostamento rispetto al primo campione | 0 m | 0 m |
| Massima variazione di yaw rispetto al primo campione | 0,0105° | 0° |

Le frequenze inferiori viste con una precedente sonda a profondità 1 non
vanno interpretate come frequenze effettive dei publisher: quella sonda
poteva perdere campioni, soprattutto sul topic condiviso `/tf`.

Verificati anche `odom0_config` con indice 11 disabilitato, `use_sim_time`
falso e `fixed_resolution` vero. In questo avvio il lidar pubblica 450 raggi
costanti, diversi dai 410 della registrazione precedente: la dimensione
viene stimata dal driver all'avvio. I log segnalano occasionalmente più
punti reali rispetto ai 450 fissati; questo resta un aspetto da verificare.

La mappa viene pubblicata. Questi controlli dimostrano il funzionamento da
fermo della sessione minima, **non risolvono né validano ancora il problema
in movimento**. Nessuna nuova calibrazione della TF laser è stata applicata.
I risultati diagnostici sono sul robot in `/tmp/limo_minimal_validation.json`;
i log in `/tmp/limo_minimal_bringup.log` e `/tmp/limo_minimal_mapping.log`.


Il confronto successivo fra `odom` e `map` e le riproduzioni isolate della
SLAM sono descritti nel [report mapping](README_mapping_reale.md).
