# Configurazione della modalità Ackermann sul LIMO reale

## Obiettivo

Configurare il LIMO reale affinché i comandi ROS pubblicati su `/cmd_vel`
vengano convertiti secondo la cinematica Ackermann, mantenendo la stessa
impostazione usata nella simulazione.

Il robot era stato convertito meccanicamente in Ackermann e risultava
controllabile correttamente tramite l'app AgileX. Il driver ROS continuava però
a interpretare i comandi come Mecanum perché il firmware riportava sempre:

```yaml
motion_mode: 2
```

## Stato osservato

Il topic `/limo_status` mostrava:

```yaml
vehicle_state: 0
control_mode: 1
battery_voltage: 10.3
error_code: 0
motion_mode: 2
```

I valori rilevanti sono:

- `control_mode: 1`: il telaio accetta i comandi ROS;
- `motion_mode: 2`: il firmware dichiara ancora la modalità Mecanum;
- `error_code: 0`: il telaio non segnala errori.

Quando il controllo veniva assunto dall'app, `control_mode` diventava `2`.
L'app doveva quindi essere chiusa o disconnessa prima di usare il teleop ROS.

## Perché il robot non sterzava correttamente

Il callback originale del driver selezionava direttamente la conversione del
comando in base al valore ricevuto dal firmware:

```cpp
switch (motion_mode_) {
```

Con `motion_mode_ == 2`, un messaggio `geometry_msgs/msg/Twist` veniva sempre
trattato come comando Mecanum. Questo accadeva anche se la geometria del robot
era già stata convertita fisicamente in Ackermann.

Il funzionamento tramite telefono confermava che servosterzo e conversione
meccanica Ackermann erano operativi. Il problema era quindi nella scelta della
conversione effettuata dal driver ROS.

## Soluzione adottata

È stato aggiunto un parametro chiamato `motion_mode`, indipendente dal valore
grezzo comunicato dal telaio:

```text
-1 = modalità automatica, usa il feedback del firmware
 0 = quattro ruote differenziali
 1 = Ackermann
 2 = Mecanum
```

La modalità automatica `-1` resta il comportamento predefinito, preservando la
compatibilità con le altre launch.

### Modifica al driver

Nel file:

```text
workspace/src/limo_ros2/limo_base/include/limo_base/limo_driver.h
```

è stato aggiunto:

```cpp
int64_t motion_mode_override_ = -1;
```

Nel costruttore di `LimoDriver`, nel file:

```text
workspace/src/limo_ros2/limo_base/src/limo_driver.cpp
```

il parametro viene dichiarato e letto:

```cpp
this->declare_parameter("motion_mode");
this->get_parameter_or<int64_t>("motion_mode", motion_mode_override_, -1);
```

Quando l'override è valido, il nodo stampa all'avvio:

```text
Command motion mode override: 1
```

All'arrivo di ogni `/cmd_vel`, il driver legge il parametro e sceglie la
conversione da applicare:

```cpp
int64_t configured_mode = -1;
this->get_parameter_or<int64_t>("motion_mode", configured_mode, -1);

const uint8_t command_motion_mode =
    configured_mode >= MODE_FOUR_DIFF && configured_mode <= MODE_MCNAMU
        ? static_cast<uint8_t>(configured_mode)
        : motion_mode_;

switch (command_motion_mode) {
```

In questo modo, con `motion_mode=1`, `linear.x` e `angular.z` vengono convertiti
in velocità longitudinale e angolo di sterzo Ackermann.

### Modifica alla launch di limo_base

Nel file:

```text
workspace/src/limo_ros2/limo_base/launch/limo_base.launch.py
```

è stato esposto l'argomento:

```python
DeclareLaunchArgument(
    'motion_mode',
    default_value='-1',
)
```

e il relativo parametro viene passato al nodo `limo_base`.

### Configurazione della launch reale

Nel file:

```text
workspace/src/custom_start/launch/limo_real.launch.py
```

la modalità dei comandi viene forzata ad Ackermann:

```python
'motion_mode': '1',
```

Questa impostazione riguarda soltanto l'avvio del robot reale tramite
`custom_start limo_real.launch.py`.

## Significato di `/limo_status`

Il topic `/limo_status` continua intenzionalmente a rappresentare il feedback
grezzo del telaio. È quindi possibile osservare contemporaneamente:

```text
parametro /limo_base motion_mode: 1
topic /limo_status motion_mode: 2
```

Questi valori non sono più in conflitto:

- il parametro `1` seleziona la conversione Ackermann dei comandi ROS;
- il valore `2` documenta ciò che il firmware continua a dichiarare.

La verifica corretta dell'override è il parametro del nodo e il messaggio di
log `Command motion mode override: 1`, non il campo del topic di stato.

## Build

Arrestare prima tutte le launch, quindi eseguire:

```bash
cd ~/Ajeje_Brazorf_ws/Limo_humble/limo_sim/workspace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash

colcon build \
  --packages-select limo_base custom_start \
  --symlink-install \
  --cmake-clean-first \
  --cmake-clean-cache

source install/local_setup.bash
```

## Avvio e verifica

Chiudere prima l'app AgileX, affinché il controllo rimanga assegnato a ROS.

```bash
ros2 launch custom_start limo_real.launch.py
```

Nel log deve apparire:

```text
Command motion mode override: 1
```

Controllare il parametro effettivo:

```bash
ros2 param get /limo_base motion_mode
```

Risultato atteso:

```text
Integer value is: 1
```

Controllare inoltre che ROS abbia il controllo:

```bash
ros2 topic echo /limo_status
```

Il campo importante per il controllo è:

```yaml
control_mode: 1
```

Il campo `motion_mode` può rimanere `2`, perché arriva direttamente dal
firmware.

## Teleoperazione Ackermann

Avviare il teleop con velocità iniziali moderate:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p speed:=0.15 -p turn:=0.6
```

Comandi utili:

```text
u = avanti sterzando a sinistra
i = avanti diritto
o = avanti sterzando a destra
m = indietro sterzando
, = indietro diritto
. = indietro sterzando
```

I tasti `j` e `l` inviano una rotazione sul posto con `linear.x=0`. Questa
manovra non è realizzabile con la cinematica Ackermann; per osservare una curva
occorre usare `u` oppure `o`.

Il flusso finale dei comandi è:

```text
teleop_twist_keyboard
        |
        v
 /cmd_vel (Twist)
        |
        v
 override motion_mode=1
        |
        v
 conversione Ackermann
        |
        v
 velocità + angolo di sterzo inviati al telaio
```

## Note

- La tensione osservata era circa `10.3 V`; è consigliabile eseguire le prove
  con la batteria carica, soprattutto per il corretto azionamento del servo.
- L'override modifica l'interpretazione dei comandi ROS e non riprogramma il
  firmware del telaio.
- La configurazione meccanica delle levette anteriori deve rimanere in
  posizione Ackermann durante l'uso di questo override.
