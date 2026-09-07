# Avvio della camera Orbbec DaBai U3 su LIMO

## Ambiente

- Robot: AgileX LIMO con NVIDIA Jetson Nano
- Camera: Orbbec DaBai U3
- Sistema ROS: ROS 2 Foxy
- Workspace: `~/Ajeje_Brazorf_ws/Limo_humble/limo_sim/workspace`
- Interfacce USB rilevate:
  - `2bc5:060e`: sensore depth gestito da OpenNI2
  - `2bc5:050e`: interfaccia RGB/UVC

## Problemi riscontrati

### Runtime OpenNI2 mancante durante la build

La build di `astra_camera` inizialmente terminava con:

```text
ament_cmake_symlink_install_files() can't find
.../astra_camera/openni2_redist/arm64/libOpenNI2.so
```

La directory `openni2_redist/arm64` era assente. Inoltre la regola generica
`[Aa][Rr][Mm]64/` presente in `.gitignore` può nascondere questa directory a
Git. È stato quindi ripristinato l'intero runtime ARM64 ufficiale di Orbbec,
compresi:

```text
openni2_redist/arm64/libOpenNI2.so
openni2_redist/arm64/OpenNI2/Drivers/liborbbec.so
openni2_redist/arm64/OpenNI2/Drivers/orbbec.ini
```

### Dipendenza ROS mancante

La launch completa si arrestava perché non trovava `robot_localization`:

```text
PackageNotFoundError: package 'robot_localization' not found
```

La dipendenza è stata installata con:

```bash
sudo apt update
sudo apt install ros-foxy-robot-localization
```

### Profilo della camera errato

`custom_start/launch/limo_real.launch.py` includeva il profilo generico
`astra.launch.xml`. È stato sostituito con il profilo specifico:

```text
dabai_u3.launch.xml
```

La risoluzione depth è stata inoltre corretta da `640x480` a `640x400`, come
previsto dalla configurazione DaBai U3 inclusa nel driver.

### Caricamento della libreria OpenNI2 sbagliata

Nonostante i file Orbbec fossero installati, il comando seguente mostrava che
il nodo caricava la libreria di sistema:

```bash
ldd install/astra_camera/lib/astra_camera/astra_camera_node | grep OpenNI
```

Risultato errato:

```text
libOpenNI2.so.0 => /usr/lib/libOpenNI2.so.0
```

La copia Orbbec era installata soltanto come `libOpenNI2.so`, mentre il nodo
richiedeva il nome ABI `libOpenNI2.so.0`. In `astra_camera/CMakeLists.txt` è
stata aggiunta l'installazione dello stesso runtime anche con questo nome:

```cmake
install(FILES openni2_redist/${HOST_PLATFORM}/libOpenNI2.so
  DESTINATION lib/
  RENAME libOpenNI2.so.0
)
```

Questo impedisce al loader di scegliere silenziosamente la versione OpenNI2
di sistema, incompatibile con il driver Orbbec incluso.

## Regole USB

Le regole `udev` Orbbec includono entrambi gli identificativi USB della
DaBai U3. Si installano con:

```bash
cd ~/Ajeje_Brazorf_ws/Limo_humble/limo_sim/workspace/src/ros2_astra_camera/astra_camera/scripts
sudo bash install.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Dopo questi comandi occorre scollegare e ricollegare fisicamente la camera.

## Build finale

```bash
cd ~/Ajeje_Brazorf_ws/Limo_humble/limo_sim/workspace
source /opt/ros/foxy/setup.bash

colcon build \
  --packages-select astra_camera custom_start \
  --symlink-install \
  --cmake-clean-cache

source install/setup.bash
```

Verificare che venga caricata la libreria del workspace:

```bash
ldd install/astra_camera/lib/astra_camera/astra_camera_node | grep OpenNI
```

Il percorso deve puntare a:

```text
.../workspace/install/astra_camera/lib/libOpenNI2.so.0
```

e non a `/usr/lib/libOpenNI2.so.0`.

## Verifica della camera

Controllare prima il collegamento USB:

```bash
lsusb | grep -i 2bc5
```

Provare quindi l'enumerazione OpenNI2:

```bash
ros2 run astra_camera list_devices_node
```

Risultato ottenuto:

```text
Found 1 devices
Device connected: Astra
URI: 2bc5/060e@1/9
Serial number: AU15C2106KW
```

La denominazione generica `Astra` mostrata da OpenNI2 è normale anche usando
il profilo DaBai U3.

## Avvio completo

```bash
cd ~/Ajeje_Brazorf_ws/Limo_humble/limo_sim/workspace
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch custom_start limo_real.launch.py
```

Per controllare i flussi senza interfaccia grafica:

```bash
ros2 topic hz /rgb/image_raw
ros2 topic hz /depth_camera/depth/image_raw
```

Per visualizzarli da una sessione con display grafico:

```bash
ros2 run rqt_image_view rqt_image_view
```

L'errore Qt `could not connect to display` indica una sessione SSH o remota
senza inoltro grafico e non un guasto della camera.

## Avvisi non bloccanti

Il messaggio seguente non impedisce il funzionamento della camera:

```text
USB events thread - failed to set priority. This might cause loss of data...
```

Indica soltanto che il thread USB non ha ottenuto una priorità real-time.
