# Pianificazione delle traiettorie

`traj_package` contiene il planner Nav2 SMAC e `rviz_goal_bridge`, che riceve
gli obiettivi RViz su `/goal_pose`, cerca una soluzione fattibile e pubblica
il percorso. Può essere avviato senza `limo_controller`:

```bash
ros2 launch traj_package trajectory.launch.py
```

Richiede la mappa e le trasformazioni del robot. La configurazione personalizzata
si passa con `planner_params_file:=/percorso/parametri.yaml`.

## Uscite

| Topic | Tipo | Contenuto |
| --- | --- | --- |
| `/limo/planning/path` | `nav_msgs/Path` | Ultimo percorso validato |
| `/limo/planning/status` | `std_msgs/String` | Stato e problemi della pianificazione |

Entrambi i topic usano QoS reliable, transient-local, profondità 1. Un consumer
avviato dopo la pianificazione può quindi ricevere l'ultimo percorso usando
lo stesso QoS. Una nuova richiesta pubblica prima un percorso vuoto per
invalidare il precedente; soltanto una ricerca riuscita pubblica un nuovo
percorso non vuoto.

## Esecuzione

Il pacchetto non espone servizi di controllo e non invia goal `FollowPath`.
I precedenti flag `enable_control` e `auto_start_control` sono stati rimossi.

L'esecuzione appartiene a `limo_controller`: il suo nodo `path_executor` riceve
il percorso e gestisce START, pausa, ripresa e annullamento tramite i servizi
`/limo/control/set_active` e `/limo/control/set_enabled`. Un percorso nuovo
interrompe quello in esecuzione e richiede un nuovo START. La GUI mostra anche
lo stato della pianificazione.

`user_package/limo_app.launch.py` compone mapping, pianificazione e controllo.
