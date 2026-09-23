# Computer vision profiles

Start CV explicitly, independently of mapping:

```bash
# Physical robot, after custom_start/limo_real.launch.py starts the sensors:
ros2 launch cv_package cv_real.launch.py
# Simulation:
ros2 launch cv_package cv_sim.launch.py
```

`cv_real.yaml` enables `visual_ptcld.road_boardwalk_only`. The outgoing cloud
`/limo/cv_package/visual_ptcld/points` contains only these semantic classes:

| Class ID | Meaning | Mapping cost |
| --- | --- | --- |
| 1 | Border road | 0 |
| 5 | Interior road | 0 |
| 4 | Boardwalk | 90 |
| 6 | Interior boardwalk | 90 |

Yellow-line points (2) and unclassified background points (3) are excluded.
They are never promoted to road or boardwalk. Filtering happens after metric
boardwalk recognition, before cloud voxelization and publication. The intermediate
lane-label image still contains background candidates needed to recognize
boardwalk and yellow labels needed to avoid treating yellow pixels as road.
Those intermediate labels are not map observations.

`road_boardwalk_only` is a startup parameter and requires `enable_boardwalk: true`.
It defaults to false, so the simulation CV retains all existing classes and rules.
Offline and online mapping profiles no longer start CV automatically. Launch
CV separately; the mapping launch's explicit `start_cv:=true` override remains
available. Restart CV after changing its profile.
