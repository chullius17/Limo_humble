# LIMO application launch

The application includes online localization/mapping, trajectory planning and
control. Start sensors first. CV is normally started separately:

```bash
ros2 launch cv_package cv_real.launch.py
ros2 launch user_package limo_app_real.launch.py
```

Alternatively, start CV once through the application:

```bash
ros2 launch user_package limo_app_real.launch.py start_cv:=true
```

`start_cv` defaults to false to avoid a duplicate detector when CV is already
running. It is forwarded to online mapping, whose `mapping_real.yaml` selects
`cv_real.yaml` and the waterfall detector. `limo_app_sim.launch.py` selects
`cv_sim.yaml` and the HSV detector. `limo_app.launch.py` also uses the simulation
profile, with a wall clock; changing the clock does not change the detector.

An optional `cv_config:=/absolute/path/to/profile.yaml` overrides the selected
CV profile when starting CV. All consumers continue to receive the semantic
cloud on `/limo/cv_package/visual_ptcld/points`; the CV launch connects waterfall
labels to the point-cloud node. Detector runtime parameters are under `/lane_node`.
