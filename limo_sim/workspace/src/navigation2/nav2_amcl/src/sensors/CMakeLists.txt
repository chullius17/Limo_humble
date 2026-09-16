add_library(sensors_lib SHARED
  cv_likelihood_model.cpp
  cv_point_cloud.cpp
  laser/laser.cpp
  laser/beam_model.cpp
  laser/likelihood_field_model.cpp
  laser/likelihood_field_model_prob.cpp
)
ament_target_dependencies(sensors_lib nav_msgs sensor_msgs tf2)
# map_update_cspace
target_link_libraries(sensors_lib pf_lib map_lib)

install(TARGETS
  sensors_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
