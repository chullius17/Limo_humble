// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#ifndef NAV2_AMCL__SENSORS__CV__CV_POINT_CLOUD_HPP_
#define NAV2_AMCL__SENSORS__CV__CV_POINT_CLOUD_HPP_

#include <string>
#include <vector>

#include "nav2_amcl/sensors/cv/cv_likelihood_model.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/LinearMath/Transform.h"

namespace nav2_amcl
{

/// Select obstacles (yellow lines/boardwalk/interior boardwalk: 2/4/6) and
/// roads (exterior/interior: 1/5); ignore soft obstacles (3) and unknown classes.
/// Transform to the robot frame at laser time, then merge into XY voxels per
/// polarity. Each centroid votes once, with occupancy 1 (obstacle) or 0 (road).
/// Supports organized clouds, padded rows, arbitrary field offsets and endian.
/// Invalid layouts return false; empty or wholly invalid observations return
/// true with an empty template (no evidence).
bool voxelizeCvCloud(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const tf2::Transform & cloud_to_base, double voxel_size,
  std::vector<CvTemplateCell2D> & cells, std::string & error);

}  // namespace nav2_amcl
#endif  // NAV2_AMCL__SENSORS__CV__CV_POINT_CLOUD_HPP_
