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

/// Select turquoise/white/boardwalk (2/3/4), transform to the robot frame at
/// laser time, then merge all classes into XY voxels. Each centroid votes once.
/// Supports organized clouds, padded rows, arbitrary field offsets and endian.
/// Invalid layouts return false; empty or wholly invalid observations return
/// true with an empty template (no negative evidence).
bool voxelizeCvCloud(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const tf2::Transform & cloud_to_base, double voxel_size,
  std::vector<CvTemplateCell2D> & cells, std::string & error);

}  // namespace nav2_amcl
#endif  // NAV2_AMCL__SENSORS__CV__CV_POINT_CLOUD_HPP_
