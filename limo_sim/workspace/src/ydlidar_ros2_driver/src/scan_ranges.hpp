#ifndef YDLIDAR_ROS2_DRIVER_SCAN_RANGES_HPP_
#define YDLIDAR_ROS2_DRIVER_SCAN_RANGES_HPP_

#include <cmath>
#include <limits>
#include <vector>

namespace ydlidar_ros2_driver
{
// Apply this after binning so unfilled bins and invalid SDK samples are covered.
inline void normalizeInvalidRanges(
  std::vector<float> & ranges, float range_min, float range_max,
  bool invalid_range_is_inf)
{
  if (!invalid_range_is_inf) {
    return;
  }
  for (float & range : ranges) {
    if (!std::isfinite(range) || range <= 0.0f ||
      range < range_min || range > range_max)
    {
      range = std::numeric_limits<float>::infinity();
    }
  }
}
}  // namespace ydlidar_ros2_driver

#endif  // YDLIDAR_ROS2_DRIVER_SCAN_RANGES_HPP_
