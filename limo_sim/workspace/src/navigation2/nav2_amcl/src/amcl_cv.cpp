// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#include "nav2_amcl/amcl_node.hpp"
#include "nav2_amcl/sensors/cv/cv_point_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "nav2_util/string_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/buffer.h"

namespace nav2_amcl
{
void AmclNode::cvMapReceived(nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(cv_mutex_);
  if (nav2_util::strip_leading_slash(msg->header.frame_id) != global_frame_id_ ||
    !cv_likelihood_model_ || !cv_likelihood_model_->setMap(*msg))
  {
    RCLCPP_ERROR(get_logger(), "Rejected CV obstacle map: invalid layout or global frame");
    return;
  }
  RCLCPP_INFO(
    get_logger(), "Received CV obstacle map: %u x %u @ %.3f m",
    msg->info.width, msg->info.height, msg->info.resolution);
}

void AmclNode::cvCloudReceived(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!active_ || msg->header.frame_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(cv_mutex_);
  cv_cloud_buffer_.push_back(msg);
  while (cv_cloud_buffer_.size() > static_cast<std::size_t>(cv_buffer_size_)) {
    cv_cloud_buffer_.pop_front();
  }
}

bool AmclNode::hasValidLaserInformation(const sensor_msgs::msg::LaserScan & scan) const
{
  if (!std::isfinite(scan.range_min) || !std::isfinite(scan.range_max) ||
    scan.range_min < 0.0 || scan.range_max <= scan.range_min ||
    !std::isfinite(scan.angle_min) || !std::isfinite(scan.angle_increment))
  {
    return false;
  }
  const double minimum = laser_min_range_ > 0.0 ?
    std::max(laser_min_range_, static_cast<double>(scan.range_min)) : scan.range_min;
  const double maximum = laser_max_range_ > 0.0 ?
    std::min(laser_max_range_, static_cast<double>(scan.range_max)) : scan.range_max;
  return std::any_of(
    scan.ranges.begin(), scan.ranges.end(),
    [minimum, maximum](float range) {
      return std::isfinite(range) && range > minimum && range < maximum;
    });
}

bool AmclNode::applyCvFusion(
  pf_sample_set_t * set, const builtin_interfaces::msg::Time & laser_stamp,
  bool lidar_information_valid)
{
  if (cv_weight_factor_ == 0.0 || cv_sad_gain_ == 0.0 || !set || set->sample_count <= 0) {
    return false;
  }
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
  double time_error = std::numeric_limits<double>::infinity();
  bool reused = false;
  {
    std::lock_guard<std::mutex> lock(cv_mutex_);
    if (!cv_likelihood_model_ || !cv_likelihood_model_->ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for CV obstacle map");
      return false;
    }
    for (const auto & candidate : cv_cloud_buffer_) {
      const double error = std::abs(
        (rclcpp::Time(candidate->header.stamp) - rclcpp::Time(laser_stamp)).seconds());
      if (error < time_error) {
        time_error = error;
        cloud = candidate;
      }
    }
    if (!cloud || time_error > cv_sync_tolerance_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "No synchronized CV cloud (dt=%.3f s, tolerance=%.3f s); skipping CV update",
        time_error, cv_sync_tolerance_);
      return false;
    }
    if (last_fused_cv_cloud_) {
      // Never reprocess the same scan or go backwards through the CV buffer.
      if (rclcpp::Time(laser_stamp) <= rclcpp::Time(last_cv_fusion_stamp_) ||
        rclcpp::Time(cloud->header.stamp) < rclcpp::Time(last_fused_cv_cloud_->header.stamp))
      {
        return false;
      }
      reused = cloud->header.stamp == last_fused_cv_cloud_->header.stamp;
      // Match Humble's timeout, but allow reuse only without useful lidar.
      // The timeout above always refers to the original cloud timestamp.
      if (reused && ((lidar_information_valid && laser_weight_factor_ > 0.0) ||
        rclcpp::Time(cloud->header.stamp) > rclcpp::Time(laser_stamp)))
      {
        return false;
      }
    }
  }

  tf2::Transform cloud_to_base;
  try {
    // Transform source at cloud time into base at laser time using only odom.
    // Using map here would incorrectly depend on AMCL's current best estimate.
    const auto transform = tf_buffer_->lookupTransform(
      base_frame_id_, tf2_ros::fromMsg(laser_stamp),
      nav2_util::strip_leading_slash(cloud->header.frame_id),
      tf2_ros::fromMsg(cloud->header.stamp), odom_frame_id_);
    tf2::fromMsg(transform.transform, cloud_to_base);
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Cannot synchronize CV cloud: %s", error.what());
    return false;
  }
  std::vector<CvTemplateCell2D> cells;
  std::string error;
  if (!voxelizeCvCloud(*cloud, cloud_to_base, cv_voxel_size_, cells, error)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Rejected CV cloud: %s", error.c_str());
    return false;
  }
  if (cells.size() < cv_min_points_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "CV cloud has %zu obstacle/road voxels, minimum %.0f; skipping CV update",
      cells.size(), cv_min_points_);
    return false;
  }
  const auto non_road_voxels = static_cast<std::size_t>(std::count_if(
      cells.begin(), cells.end(), [](const CvTemplateCell2D & cell) {
        return cell.occupancy > 0.5;
      }));
  if (non_road_voxels < static_cast<std::size_t>(cv_min_non_road_voxels_)) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "CV cloud has %zu non-road voxels, minimum %d; skipping CV update",
      non_road_voxels, cv_min_non_road_voxels_);
    return false;
  }
  std::lock_guard<std::mutex> lock(cv_mutex_);
  const auto score = cv_likelihood_model_->scoreSad(set, cells);
  CvLikelihoodModel::QualityReport quality;
  if (cv_quality_gate_enabled_ && !CvLikelihoodModel::assessQuality(
      set, score, cv_weight_factor_ * cv_sad_gain_, cv_quality_limits_, quality))
  {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "CV update rejected: %s, information=%.4f, position_stddev=%.3f m, yaw_stddev=%.3f rad",
      quality.reason, quality.information, quality.position_stddev, quality.yaw_stddev);
    return false;
  }
  if (!CvLikelihoodModel::fuseWeights(
      set, score, laser_weight_factor_, cv_weight_factor_, cv_sad_gain_))
  {
    return false;
  }
  last_fused_cv_cloud_ = cloud;
  last_cv_fusion_stamp_ = laser_stamp;
  if (workload_logging_enabled_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "CV cloud fusion: dt=%.4f s input=%llu voxels=%zu non_road=%zu particles=%d "
      "evaluations=%llu reused=%s lidar_valid=%s laser_weight=%.3f",
      time_error, static_cast<unsigned long long>(cloud->width) * cloud->height,
      cells.size(), non_road_voxels, set->sample_count,
      static_cast<unsigned long long>(cells.size()) * set->sample_count,
      reused ? "true" : "false", lidar_information_valid ? "true" : "false",
      laser_weight_factor_);
  }
  return true;
}
}  // namespace nav2_amcl
