// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#include "nav2_amcl/amcl_node.hpp"
#include "nav2_amcl/sensors/cv/cv_point_cloud.hpp"

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

bool AmclNode::applyCvFusion(
  pf_sample_set_t * set, const builtin_interfaces::msg::Time & laser_stamp)
{
  if (cv_weight_factor_ == 0.0 || cv_sad_gain_ == 0.0 || !set || set->sample_count <= 0) {
    return false;
  }
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
  double time_error = std::numeric_limits<double>::infinity();
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
        "No synchronized CV cloud (dt=%.3f s, tolerance=%.3f s); using laser only",
        time_error, cv_sync_tolerance_);
      return false;
    }
    // A repeated laser update must not count the same semantic frame twice.
    if (last_fused_cv_cloud_ && cloud->header.stamp == last_fused_cv_cloud_->header.stamp) {
      return false;
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
      "CV cloud has %zu obstacle voxels, minimum %.0f; using laser only",
      cells.size(), cv_min_points_);
    return false;
  }
  std::lock_guard<std::mutex> lock(cv_mutex_);
  const auto score = cv_likelihood_model_->scoreSad(set, cells);
  if (!CvLikelihoodModel::fuseWeights(
      set, score, laser_weight_factor_, cv_weight_factor_, cv_sad_gain_))
  {
    return false;
  }
  last_fused_cv_cloud_ = cloud;
  if (workload_logging_enabled_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "CV cloud fusion: dt=%.4f s input=%llu voxels=%zu particles=%d evaluations=%llu",
      time_error, static_cast<unsigned long long>(cloud->width) * cloud->height,
      cells.size(), set->sample_count,
      static_cast<unsigned long long>(cells.size()) * set->sample_count);
  }
  return true;
}
}  // namespace nav2_amcl
