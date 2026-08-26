#include "limo_inflation/border_follow_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include <opencv2/imgproc.hpp>
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace limo_inflation
{

void BorderFollowLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("Unable to lock lifecycle node");
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter(
    "source_topic",
    rclcpp::ParameterValue(
      "/limo/nav_map_package/online/nav_map/combined_grid"));
  declareParameter("obstacle_threshold", rclcpp::ParameterValue(10));
  declareParameter("robot_width", rclcpp::ParameterValue(0.20));
  declareParameter("safety_margin", rclcpp::ParameterValue(0.05));
  declareParameter("distance_tolerance", rclcpp::ParameterValue(0.02));
  declareParameter("near_max_cost", rclcpp::ParameterValue(200));
  declareParameter("far_cost_slope", rclcpp::ParameterValue(20.0));
  declareParameter("far_max_cost", rclcpp::ParameterValue(25));

  double robot_width;
  double safety_margin;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".source_topic", source_topic_);
  node->get_parameter(name_ + ".obstacle_threshold", obstacle_threshold_);
  node->get_parameter(name_ + ".robot_width", robot_width);
  node->get_parameter(name_ + ".safety_margin", safety_margin);
  node->get_parameter(name_ + ".distance_tolerance", profile_.tolerance);
  int near_max_cost;
  int far_max_cost;
  node->get_parameter(name_ + ".near_max_cost", near_max_cost);
  node->get_parameter(name_ + ".far_cost_slope", profile_.far_cost_slope);
  node->get_parameter(name_ + ".far_max_cost", far_max_cost);
  profile_.target_distance = 0.5 * robot_width + safety_margin;
  profile_.near_max_cost = static_cast<std::uint8_t>(near_max_cost);
  profile_.far_max_cost = static_cast<std::uint8_t>(far_max_cost);

  if (obstacle_threshold_ < 0 || obstacle_threshold_ > 100) {
    throw std::invalid_argument("obstacle_threshold must be in [0, 100]");
  }
  if (robot_width <= 0.0 || safety_margin < 0.0 ||
    profile_.tolerance < 0.0 ||
    profile_.tolerance >= profile_.target_distance)
  {
    throw std::invalid_argument("Invalid border-follow distance parameters");
  }
  if (near_max_cost < 0 || near_max_cost > 252 ||
    far_max_cost < 0 || far_max_cost > 252 ||
    profile_.far_cost_slope < 0.0)
  {
    throw std::invalid_argument("Invalid border-follow cost parameters");
  }

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    source_topic_, qos,
    std::bind(&BorderFollowLayer::mapCallback, this, std::placeholders::_1));
  current_ = true;

  RCLCPP_INFO(
    logger_,
    "Border-follow layer: source=%s threshold=%d target=%.3f +/- %.3f m",
    source_topic_.c_str(), obstacle_threshold_, profile_.target_distance,
    profile_.tolerance);
}

void BorderFollowLayer::mapCallback(
  const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  source_map_ = msg;
}

void BorderFollowLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_ || !layered_costmap_) {
    return;
  }
  const auto * costmap = layered_costmap_->getCostmap();
  *min_x = std::min(*min_x, costmap->getOriginX());
  *min_y = std::min(*min_y, costmap->getOriginY());
  *max_x = std::max(
    *max_x, costmap->getOriginX() + costmap->getSizeInMetersX());
  *max_y = std::max(
    *max_y, costmap->getOriginY() + costmap->getSizeInMetersY());
}

bool BorderFollowLayer::geometryMatches(
  const nav_msgs::msg::OccupancyGrid & source,
  const nav2_costmap_2d::Costmap2D & master) const
{
  return source.info.width == master.getSizeInCellsX() &&
         source.info.height == master.getSizeInCellsY() &&
         std::abs(source.info.resolution - master.getResolution()) < 1e-6 &&
         std::abs(source.info.origin.position.x - master.getOriginX()) < 1e-6 &&
         std::abs(source.info.origin.position.y - master.getOriginY()) < 1e-6;
}

void BorderFollowLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  const int min_i, const int min_j, const int max_i, const int max_j)
{
  if (!enabled_) {
    return;
  }

  nav_msgs::msg::OccupancyGrid::SharedPtr source;
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    source = source_map_;
  }
  if (!source || !geometryMatches(*source, master_grid)) {
    if (source) {
      RCLCPP_WARN_THROTTLE(
        logger_, *clock_, 2000,
        "Ignoring source map whose geometry differs from the master costmap");
    }
    return;
  }

  const int width = static_cast<int>(source->info.width);
  const int height = static_cast<int>(source->info.height);
  cv::Mat distance_input(height, width, CV_8UC1, cv::Scalar(255));
  bool has_obstacles = false;
  for (int y = 0; y < height; ++y) {
    auto * row = distance_input.ptr<std::uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      const auto value = source->data[y * width + x];
      if (value >= obstacle_threshold_) {
        row[x] = 0;
        has_obstacles = true;
      }
    }
  }
  if (!has_obstacles) {
    return;
  }

  cv::Mat distances_px;
  cv::distanceTransform(
    distance_input, distances_px, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  const double resolution = master_grid.getResolution();
  const int first_x = std::max(0, min_i);
  const int first_y = std::max(0, min_j);
  const int last_x = std::min(width, max_i);
  const int last_y = std::min(height, max_j);
  for (int y = first_y; y < last_y; ++y) {
    for (int x = first_x; x < last_x; ++x) {
      const auto source_value = source->data[y * width + x];
      if (source_value < 0 || source_value >= obstacle_threshold_) {
        continue;
      }
      const auto old_cost = master_grid.getCost(x, y);
      if (old_cost == nav2_costmap_2d::NO_INFORMATION) {
        continue;
      }
      const double distance = distances_px.at<float>(y, x) * resolution;
      const auto new_cost = computeBorderCost(distance, profile_);
      master_grid.setCost(x, y, std::max(old_cost, new_cost));
    }
  }
}

void BorderFollowLayer::reset()
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  source_map_.reset();
  current_ = false;
}

}  // namespace limo_inflation

PLUGINLIB_EXPORT_CLASS(
  limo_inflation::BorderFollowLayer,
  nav2_costmap_2d::Layer)
