#ifndef LIMO_INFLATION__BORDER_FOLLOW_LAYER_HPP_
#define LIMO_INFLATION__BORDER_FOLLOW_LAYER_HPP_

#include <mutex>
#include <string>
#include <vector>

#include "limo_inflation/cost_profile.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"

namespace limo_inflation
{

class BorderFollowLayer : public nav2_costmap_2d::Layer
{
public:
  BorderFollowLayer() = default;
  ~BorderFollowLayer() override = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return false;}

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  bool geometryMatches(
    const nav_msgs::msg::OccupancyGrid & source,
    const nav2_costmap_2d::Costmap2D & master) const;

  std::mutex map_mutex_;
  nav_msgs::msg::OccupancyGrid::SharedPtr source_map_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  std::string source_topic_;
  int obstacle_threshold_{10};
  CostProfile profile_;
};

}  // namespace limo_inflation

#endif  // LIMO_INFLATION__BORDER_FOLLOW_LAYER_HPP_
