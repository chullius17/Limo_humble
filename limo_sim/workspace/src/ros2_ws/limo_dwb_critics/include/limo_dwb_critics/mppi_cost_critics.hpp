#ifndef LIMO_DWB_CRITICS__MPPI_COST_CRITICS_HPP_
#define LIMO_DWB_CRITICS__MPPI_COST_CRITICS_HPP_

#include <cstddef>
#include <vector>

#include "dwb_core/trajectory_critic.hpp"
#include "dwb_critics/obstacle_footprint.hpp"

namespace limo_dwb_critics
{

// MPPI-style mean normalized center cost, with Foxy's footprint collision
// rejection at every pose. getScale deliberately bypasses DWB's resolution
// multiplier: normalization is explicit and independent of map resolution.
class MppiObstacleCritic : public dwb_critics::ObstacleFootprintCritic
{
public:
  void onInit() override;
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & velocity,
    const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D & path) override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory) override;
  double getScale() const override {return scale_;}

protected:
  double critical_cost_{300.0};
  double near_goal_distance_{0.5};
  bool near_goal_{false};
};

// Distances in meters and angles in radians, averaged over predicted poses.
// Goal and path terms have the same weights and distance gates as the LIMO
// Humble MPPI profile. Path following uses a fixed metric lookahead instead of
// MPPI's batch-wide furthest-trajectory index, which DWB critics do not expose.
class MppiPathCritic : public dwb_core::TrajectoryCritic
{
public:
  void onInit() override;
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & velocity,
    const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D & path) override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory) override;

protected:
  double goal_weight_{5.0};
  double goal_angle_weight_{3.0};
  double align_weight_{10.0};
  double follow_weight_{5.0};
  double angle_weight_{2.0};
  double goal_distance_{1.0};
  double goal_angle_distance_{0.5};
  double align_distance_{0.5};
  double follow_distance_{1.0};
  double angle_distance_{0.5};
  double max_path_occupancy_ratio_{0.15};
  double lookahead_distance_{1.25};
  double max_angle_to_furthest_{1.0};
  bool use_path_orientations_{true};
  bool forward_preference_{false};
  geometry_msgs::msg::Pose2D goal_;
  std::vector<geometry_msgs::msg::Pose2D> path_;
  std::vector<double> path_lengths_;
  std::size_t start_index_{0};
  std::size_t target_index_{0};
  double distance_to_goal_{0.0};
  bool path_blocked_{false};
  bool apply_path_angle_{false};
};

}  // namespace limo_dwb_critics
#endif  // LIMO_DWB_CRITICS__MPPI_COST_CRITICS_HPP_
