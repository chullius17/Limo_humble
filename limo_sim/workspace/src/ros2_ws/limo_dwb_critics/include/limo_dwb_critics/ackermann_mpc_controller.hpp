#ifndef LIMO_DWB_CRITICS__ACKERMANN_MPC_CONTROLLER_HPP_
#define LIMO_DWB_CRITICS__ACKERMANN_MPC_CONTROLLER_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "dwb_core/dwb_local_planner.hpp"
#include "limo_dwb_critics/sampling_mpc.hpp"

namespace limo_dwb_critics
{

class AckermannMPCController : public dwb_core::DWBLocalPlanner
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override;
  void setPlan(const nav_msgs::msg::Path & path) override;
  void deactivate() override;
  void cleanup() override;

protected:
  dwb_msgs::msg::TrajectoryScore coreScoringAlgorithm(
    const geometry_msgs::msg::Pose2D & pose,
    const nav_2d_msgs::msg::Twist2D velocity,
    std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results) override;

  std::unique_ptr<SamplingMpc> mpc_;
  virtual std::int64_t controlTimeNs() const;
  bool open_loop_{true};
  double steering_feedback_min_velocity_{0.05};

private:
  void resetPrediction();
  dwb_msgs::msg::Trajectory2D toTrajectory(const MpcRollout & rollout) const;
  std::int64_t last_time_ns_{0};
  std::int64_t last_ros_time_ns_{0};
  bool have_time_{false};
  bool have_command_{false};
  MpcControl previous_command_;
};

}  // namespace limo_dwb_critics
#endif  // LIMO_DWB_CRITICS__ACKERMANN_MPC_CONTROLLER_HPP_
