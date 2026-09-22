#include "limo_dwb_critics/ackermann_kinematics_critic.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "dwb_core/exceptions.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace limo_dwb_critics
{

bool commandRespectsAckermann(
  const double linear_x, const double linear_y, const double angular_z,
  const double min_turning_radius, const double lateral_tolerance,
  const double stopped_velocity)
{
  if (std::abs(linear_y) > lateral_tolerance) {
    return false;
  }
  if (std::abs(angular_z) <= lateral_tolerance) {
    return true;
  }
  if (std::abs(linear_x) <= stopped_velocity) {
    return false;
  }
  return std::abs(linear_x / angular_z) + 1e-9 >= min_turning_radius;
}

void AckermannKinematicsCritic::onInit()
{
  const std::string prefix = dwb_plugin_name_ + "." + name_ + ".";
  const auto declare = [this, &prefix](const std::string & name, const double value) {
      const std::string parameter = prefix + name;
      if (!nh_->has_parameter(parameter)) {
        nh_->declare_parameter(parameter, rclcpp::ParameterValue(value));
      }
    };

  declare("min_turning_radius", min_turning_radius_);
  declare("lateral_tolerance", lateral_tolerance_);
  declare("stopped_velocity", stopped_velocity_);
  nh_->get_parameter(prefix + "min_turning_radius", min_turning_radius_);
  nh_->get_parameter(prefix + "lateral_tolerance", lateral_tolerance_);
  nh_->get_parameter(prefix + "stopped_velocity", stopped_velocity_);

  if (min_turning_radius_ <= 0.0 || lateral_tolerance_ < 0.0 ||
    stopped_velocity_ < 0.0)
  {
    throw std::invalid_argument("Invalid Ackermann kinematics parameters");
  }
}

double AckermannKinematicsCritic::scoreTrajectory(
  const dwb_msgs::msg::Trajectory2D & trajectory)
{
  const auto & velocity = trajectory.velocity;
  if (!commandRespectsAckermann(
      velocity.x, velocity.y, velocity.theta, min_turning_radius_,
      lateral_tolerance_, stopped_velocity_))
  {
    throw dwb_core::IllegalTrajectoryException(
            name_, "Command violates LIMO Ackermann constraints");
  }
  return 0.0;
}

}  // namespace limo_dwb_critics

PLUGINLIB_EXPORT_CLASS(
  limo_dwb_critics::AckermannKinematicsCritic,
  dwb_core::TrajectoryCritic)
