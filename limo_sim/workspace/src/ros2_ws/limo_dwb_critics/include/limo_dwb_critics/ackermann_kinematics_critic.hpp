#ifndef LIMO_DWB_CRITICS__ACKERMANN_KINEMATICS_CRITIC_HPP_
#define LIMO_DWB_CRITICS__ACKERMANN_KINEMATICS_CRITIC_HPP_

#include "dwb_core/trajectory_critic.hpp"

namespace limo_dwb_critics
{

bool commandRespectsAckermann(
  double linear_x, double linear_y, double angular_z,
  double min_turning_radius, double lateral_tolerance,
  double stopped_velocity);

class AckermannKinematicsCritic : public dwb_core::TrajectoryCritic
{
public:
  void onInit() override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory) override;

private:
  double min_turning_radius_{0.462};
  double lateral_tolerance_{1e-6};
  double stopped_velocity_{1e-3};
};

}  // namespace limo_dwb_critics

#endif  // LIMO_DWB_CRITICS__ACKERMANN_KINEMATICS_CRITIC_HPP_
