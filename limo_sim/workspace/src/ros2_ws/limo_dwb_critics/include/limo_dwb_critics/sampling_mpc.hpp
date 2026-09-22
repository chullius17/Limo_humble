#ifndef LIMO_DWB_CRITICS__SAMPLING_MPC_HPP_
#define LIMO_DWB_CRITICS__SAMPLING_MPC_HPP_

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace limo_dwb_critics
{

struct MpcConfig
{
  double dt{0.05};
  int time_steps{50};
  int control_segments{5};
  int batch_size{768};
  int velocity_samples{8};
  int curvature_samples{15};
  double min_velocity{-0.10};
  double max_velocity{0.50};
  double max_yaw_rate{1.10};
  double acceleration{1.3};
  double deceleration{1.3};
  double yaw_acceleration{3.4};
  double steering_rate{0.8};
  double wheelbase{0.20};
  double min_turning_radius{0.462};
  double rear_axle_to_base{0.10};
  double velocity_std{0.12};
  double steering_std{0.15};
  double acceleration_weight{0.5};
  double steering_weight{0.5};
  double steering_command_weight{0.2};
  double steering_rate_change_weight{0.1};

  void validate() const;
};

// Velocity is longitudinal velocity at the rear axle; steering is the
// equivalent bicycle steering angle, not the inner wheel's angle.
struct MpcControl
{
  double velocity{0.0};
  double steering{0.0};
};

struct MpcState
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  MpcControl control;
};

struct MpcRollout
{
  std::vector<MpcState> states;  // Initial state followed by N predicted states.
  std::vector<MpcControl> targets;
  double effort_cost{0.0};
};

struct MpcSolution
{
  MpcRollout rollout;
  double cost{std::numeric_limits<double>::infinity()};
};

// Finite-sample nonlinear shooting MPC. The caller supplies the environment
// cost (DWB critics in the ROS adapter); all actuation constraints live here.
class SamplingMpc
{
public:
  explicit SamplingMpc(const MpcConfig & config = MpcConfig());
  const MpcConfig & config() const {return config_;}
  void reset();
  bool hasWarmStart() const {return !previous_targets_.empty();}
  double yawRate(const MpcControl & control) const;
  MpcControl advance(const MpcControl & current, const MpcControl & target) const;
  MpcRollout rollout(
    const MpcState & initial, const std::vector<MpcControl> & targets,
    double previous_steering_rate = 0.0) const;
  MpcSolution solve(
    const MpcState & initial,
    const std::function<double(const MpcRollout &, double)> & environment_cost,
    std::size_t shift_steps = 1);

private:
  MpcControl boundedTarget(const MpcControl & target) const;
  MpcConfig config_;
  std::vector<MpcControl> previous_targets_;
  double previous_steering_rate_{0.0};
};

}  // namespace limo_dwb_critics
#endif  // LIMO_DWB_CRITICS__SAMPLING_MPC_HPP_
