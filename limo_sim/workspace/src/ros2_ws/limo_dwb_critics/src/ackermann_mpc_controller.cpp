#include "limo_dwb_critics/ackermann_mpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dwb_core/exceptions.hpp"
#include "dwb_core/illegal_trajectory_tracker.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace limo_dwb_critics
{

void AckermannMPCController::configure(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  // Reuse DWB lifecycle, plan transformation, critic preparation, costmap
  // locking and visualization. Its standard generator is not used by our
  // overridden coreScoringAlgorithm.
  DWBLocalPlanner::configure(node, name, tf, costmap_ros);
  const auto read_double = [&](const std::string & suffix, double value) {
      const auto parameter = name + "." + suffix;
      if (!node->has_parameter(parameter)) {
        node->declare_parameter(parameter, rclcpp::ParameterValue(value));
      }
      return node->get_parameter(parameter).as_double();
    };
  const auto read_int = [&](const std::string & suffix, int value) {
      const auto parameter = name + ".MPC." + suffix;
      if (!node->has_parameter(parameter)) {
        node->declare_parameter(parameter, rclcpp::ParameterValue(value));
      }
      const auto result = node->get_parameter(parameter).as_int();
      if (result < 0 || result > 10000) {
        throw std::invalid_argument("MPC integer parameter out of range: " + parameter);
      }
      return static_cast<int>(result);
    };

  MpcConfig config;
  config.dt = read_double("MPC.model_dt", config.dt);
  config.time_steps = read_int("time_steps", config.time_steps);
  config.control_segments = read_int("control_segments", config.control_segments);
  config.batch_size = read_int("batch_size", config.batch_size);
  config.velocity_samples = read_int("velocity_samples", config.velocity_samples);
  config.curvature_samples = read_int("curvature_samples", config.curvature_samples);
  config.min_velocity = read_double("min_vel_x", config.min_velocity);
  config.max_velocity = read_double("max_vel_x", config.max_velocity);
  config.max_yaw_rate = read_double("max_vel_theta", config.max_yaw_rate);
  config.acceleration = read_double("acc_lim_x", config.acceleration);
  config.deceleration = -read_double("decel_lim_x", -config.deceleration);
  // Use the more restrictive angular limit in both directions, including
  // braking and sign changes. The model enforces it jointly with steering.
  const double angular_acceleration = read_double("acc_lim_theta", config.yaw_acceleration);
  const double angular_deceleration = -read_double("decel_lim_theta", -config.yaw_acceleration);
  if (!std::isfinite(angular_acceleration) || !std::isfinite(angular_deceleration)) {
    throw std::invalid_argument("Non-finite angular acceleration limits");
  }
  config.yaw_acceleration = std::min(angular_acceleration, angular_deceleration);
  config.wheelbase = read_double("MPC.wheelbase", config.wheelbase);
  config.rear_axle_to_base = read_double("MPC.rear_axle_to_base", config.rear_axle_to_base);
  config.min_turning_radius = read_double(
    "AckermannKinematics.min_turning_radius", config.min_turning_radius);
  config.steering_rate = read_double("MPC.max_steering_rate", config.steering_rate);
  config.velocity_std = read_double("MPC.velocity_std", config.velocity_std);
  config.steering_std = read_double("MPC.steering_std", config.steering_std);
  config.acceleration_weight = read_double("MPC.acceleration_weight", config.acceleration_weight);
  config.steering_weight = read_double("MPC.steering_weight", config.steering_weight);
  steering_feedback_min_velocity_ = read_double(
    "MPC.steering_feedback_min_velocity", steering_feedback_min_velocity_);
  if (!std::isfinite(steering_feedback_min_velocity_) || steering_feedback_min_velocity_ <= 0.0) {
    throw std::invalid_argument("MPC.steering_feedback_min_velocity must be positive");
  }
  const auto feedback_parameter = name + ".MPC.feedback";
  if (!node->has_parameter(feedback_parameter)) {
    node->declare_parameter(feedback_parameter, rclcpp::ParameterValue("OPEN_LOOP"));
  }
  const auto feedback = node->get_parameter(feedback_parameter).as_string();
  if (feedback != "OPEN_LOOP" && feedback != "CLOSED_LOOP") {
    throw std::invalid_argument("MPC.feedback must be OPEN_LOOP or CLOSED_LOOP");
  }
  open_loop_ = feedback == "OPEN_LOOP";
  double frequency = 20.0;
  node->get_parameter_or("controller_frequency", frequency, frequency);
  if (!std::isfinite(frequency) || frequency <= 0.0 ||
    std::abs(config.dt * frequency - 1.0) > 1e-6)
  {
    throw std::invalid_argument("MPC.model_dt must equal 1 / controller_frequency");
  }
  mpc_ = std::make_unique<SamplingMpc>(config);
  resetPrediction();
}

void AckermannMPCController::resetPrediction()
{
  if (mpc_) {
    mpc_->reset();
  }
  have_time_ = false;
  have_command_ = false;
  previous_command_ = MpcControl();
}

std::int64_t AckermannMPCController::controlTimeNs() const
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

void AckermannMPCController::setPlan(const nav_msgs::msg::Path & path)
{
  resetPrediction();
  DWBLocalPlanner::setPlan(path);
}

void AckermannMPCController::deactivate()
{
  resetPrediction();
  DWBLocalPlanner::deactivate();
}

void AckermannMPCController::cleanup()
{
  resetPrediction();
  mpc_.reset();
  DWBLocalPlanner::cleanup();
}

dwb_msgs::msg::Trajectory2D AckermannMPCController::toTrajectory(
  const MpcRollout & rollout) const
{
  dwb_msgs::msg::Trajectory2D trajectory;
  // DWB publishes best.traj.velocity: this MUST be the first reachable
  // command, never the terminal velocity or the sampled target.
  const auto & first = rollout.states.at(1).control;
  trajectory.velocity.x = first.velocity;
  trajectory.velocity.y = 0.0;
  trajectory.velocity.theta = mpc_->yawRate(first);
  trajectory.poses.reserve(rollout.states.size());
  trajectory.time_offsets.reserve(rollout.states.size());
  for (std::size_t t = 0; t < rollout.states.size(); ++t) {
    const auto & state = rollout.states[t];
    geometry_msgs::msg::Pose2D pose;
    pose.x = state.x;
    pose.y = state.y;
    pose.theta = state.yaw;
    trajectory.poses.push_back(pose);
    trajectory.time_offsets.push_back(rclcpp::Duration::from_seconds(t * mpc_->config().dt));
  }
  return trajectory;
}

dwb_msgs::msg::TrajectoryScore AckermannMPCController::coreScoringAlgorithm(
  const geometry_msgs::msg::Pose2D & pose,
  const nav_2d_msgs::msg::Twist2D velocity,
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results)
{
  if (!mpc_) {
    throw nav2_core::PlannerException("Ackermann MPC is not configured");
  }
  if (!std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
    !std::isfinite(velocity.theta))
  {
    resetPrediction();
    throw nav2_core::PlannerException("Non-finite odometry velocity");
  }
  const auto & config = mpc_->config();
  // Foxy's controller loop runs at wall rate. Gazebo /clock may publish at
  // only 10 Hz while control runs at 20 Hz: identical ROS timestamps are not
  // a restart and must not erase the acceleration ramp every other cycle.
  const auto now = controlTimeNs();
  const auto ros_now = node_->now().nanoseconds();
  std::size_t shift = 1;
  if (have_time_) {
    const double elapsed = (now - last_time_ns_) * 1e-9;
    if (elapsed < 0.0 || elapsed > 3.0 * config.dt || ros_now < last_ros_time_ns_) {
      resetPrediction();
    } else {
      shift = static_cast<std::size_t>(std::max(1L, std::lround(elapsed / config.dt)));
    }
  }
  last_time_ns_ = now;
  last_ros_time_ns_ = ros_now;
  have_time_ = true;

  MpcState initial;
  initial.x = pose.x;
  initial.y = pose.y;
  initial.yaw = pose.theta;
  initial.control.velocity = velocity.x;
  // A ratio of noisy yaw rate and near-zero velocity is not steering feedback.
  // Retain the last steering command below the estimation threshold instead.
  const double max_angle = std::atan(config.wheelbase / config.min_turning_radius);
  initial.control.steering = std::abs(velocity.x) >= steering_feedback_min_velocity_ ? std::clamp(
    std::atan(config.wheelbase * velocity.theta / velocity.x), -max_angle, max_angle) :
    (have_command_ ? previous_command_.steering : 0.0);
  if (open_loop_ && have_command_) {
    // Same command-state convention as Humble's OPEN_LOOP velocity smoother:
    // actuator lag must not repeatedly reset v_cmd to v_odom + a*dt. Pose is
    // still measured, so progress and obstacle costs use the actual position.
    initial.control = previous_command_;
  }

  dwb_core::IllegalTrajectoryTracker tracker;
  double best_debug = std::numeric_limits<double>::infinity();
  double worst_debug = -1.0;
  const auto evaluate = [&](const MpcRollout & rollout, double remaining) {
      const auto trajectory = toTrajectory(rollout);
      try {
        auto score = scoreTrajectory(trajectory, std::isfinite(remaining) ? remaining : -1.0);
        tracker.addLegalTrajectory();
        const double environment_cost = score.total;
        dwb_msgs::msg::CriticScore effort;
        effort.name = "MpcEffort";
        effort.scale = 1.0;
        effort.raw_score = rollout.effort_cost;
        score.scores.push_back(effort);
        score.total += rollout.effort_cost;
        if (results) {
          if (score.total < best_debug) {
            best_debug = score.total;
            results->best_index = results->twists.size();
          }
          if (score.total > worst_debug) {
            worst_debug = score.total;
            results->worst_index = results->twists.size();
          }
          results->twists.push_back(score);
        }
        return environment_cost;
      } catch (const dwb_core::IllegalTrajectoryException & error) {
        tracker.addIllegalTrajectory(error);
        if (results) {
          dwb_msgs::msg::TrajectoryScore failed;
          failed.traj = trajectory;
          failed.total = -1.0;
          dwb_msgs::msg::CriticScore reason;
          reason.name = error.getCriticName();
          reason.raw_score = -1.0;
          failed.scores.push_back(reason);
          results->twists.push_back(failed);
        }
        return std::numeric_limits<double>::infinity();
      }
    };
  try {
    const auto solution = mpc_->solve(initial, evaluate, shift);
    if (!std::isfinite(solution.cost)) {
      if (debug_trajectory_details_) {
        RCLCPP_ERROR(node_->get_logger(), "%s", tracker.getMessage().c_str());
      }
      throw dwb_core::NoLegalTrajectoriesException(tracker);
    }
    // Score the winner fully: debug short-circuiting may omit later critics
    // only for candidates that cannot win.
    auto best = scoreTrajectory(toTrajectory(solution.rollout), -1.0);
    dwb_msgs::msg::CriticScore effort;
    effort.name = "MpcEffort";
    effort.scale = 1.0;
    effort.raw_score = solution.rollout.effort_cost;
    best.scores.push_back(effort);
    best.total += solution.rollout.effort_cost;
    previous_command_ = solution.rollout.states.at(1).control;
    have_command_ = true;
    return best;
  } catch (...) {
    resetPrediction();
    throw;
  }
}

}  // namespace limo_dwb_critics

PLUGINLIB_EXPORT_CLASS(limo_dwb_critics::AckermannMPCController, nav2_core::Controller)
