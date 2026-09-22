#include "limo_dwb_critics/sampling_mpc.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace limo_dwb_critics
{
namespace
{
bool finiteControl(const MpcControl & control)
{
  return std::isfinite(control.velocity) && std::isfinite(control.steering);
}

double approach(double value, double target, double amount)
{
  return value + std::clamp(target - value, -amount, amount);
}
}  // namespace

void MpcConfig::validate() const
{
  for (double value : {dt, max_velocity, max_yaw_rate, acceleration, deceleration,
      yaw_acceleration, steering_rate, wheelbase, min_turning_radius,
      velocity_std, steering_std})
  {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument("MPC rates, dimensions and sampling deviations must be positive");
    }
  }
  if (!std::isfinite(min_velocity) || min_velocity > 0.0 ||
    !std::isfinite(rear_axle_to_base) || rear_axle_to_base < 0.0 ||
    !std::isfinite(acceleration_weight) || acceleration_weight < 0.0 ||
    !std::isfinite(steering_weight) || steering_weight < 0.0 ||
    time_steps < 2 || time_steps > 1000 || control_segments < 1 ||
    control_segments > time_steps || velocity_samples < 2 || velocity_samples > 100 ||
    curvature_samples < 3 || curvature_samples > 100 || batch_size > 10000 ||
    batch_size < 2 + velocity_samples * curvature_samples)
  {
    throw std::invalid_argument("Invalid MPC horizon, sample count, limits or weights");
  }
}

SamplingMpc::SamplingMpc(const MpcConfig & config)
: config_(config)
{
  config_.validate();
}

void SamplingMpc::reset()
{
  previous_targets_.clear();
}

double SamplingMpc::yawRate(const MpcControl & control) const
{
  return control.velocity * std::tan(control.steering) / config_.wheelbase;
}

MpcControl SamplingMpc::boundedTarget(const MpcControl & target) const
{
  const double velocity = std::clamp(
    target.velocity, config_.min_velocity, config_.max_velocity);
  double curvature = 1.0 / config_.min_turning_radius;
  if (std::abs(velocity) > 1e-9) {
    curvature = std::min(curvature, config_.max_yaw_rate / std::abs(velocity));
  }
  const double angle = std::atan(config_.wheelbase * curvature);
  // The LIMO Twist driver requests centered steering when commanded to stop.
  return {velocity, velocity == 0.0 ? 0.0 : std::clamp(target.steering, -angle, angle)};
}

MpcControl SamplingMpc::advance(
  const MpcControl & current, const MpcControl & requested) const
{
  if (!finiteControl(current) || !finiteControl(requested)) {
    throw std::invalid_argument("Non-finite MPC control");
  }
  const auto target = boundedTarget(requested);
  // Brake to zero before reversing; never jump across zero in one time step.
  const double desired_velocity = current.velocity * target.velocity < 0.0 ?
    0.0 : target.velocity;
  const double acceleration = std::abs(desired_velocity) > std::abs(current.velocity) ?
    config_.acceleration : config_.deceleration;
  const double next_velocity = approach(
    current.velocity, desired_velocity, acceleration * config_.dt);
  const double next_steering = approach(
    current.steering, target.steering, config_.steering_rate * config_.dt);
  const double dv = next_velocity - current.velocity;
  const double ds = next_steering - current.steering;

  // Bound d(v*tan(delta)/L) along the joint update. Scaling BOTH updates
  // preserves longitudinal/steering limits and the Ackermann relation.
  const double tangent = std::tan(std::max(
      std::abs(current.steering), std::abs(next_steering)));
  const double derivative_bound =
    (std::abs(dv) * tangent + std::max(std::abs(current.velocity),
      std::abs(next_velocity)) * (1.0 + tangent * tangent) * std::abs(ds)) /
    config_.wheelbase;
  double fraction = derivative_bound > 0.0 ?
    std::min(1.0, config_.yaw_acceleration * config_.dt / derivative_bound) : 1.0;
  const auto interpolate = [&](double f) -> MpcControl {
      return {current.velocity + f * dv, current.steering + f * ds};
    };
  // An overspeed observation is allowed to recover gradually, not clipped
  // instantaneously. Nominal states always obey max_yaw_rate.
  const double yaw_limit = std::max(config_.max_yaw_rate, std::abs(yawRate(current)));
  if (std::abs(yawRate(interpolate(fraction))) > yaw_limit) {
    double low = 0.0;
    double high = fraction;
    for (int i = 0; i < 40; ++i) {
      const double mid = (low + high) * 0.5;
      if (std::abs(yawRate(interpolate(mid))) <= yaw_limit) {
        low = mid;
      } else {
        high = mid;
      }
    }
    fraction = low;
  }
  return interpolate(fraction);
}

MpcRollout SamplingMpc::rollout(
  const MpcState & initial, const std::vector<MpcControl> & targets) const
{
  if (targets.size() != static_cast<std::size_t>(config_.time_steps) ||
    !std::isfinite(initial.x) || !std::isfinite(initial.y) ||
    !std::isfinite(initial.yaw) || !finiteControl(initial.control) ||
    std::abs(initial.control.steering) >
    std::atan(config_.wheelbase / config_.min_turning_radius) + 1e-9)
  {
    throw std::invalid_argument("Invalid MPC initial state or sequence length");
  }
  MpcRollout result;
  result.targets = targets;
  result.states.reserve(targets.size() + 1);
  result.states.push_back(initial);
  auto state = initial;
  for (const auto & target : targets) {
    const auto control = advance(state.control, target);
    const double normalized_acceleration = (control.velocity - state.control.velocity) /
      (config_.dt * std::max(config_.acceleration, config_.deceleration));
    const double normalized_steering = (control.steering - state.control.steering) /
      (config_.dt * config_.steering_rate);
    result.effort_cost +=
      config_.acceleration_weight * normalized_acceleration * normalized_acceleration +
      config_.steering_weight * normalized_steering * normalized_steering;

    // Exact constant-twist arc at the rear axle, then translate back to
    // base_link. Commands have no lateral component, but base_link can move
    // laterally relative to its heading because it is ahead of the axle.
    const double yaw_delta = yawRate(control) * config_.dt;
    const double half_delta = yaw_delta * 0.5;
    const double sinc = std::abs(half_delta) < 1e-9 ? 1.0 :
      std::sin(half_delta) / half_delta;
    const double distance = control.velocity * config_.dt * sinc;
    const double next_yaw = state.yaw + yaw_delta;
    state.x += distance * std::cos(state.yaw + half_delta) +
      config_.rear_axle_to_base * (std::cos(next_yaw) - std::cos(state.yaw));
    state.y += distance * std::sin(state.yaw + half_delta) +
      config_.rear_axle_to_base * (std::sin(next_yaw) - std::sin(state.yaw));
    state.yaw = next_yaw;
    state.control = control;
    result.states.push_back(state);
  }
  result.effort_cost /= targets.size();
  return result;
}

MpcSolution SamplingMpc::solve(
  const MpcState & initial,
  const std::function<double(const MpcRollout &, double)> & environment_cost,
  std::size_t shift_steps)
{
  const std::size_t count = static_cast<std::size_t>(config_.time_steps);
  std::vector<MpcControl> nominal(count, boundedTarget(initial.control));
  if (previous_targets_.size() == count) {
    for (std::size_t t = 0; t < count; ++t) {
      nominal[t] = previous_targets_[std::min(t + std::min(shift_steps, count), count - 1)];
    }
  }
  // Warm state is committed only after a successful search, including when
  // a caller's evaluator throws. Failed searches cannot retain an old plan.
  reset();
  MpcSolution best;
  const auto consider = [&](const std::vector<MpcControl> & targets) {
      auto prediction = rollout(initial, targets);
      const double remaining = best.cost - prediction.effort_cost;
      if (remaining < 0.0) {
        return;
      }
      const double cost = environment_cost(prediction, remaining);
      if (std::isfinite(cost) && cost >= 0.0 && cost + prediction.effort_cost < best.cost) {
        best.cost = cost + prediction.effort_cost;
        best.rollout = std::move(prediction);
      }
    };

  consider(std::vector<MpcControl>(count));  // Full-horizon braking candidate.
  consider(nominal);
  // Deterministic constant-curvature seeds cover the whole operating range.
  // The remaining samples vary their targets across control_segments blocks.
  for (int v = 0; v < config_.velocity_samples; ++v) {
    const double velocity = config_.min_velocity +
      (config_.max_velocity - config_.min_velocity) * v / (config_.velocity_samples - 1);
    for (int k = 0; k < config_.curvature_samples; ++k) {
      const double curvature = (-1.0 + 2.0 * k / (config_.curvature_samples - 1)) /
        config_.min_turning_radius;
      consider(std::vector<MpcControl>(count, boundedTarget(
            {velocity, std::atan(config_.wheelbase * curvature)})));
    }
  }
  std::normal_distribution<double> noise(0.0, 1.0);
  std::uniform_real_distribution<double> velocity_distribution(
    config_.min_velocity, config_.max_velocity);
  std::uniform_real_distribution<double> curvature_distribution(
    -1.0 / config_.min_turning_radius, 1.0 / config_.min_turning_radius);
  const int remaining = config_.batch_size - 2 -
    config_.velocity_samples * config_.curvature_samples;
  for (int sample = 0; sample < remaining; ++sample) {
    std::vector<MpcControl> targets(count);
    for (int segment = 0; segment < config_.control_segments; ++segment) {
      const std::size_t begin = segment * count / config_.control_segments;
      const std::size_t end = (segment + 1) * count / config_.control_segments;
      const double velocity_noise = config_.velocity_std * noise(random_);
      const double steering_noise = config_.steering_std * noise(random_);
      const MpcControl broad_sample = boundedTarget({velocity_distribution(random_),
          std::atan(config_.wheelbase * curvature_distribution(random_))});
      for (std::size_t t = begin; t < end; ++t) {
        // Keep global exploration even when the previous winner is trapped.
        targets[t] = sample % 4 == 0 ? broad_sample : boundedTarget({
            nominal[t].velocity + velocity_noise, nominal[t].steering + steering_noise});
      }
    }
    consider(targets);
  }
  if (std::isfinite(best.cost)) {
    previous_targets_ = best.rollout.targets;
  }
  return best;
}

}  // namespace limo_dwb_critics
