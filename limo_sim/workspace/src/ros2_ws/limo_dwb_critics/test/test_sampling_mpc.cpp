#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "limo_dwb_critics/sampling_mpc.hpp"

using limo_dwb_critics::MpcConfig;
using limo_dwb_critics::MpcControl;
using limo_dwb_critics::MpcRollout;
using limo_dwb_critics::MpcState;
using limo_dwb_critics::SamplingMpc;

TEST(SamplingMpc, RejectsInvalidConfigurationAndState)
{
  MpcConfig config;
  config.steering_rate = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(SamplingMpc model(config), std::invalid_argument);
  config = MpcConfig();
  config.batch_size = 2;
  EXPECT_THROW(SamplingMpc model(config), std::invalid_argument);
  config = MpcConfig();
  config.control_segments = config.time_steps + 1;
  EXPECT_THROW(SamplingMpc model(config), std::invalid_argument);
  SamplingMpc model;
  MpcState state;
  state.x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(model.rollout(state, std::vector<MpcControl>(50)), std::invalid_argument);
  EXPECT_THROW(model.rollout(MpcState(), {}), std::invalid_argument);
}

TEST(SamplingMpc, IntegratesKnownRearAxleCircleAndBaseOffset)
{
  SamplingMpc model;
  const auto & config = model.config();
  for (double speed : {-0.1, 0.4}) {
    MpcState initial;
    initial.control = {speed, std::atan(config.wheelbase / 0.7)};
    const auto result = model.rollout(
      initial, std::vector<MpcControl>(config.time_steps, initial.control));
    const double yaw = speed * config.dt * config.time_steps / 0.7;
    EXPECT_NEAR(result.states.back().yaw, yaw, 1e-10);
    EXPECT_NEAR(result.states.back().x,
      0.7 * std::sin(yaw) + config.rear_axle_to_base * (std::cos(yaw) - 1.0), 1e-10);
    EXPECT_NEAR(result.states.back().y,
      0.7 * (1.0 - std::cos(yaw)) + config.rear_axle_to_base * std::sin(yaw), 1e-10);
  }
}

TEST(SamplingMpc, EnforcesPhysicalLimitsAcrossTimeVaryingSequences)
{
  SamplingMpc model;
  const auto & config = model.config();
  std::mt19937 random(123);
  std::uniform_real_distribution<double> speed(-0.4, 0.9);
  std::uniform_real_distribution<double> steering(-0.8, 0.8);
  for (int sample = 0; sample < 100; ++sample) {
    std::vector<MpcControl> targets;
    for (int t = 0; t < config.time_steps; ++t) {
      targets.push_back({speed(random), steering(random)});
    }
    MpcState initial;
    initial.control = {sample % 2 == 0 ? 0.4 : -0.08, 0.3};
    const auto result = model.rollout(initial, targets);
    ASSERT_EQ(result.states.size(), targets.size() + 1);
    for (std::size_t t = 1; t < result.states.size(); ++t) {
      const auto & before = result.states[t - 1].control;
      const auto & after = result.states[t].control;
      EXPECT_LE(std::abs(std::tan(after.steering) / config.wheelbase),
        1.0 / config.min_turning_radius + 1e-10);
      EXPECT_LE(std::abs(after.steering - before.steering),
        config.steering_rate * config.dt + 1e-10);
      EXPECT_LE(std::abs(after.velocity - before.velocity),
        std::max(config.acceleration, config.deceleration) * config.dt + 1e-10);
      EXPECT_LE(std::abs(model.yawRate(after) - model.yawRate(before)),
        config.yaw_acceleration * config.dt + 1e-10);
      EXPECT_LE(std::abs(model.yawRate(after)), config.max_yaw_rate + 1e-10);
      EXPECT_GE(after.velocity, config.min_velocity - 1e-10);
      EXPECT_LE(after.velocity, config.max_velocity + 1e-10);
      EXPECT_GE(before.velocity * after.velocity, -1e-12);
      if (after.velocity == 0.0) {
        EXPECT_DOUBLE_EQ(model.yawRate(after), 0.0);
      }
    }
  }
}

TEST(SamplingMpc, BrakesBeforeReversingAndRecoversFromOverspeed)
{
  MpcConfig config;
  config.acceleration = 0.4;
  config.deceleration = 0.8;
  SamplingMpc model(config);
  MpcState initial;
  initial.control.velocity = 0.12;
  const auto reverse = model.rollout(
    initial, std::vector<MpcControl>(config.time_steps, {-0.1, 0.0}));
  EXPECT_NEAR(reverse.states[1].control.velocity, 0.08, 1e-12);
  bool stopped = false;
  for (const auto & state : reverse.states) {
    stopped = stopped || std::abs(state.control.velocity) < 1e-12;
    if (state.control.velocity < 0.0) {
      EXPECT_TRUE(stopped);
    }
  }
  EXPECT_NEAR(reverse.states.back().control.velocity, -0.1, 1e-12);
  initial.control.velocity = 0.7;
  const auto brake = model.rollout(initial, std::vector<MpcControl>(config.time_steps));
  EXPECT_NEAR(brake.states[1].control.velocity, 0.66, 1e-12);
  EXPECT_DOUBLE_EQ(brake.states.back().control.velocity, 0.0);
}

TEST(SamplingMpc, ExploresChangingSteeringAndReturnsReachableFirstCommand)
{
  SamplingMpc model;
  bool changing_sequence = false;
  int evaluations = 0;
  const auto solution = model.solve(MpcState(), [&](const MpcRollout & rollout, double) {
      ++evaluations;
      for (std::size_t t = 1; t < rollout.targets.size(); ++t) {
        changing_sequence = changing_sequence ||
          rollout.targets[t].steering != rollout.targets[t - 1].steering;
      }
      const auto & end = rollout.states.back();
      return 10.0 + 100.0 * std::pow(end.x - 0.8, 2) + 100.0 * std::pow(end.y - 0.2, 2);
    });
  EXPECT_TRUE(changing_sequence);
  EXPECT_EQ(evaluations, model.config().batch_size);
  ASSERT_TRUE(std::isfinite(solution.cost));
  EXPECT_TRUE(model.hasWarmStart());
  EXPECT_GT(solution.rollout.states.back().x, 0.5);
  EXPECT_LE(solution.rollout.states[1].control.velocity,
    model.config().acceleration * model.config().dt + 1e-12);
}

TEST(SamplingMpc, ShiftsWinnerAndClearsWarmStartAfterFailure)
{
  SamplingMpc model;
  const auto objective = [](const MpcRollout & rollout, double) {
      return 10.0 + 100.0 * std::pow(rollout.states.back().x - 0.8, 2);
    };
  const auto first = model.solve(MpcState(), objective);
  ASSERT_TRUE(std::isfinite(first.cost));
  int index = 0;
  model.solve(first.rollout.states[1], [&](const MpcRollout & rollout, double) {
      if (index == 1) {  // Candidate 0 brakes; candidate 1 is the shifted winner.
        for (std::size_t t = 0; t < rollout.targets.size(); ++t) {
          const auto & expected = first.rollout.targets[
            std::min(t + 2, first.rollout.targets.size() - 1)];
          EXPECT_DOUBLE_EQ(rollout.targets[t].velocity, expected.velocity);
          EXPECT_DOUBLE_EQ(rollout.targets[t].steering, expected.steering);
        }
      }
      ++index;
      return objective(rollout, 0.0);
    }, 2);
  EXPECT_TRUE(model.hasWarmStart());
  const auto failed = model.solve(MpcState(), [](const MpcRollout &, double) {
      return std::numeric_limits<double>::infinity();
    });
  EXPECT_FALSE(std::isfinite(failed.cost));
  EXPECT_FALSE(model.hasWarmStart());
  model.solve(MpcState(), objective);
  EXPECT_THROW(model.solve(MpcState(), [](const MpcRollout &, double) -> double {
      throw std::runtime_error("evaluation failure");
    }), std::runtime_error);
  EXPECT_FALSE(model.hasWarmStart());
  model.solve(MpcState(), objective);
  model.reset();
  EXPECT_FALSE(model.hasWarmStart());
}

TEST(SamplingMpc, SelectsBrakingWhenMovingPredictionsCollide)
{
  SamplingMpc model;
  MpcState initial;
  initial.control.velocity = 0.3;
  const auto solution = model.solve(initial, [](const MpcRollout & rollout, double) {
      for (const auto & state : rollout.states) {
        if (state.x >= 0.05) {
          return std::numeric_limits<double>::infinity();
        }
      }
      return 100.0 * std::abs(rollout.states.back().control.velocity);
    });
  ASSERT_TRUE(std::isfinite(solution.cost));
  EXPECT_LT(solution.rollout.states[1].control.velocity, initial.control.velocity);
  EXPECT_NEAR(solution.rollout.states.back().control.velocity, 0.0, 1e-12);
}

TEST(SamplingMpc, RecedingHorizonTracksAnSReference)
{
  SamplingMpc model;
  MpcState current;
  const auto reference_y = [](double x) {return 0.18 * std::sin(3.0 * x);};
  const auto reference_yaw = [](double x) {return std::atan(0.54 * std::cos(3.0 * x));};
  double maximum_error = 0.0;
  bool positive_steering = false;
  bool negative_steering = false;
  // Synthetic closed-loop model test, not Gazebo or a hardware validation.
  for (int cycle = 0; cycle < 120; ++cycle) {
    const auto solution = model.solve(current, [&](const MpcRollout & rollout, double) {
        double cost = 0.0;
        for (const auto & state : rollout.states) {
          cost += 30.0 * std::pow(state.y - reference_y(state.x), 2) +
            2.0 * std::pow(state.yaw - reference_yaw(state.x), 2) +
            4.0 * std::pow(state.control.velocity - 0.3, 2);
        }
        return cost;
      });
    ASSERT_TRUE(std::isfinite(solution.cost));
    current = solution.rollout.states[1];
    maximum_error = std::max(maximum_error, std::abs(current.y - reference_y(current.x)));
    positive_steering = positive_steering || current.control.steering > 0.05;
    negative_steering = negative_steering || current.control.steering < -0.05;
  }
  EXPECT_GT(current.x, 1.0);
  EXPECT_LT(maximum_error, 0.15);
  EXPECT_TRUE(positive_steering);
  EXPECT_TRUE(negative_steering);
}
