#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "dwb_core/exceptions.hpp"
#include "dwb_core/illegal_trajectory_tracker.hpp"
#include "limo_dwb_critics/ackermann_mpc_controller.hpp"

class ControllerHarness : public limo_dwb_critics::AckermannMPCController
{
public:
  using AckermannMPCController::coreScoringAlgorithm;

  ControllerHarness()
  {
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("mpc_test");
    mpc_ = std::make_unique<limo_dwb_critics::SamplingMpc>();
    debug_trajectory_details_ = false;
  }

  bool reject{false};
  bool warm() const {return mpc_->hasWarmStart();}
  void advanceTime(double seconds = 0.05) {time_ns_ += static_cast<std::int64_t>(seconds * 1e9);}
  void closedLoop() {open_loop_ = false;}
  void freezeRosClock() {node_->set_parameter(rclcpp::Parameter("use_sim_time", true));}
  std::int64_t controlTimeNs() const override {return time_ns_;}

  dwb_msgs::msg::TrajectoryScore scoreTrajectory(
    const dwb_msgs::msg::Trajectory2D & trajectory, double) override
  {
    if (reject) {
      throw dwb_core::IllegalTrajectoryException("test_obstacle", "Blocked");
    }
    dwb_msgs::msg::TrajectoryScore score;
    score.traj = trajectory;
    score.total = 10.0 + 100.0 * std::pow(trajectory.poses.back().x - 0.8, 2);
    return score;
  }

private:
  std::int64_t time_ns_{1000000000};
};

class MpcControllerTest : public ::testing::Test
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

TEST_F(MpcControllerTest, ReturnsFirstReachableCommandWithCorrectTrajectoryTimes)
{
  ControllerHarness controller;
  auto results = std::make_shared<dwb_msgs::msg::LocalPlanEvaluation>();
  const auto best = controller.coreScoringAlgorithm(
    geometry_msgs::msg::Pose2D(), nav_2d_msgs::msg::Twist2D(), results);
  EXPECT_GT(best.traj.velocity.x, 0.0);
  EXPECT_LE(best.traj.velocity.x, 1.3 * 0.05 + 1e-12);
  EXPECT_DOUBLE_EQ(best.traj.velocity.y, 0.0);
  EXPECT_LE(std::abs(best.traj.velocity.theta), std::abs(best.traj.velocity.x) / 0.462 + 1e-12);
  ASSERT_EQ(best.traj.poses.size(), 51U);
  ASSERT_EQ(best.traj.time_offsets.size(), best.traj.poses.size());
  EXPECT_DOUBLE_EQ(rclcpp::Duration(best.traj.time_offsets.front()).seconds(), 0.0);
  EXPECT_DOUBLE_EQ(rclcpp::Duration(best.traj.time_offsets.back()).seconds(), 2.5);
  for (std::size_t i = 1; i < best.traj.time_offsets.size(); ++i) {
    EXPECT_GT(rclcpp::Duration(best.traj.time_offsets[i]).nanoseconds(),
      rclcpp::Duration(best.traj.time_offsets[i - 1]).nanoseconds());
  }
  ASSERT_FALSE(results->twists.empty());
  ASSERT_LT(results->best_index, results->twists.size());
  EXPECT_DOUBLE_EQ(results->twists[results->best_index].total, best.total);
  EXPECT_EQ(best.scores.back().name, "MpcEffort");
  EXPECT_TRUE(controller.warm());
}

TEST_F(MpcControllerTest, ThrowsAndDiscardsOldSequenceIfAllTrajectoriesFail)
{
  ControllerHarness controller;
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> results;
  controller.coreScoringAlgorithm(
    geometry_msgs::msg::Pose2D(), nav_2d_msgs::msg::Twist2D(), results);
  ASSERT_TRUE(controller.warm());
  controller.reject = true;
  EXPECT_THROW(controller.coreScoringAlgorithm(
      geometry_msgs::msg::Pose2D(), nav_2d_msgs::msg::Twist2D(), results),
    dwb_core::NoLegalTrajectoriesException);
  EXPECT_FALSE(controller.warm());
}

TEST_F(MpcControllerTest, CommandRampSurvivesActuatorLagAndRepeatedRosTimestamps)
{
  ControllerHarness controller;
  controller.freezeRosClock();
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> results;
  nav_2d_msgs::msg::Twist2D odometry;
  double previous_speed = 0.0;
  double previous_steering = 0.0;
  double maximum_speed = 0.0;
  // Reproduce the report: odometry stays near zero while the motor cannot
  // get moving on a single 0.065 m/s acceleration increment.
  for (int cycle = 0; cycle < 10; ++cycle) {
    odometry.x = 0.002;
    odometry.theta = cycle % 2 == 0 ? 0.03 : -0.03;
    const auto best = controller.coreScoringAlgorithm(
      geometry_msgs::msg::Pose2D(), odometry, results);
    const auto & command = best.traj.velocity;
    const double steering = std::abs(command.x) > 1e-9 ?
      std::atan(0.2 * command.theta / command.x) : 0.0;
    EXPECT_LE(std::abs(command.x - previous_speed), 0.067 + 1e-9);
    EXPECT_LE(std::abs(steering - previous_steering), 0.8 * 0.05 + 1e-9);
    previous_speed = command.x;
    previous_steering = steering;
    maximum_speed = std::max(maximum_speed, command.x);
    controller.advanceTime();
  }
  EXPECT_GT(maximum_speed, 0.2);
  // An interruption resets the estimate, so resuming cannot reuse a fast
  // stale command when the measured robot is stopped.
  controller.advanceTime(1.0);
  const auto restarted = controller.coreScoringAlgorithm(
    geometry_msgs::msg::Pose2D(), odometry, results);
  EXPECT_LE(restarted.traj.velocity.x, 0.067 + 1e-9);
}

TEST_F(MpcControllerTest, ClosedLoopOptionContinuesToUseMeasuredVelocity)
{
  ControllerHarness controller;
  controller.closedLoop();
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> results;
  nav_2d_msgs::msg::Twist2D odometry;
  odometry.x = 0.002;
  odometry.theta = 0.03;
  for (int cycle = 0; cycle < 4; ++cycle) {
    const auto best = controller.coreScoringAlgorithm(
      geometry_msgs::msg::Pose2D(), odometry, results);
    EXPECT_LE(best.traj.velocity.x, 0.067 + 1e-9);
    controller.advanceTime();
  }
}
