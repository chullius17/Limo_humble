#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "dwb_core/exceptions.hpp"
#include "limo_dwb_critics/ackermann_mpc_controller.hpp"
#include "limo_dwb_critics/mppi_cost_critics.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace
{
geometry_msgs::msg::Pose2D pose(double x, double y, double yaw = 0.0)
{
  geometry_msgs::msg::Pose2D result;
  result.x = x;
  result.y = y;
  result.theta = yaw;
  return result;
}

nav_2d_msgs::msg::Path2D straightPath()
{
  nav_2d_msgs::msg::Path2D result;
  result.header.frame_id = "odom";
  for (int i = 0; i <= 30; ++i) {
    result.poses.push_back(pose(0.05 * i, 0.0));
  }
  return result;
}
}  // namespace

class MppiCostsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    auto options = rclcpp::NodeOptions().arguments(
      {"--ros-args", "--params-file", LIMO_MPC_PARAMS_PATH});
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("controller_server", options);
    costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>("mpc_test_costmap");
    costmap_->set_parameters({
      rclcpp::Parameter("plugins", std::vector<std::string>{}),
      rclcpp::Parameter("global_frame", "odom"),
      rclcpp::Parameter("footprint", "[[-0.161,-0.110],[-0.161,0.110],[0.161,0.110],[0.161,-0.110]]"),
      rclcpp::Parameter("footprint_padding", 0.0)});
    ASSERT_EQ(costmap_->on_configure(rclcpp_lifecycle::State()), nav2_util::CallbackReturn::SUCCESS);
    costmap_->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
    clearMap();
  }

  void TearDown() override
  {
    costmap_->on_cleanup(rclcpp_lifecycle::State());
    costmap_.reset();
    node_.reset();
    rclcpp::shutdown();
  }

  void clearMap()
  {
    auto * map = costmap_->getCostmap();
    std::fill(map->getCharMap(), map->getCharMap() + map->getSizeInCellsX() *
      map->getSizeInCellsY(), nav2_costmap_2d::FREE_SPACE);
  }

  void setCost(double x, double y, unsigned char cost)
  {
    unsigned int mx, my;
    ASSERT_TRUE(costmap_->getCostmap()->worldToMap(x, y, mx, my));
    costmap_->getCostmap()->setCost(mx, my, cost);
  }

  template<typename Critic>
  void initialize(Critic & critic, std::string name)
  {
    std::string parent = "FollowPath";
    critic.initialize(node_, name, parent, costmap_);
  }

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
};

TEST_F(MppiCostsTest, NormalizesObstacleCostIndependentlyOfHorizonAndResolution)
{
  limo_dwb_critics::MppiObstacleCritic critic;
  initialize(critic, "MppiObstacle");
  ASSERT_TRUE(critic.prepare(pose(0, 0), nav_2d_msgs::msg::Twist2D(), pose(2, 0), straightPath()));
  setCost(0.0, 0.0, 127);
  dwb_msgs::msg::Trajectory2D trajectory;
  trajectory.poses.assign(11, pose(0, 0));
  EXPECT_DOUBLE_EQ(critic.getScale(), 3.0);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(trajectory) * critic.getScale(), 1.5);
  trajectory.poses.assign(51, pose(0, 0));
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(trajectory) * critic.getScale(), 1.5);
  costmap_->getCostmap()->resizeMap(50, 50, 0.1, -2.5, -2.5);
  clearMap();
  setCost(0.0, 0.0, 127);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(trajectory) * critic.getScale(), 1.5);
}

TEST_F(MppiCostsTest, NearGoalDisablesRepulsionButNeverCollisionRejection)
{
  limo_dwb_critics::MppiObstacleCritic critic;
  initialize(critic, "MppiObstacle");
  ASSERT_TRUE(critic.prepare(pose(0, 0), nav_2d_msgs::msg::Twist2D(), pose(0.3, 0), straightPath()));
  dwb_msgs::msg::Trajectory2D trajectory;
  trajectory.poses.assign(5, pose(0, 0));
  setCost(0.0, 0.0, 200);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(trajectory), 0.0);
  setCost(0.0, 0.0, nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  EXPECT_NEAR(critic.scoreTrajectory(trajectory), 300.0 / 254.0, 1e-12);
  setCost(0.0, 0.0, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_THROW(critic.scoreTrajectory(trajectory), dwb_core::IllegalTrajectoryException);
  clearMap();
  setCost(0.15, 0.1, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_THROW(critic.scoreTrajectory(trajectory), dwb_core::IllegalTrajectoryException);
  clearMap();
  trajectory.poses.back().x = 3.0;
  EXPECT_THROW(critic.scoreTrajectory(trajectory), dwb_core::IllegalTrajectoryException);
}

TEST_F(MppiCostsTest, UsesHumbleGoalWeightsAndWrappedYawError)
{
  limo_dwb_critics::MppiPathCritic critic;
  initialize(critic, "MppiPath");
  ASSERT_TRUE(critic.prepare(pose(0, 0), nav_2d_msgs::msg::Twist2D(), pose(0.3, 0, -3.1), straightPath()));
  dwb_msgs::msg::Trajectory2D trajectory;
  trajectory.poses = {pose(0, 0), pose(0.1, 0, 3.1), pose(0.2, 0, 3.1)};
  const double wrapped_angle = 2.0 * std::acos(-1.0) - 6.2;
  EXPECT_NEAR(critic.scoreTrajectory(trajectory), 5.0 * 0.15 + 3.0 * wrapped_angle, 1e-12);
}

TEST_F(MppiCostsTest, RewardsProgressAndRelaxesAlignmentWhenPathBlocked)
{
  limo_dwb_critics::MppiPathCritic critic;
  initialize(critic, "MppiPath");
  auto path = straightPath();
  const auto prepare = [&]() {
      return critic.prepare(pose(0, 0), nav_2d_msgs::msg::Twist2D(), pose(2, 0), path);
    };
  ASSERT_TRUE(prepare());
  dwb_msgs::msg::Trajectory2D stopped;
  stopped.poses = {pose(0, 0), pose(0, 0), pose(0, 0)};
  dwb_msgs::msg::Trajectory2D progress;
  progress.poses = {pose(0, 0), pose(0.2, 0), pose(0.4, 0)};
  EXPECT_LT(critic.scoreTrajectory(progress), critic.scoreTrajectory(stopped));
  dwb_msgs::msg::Trajectory2D detour;
  detour.poses = {pose(0, 0), pose(0.2, 0.2), pose(0.4, 0.2)};
  const double clear_path_cost = critic.scoreTrajectory(detour);
  for (int i = 2; i < 20; ++i) {
    setCost(i * 0.05, 0.0, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  ASSERT_TRUE(prepare());
  EXPECT_LT(critic.scoreTrajectory(detour), clear_path_cost);
}

TEST_F(MppiCostsTest, RealYamlConfiguresAndRunsTheFullDwbControllerPipeline)
{
  limo_dwb_critics::AckermannMPCController controller;
  auto tf = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  ASSERT_NO_THROW(controller.configure(node_, "FollowPath", tf, costmap_));
  controller.activate();
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  for (int i = 0; i <= 40; ++i) {
    geometry_msgs::msg::PoseStamped point;
    point.header = path.header;
    point.pose.position.x = i * 0.05;
    point.pose.orientation.w = 1.0;
    path.poses.push_back(point);
  }
  controller.setPlan(path);
  geometry_msgs::msg::PoseStamped current;
  current.header.frame_id = "odom";
  current.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist velocity;
  const auto begin = std::chrono::steady_clock::now();
  const auto command = controller.computeVelocityCommands(current, velocity);
  const auto duration = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - begin).count();
  RecordProperty("full_pipeline_ms", duration);
  EXPECT_GT(command.twist.linear.x, 0.0);
  EXPECT_LE(command.twist.linear.x, 1.3 * 0.05 + 1e-12);
  EXPECT_DOUBLE_EQ(command.twist.linear.y, 0.0);
  EXPECT_LE(std::abs(command.twist.angular.z), command.twist.linear.x / 0.462 + 1e-12);
  controller.deactivate();
  controller.cleanup();
}
