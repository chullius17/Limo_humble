#include <gtest/gtest.h>

#include "dwb_core/trajectory_critic.hpp"
#include "nav2_core/controller.hpp"
#include "pluginlib/class_loader.hpp"

TEST(AckermannKinematicsPluginTest, IsDiscoverableByPluginlib)
{
  pluginlib::ClassLoader<dwb_core::TrajectoryCritic> loader(
    "dwb_core", "dwb_core::TrajectoryCritic");

  auto critic = loader.createSharedInstance(
    "limo_dwb_critics::AckermannKinematicsCritic");
  ASSERT_NE(critic, nullptr);
}

TEST(AckermannMPCPluginTest, IsDiscoverableAsNav2Controller)
{
  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");
  auto controller = loader.createSharedInstance("limo_dwb_critics::AckermannMPCController");
  ASSERT_NE(controller, nullptr);
}
