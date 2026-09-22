#include <gtest/gtest.h>

#include "limo_dwb_critics/ackermann_kinematics_critic.hpp"

using limo_dwb_critics::commandRespectsAckermann;

TEST(AckermannKinematicsTest, AcceptsStraightForwardAndReverseCommands)
{
  EXPECT_TRUE(commandRespectsAckermann(0.5, 0.0, 0.0, 0.462, 1e-6, 1e-3));
  EXPECT_TRUE(commandRespectsAckermann(-0.1, 0.0, 0.0, 0.462, 1e-6, 1e-3));
}

TEST(AckermannKinematicsTest, RejectsLateralAndRotateInPlaceCommands)
{
  EXPECT_FALSE(commandRespectsAckermann(0.2, 0.01, 0.1, 0.462, 1e-6, 1e-3));
  EXPECT_FALSE(commandRespectsAckermann(0.0, 0.0, 0.5, 0.462, 1e-6, 1e-3));
}

TEST(AckermannKinematicsTest, EnforcesMinimumTurningRadiusBothDirections)
{
  EXPECT_TRUE(commandRespectsAckermann(0.5, 0.0, 1.0, 0.462, 1e-6, 1e-3));
  EXPECT_TRUE(commandRespectsAckermann(-0.5, 0.0, -1.0, 0.462, 1e-6, 1e-3));
  EXPECT_FALSE(commandRespectsAckermann(0.2, 0.0, 0.5, 0.462, 1e-6, 1e-3));
  EXPECT_FALSE(commandRespectsAckermann(-0.2, 0.0, 0.5, 0.462, 1e-6, 1e-3));
}
