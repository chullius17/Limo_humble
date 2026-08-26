#include <gtest/gtest.h>

#include "limo_inflation/cost_profile.hpp"

using limo_inflation::computeBorderCost;
using limo_inflation::CostProfile;

TEST(CostProfileTest, CreatesZeroCostBand)
{
  const CostProfile profile;
  EXPECT_EQ(computeBorderCost(0.13, profile), 0);
  EXPECT_EQ(computeBorderCost(0.15, profile), 0);
  EXPECT_EQ(computeBorderCost(0.17, profile), 0);
}

TEST(CostProfileTest, DecreasesTowardPreferredBand)
{
  const CostProfile profile;
  EXPECT_EQ(computeBorderCost(0.0, profile), 200);
  EXPECT_GT(computeBorderCost(0.05, profile),
    computeBorderCost(0.10, profile));
}

TEST(CostProfileTest, RisesSlowlyAndSaturatesFarFromObstacle)
{
  const CostProfile profile;
  EXPECT_EQ(computeBorderCost(0.30, profile), 3);
  EXPECT_EQ(computeBorderCost(1.00, profile), 17);
  EXPECT_EQ(computeBorderCost(10.0, profile), 25);
}
