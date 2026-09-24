#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "scan_ranges.hpp"

TEST(ScanRanges, ConvertsMissingAndInvalidSamples)
{
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> ranges = {
    0.0f, -1.0f, 0.05f, 12.1f, inf, -inf,
    std::numeric_limits<float>::quiet_NaN(), 0.1f, 1.5f, 12.0f};
  ydlidar_ros2_driver::normalizeInvalidRanges(ranges, 0.1f, 12.0f, true);
  ASSERT_EQ(ranges.size(), 10u);
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(ranges[i], inf);
  }
  EXPECT_FLOAT_EQ(ranges[7], 0.1f);
  EXPECT_FLOAT_EQ(ranges[8], 1.5f);
  EXPECT_FLOAT_EQ(ranges[9], 12.0f);
}

TEST(ScanRanges, PreservesLegacyRepresentationWhenDisabled)
{
  std::vector<float> ranges = {0.0f, 0.05f, 1.5f, 12.1f};
  const auto original = ranges;
  ydlidar_ros2_driver::normalizeInvalidRanges(ranges, 0.1f, 12.0f, false);
  EXPECT_EQ(ranges, original);
}

TEST(ScanRanges, UnfilledBinsNeverBecomeZeroRangeObstacles)
{
  std::vector<float> ranges(410, 0.0f);
  ranges[25] = 2.0f;
  ydlidar_ros2_driver::normalizeInvalidRanges(ranges, 0.0f, 12.0f, true);
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (i == 25) {
      EXPECT_FLOAT_EQ(ranges[i], 2.0f);
    } else {
      EXPECT_EQ(ranges[i], std::numeric_limits<float>::infinity());
    }
  }
}
