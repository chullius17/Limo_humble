#ifndef LIMO_INFLATION__COST_PROFILE_HPP_
#define LIMO_INFLATION__COST_PROFILE_HPP_

#include <cstdint>

namespace limo_inflation
{

struct CostProfile
{
  double target_distance{0.15};
  double second_lane_distance{0.35};
  double tolerance{0.02};
  std::uint8_t near_max_cost{60};
  std::uint8_t inter_lane_peak_cost{15};
  double far_cost_slope{5.0};
  std::uint8_t far_max_cost{8};
};

std::uint8_t computeBorderCost(double distance, const CostProfile & profile);

}  // namespace limo_inflation

#endif  // LIMO_INFLATION__COST_PROFILE_HPP_
