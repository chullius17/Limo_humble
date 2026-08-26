#include "limo_inflation/cost_profile.hpp"

#include <algorithm>
#include <cmath>

namespace limo_inflation
{

std::uint8_t computeBorderCost(
  const double distance, const CostProfile & profile)
{
  const double lower = profile.target_distance - profile.tolerance;
  const double upper = profile.target_distance + profile.tolerance;

  if (distance < lower) {
    const double ratio = std::clamp(distance / lower, 0.0, 1.0);
    return static_cast<std::uint8_t>(std::lround(
      static_cast<double>(profile.near_max_cost) * (1.0 - ratio)));
  }
  if (distance <= upper) {
    return 0;
  }

  const double far_cost = profile.far_cost_slope * (distance - upper);
  return static_cast<std::uint8_t>(std::lround(std::clamp(
    far_cost, 0.0, static_cast<double>(profile.far_max_cost))));
}

}  // namespace limo_inflation
