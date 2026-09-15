#include "limo_inflation/cost_profile.hpp"

#include <algorithm>
#include <cmath>

namespace limo_inflation
{

std::uint8_t computeBorderCost(
  const double distance, const CostProfile & profile)
{
  const double first_lower = profile.target_distance - profile.tolerance;
  const double first_upper = profile.target_distance + profile.tolerance;
  const double second_lower =
    profile.second_lane_distance - profile.tolerance;
  const double second_upper =
    profile.second_lane_distance + profile.tolerance;

  if (distance < first_lower) {
    const double ratio = std::clamp(distance / first_lower, 0.0, 1.0);
    return static_cast<std::uint8_t>(std::lround(
      static_cast<double>(profile.near_max_cost) * (1.0 - ratio)));
  }
  if (distance <= first_upper) {
    return 0;
  }
  if (distance < second_lower) {
    // A bounded triangular ridge separates the two preferred lanes.
    const double midpoint = 0.5 * (first_upper + second_lower);
    const double half_span = 0.5 * (second_lower - first_upper);
    const double normalized = 1.0 - std::abs(distance - midpoint) / half_span;
    return static_cast<std::uint8_t>(std::lround(
      static_cast<double>(profile.inter_lane_peak_cost) *
      std::clamp(normalized, 0.0, 1.0)));
  }
  if (distance <= second_upper) {
    return 0;
  }

  const double far_cost = profile.far_cost_slope * (distance - second_upper);
  return static_cast<std::uint8_t>(std::lround(std::clamp(
    far_cost, 0.0, static_cast<double>(profile.far_max_cost))));
}

}  // namespace limo_inflation
