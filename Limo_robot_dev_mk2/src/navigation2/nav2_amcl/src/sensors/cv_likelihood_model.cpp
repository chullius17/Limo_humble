// Copyright (c) 2026 Giulio Cataldo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_amcl/sensors/cv/cv_likelihood_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace nav2_amcl
{

CvLikelihoodModel::CvLikelihoodModel(const Parameters & parameters)
: parameters_(parameters)
{
}

CvLikelihoodModel::~CvLikelihoodModel()
{
  // map_alloc() uses C allocation internally, so release it with map_free().
  if (map_ != nullptr) {
    map_free(map_);
  }
}

bool CvLikelihoodModel::setMap(const nav_msgs::msg::OccupancyGrid & map_msg)
{
  // Reject malformed maps before allocating a replacement for the current map.
  const auto width = static_cast<int>(map_msg.info.width);
  const auto height = static_cast<int>(map_msg.info.height);
  const auto expected_size = static_cast<std::size_t>(width) *
    static_cast<std::size_t>(height);
  if (
    width <= 0 || height <= 0 || map_msg.info.resolution <= 0.0 ||
    map_msg.data.size() != expected_size)
  {
    return false;
  }

  // Build a private AMCL map containing only semantic occupancy state.
  map_t * new_map = map_alloc();
  new_map->size_x = width;
  new_map->size_y = height;
  new_map->scale = map_msg.info.resolution;
  new_map->cells = static_cast<map_cell_t *>(
    std::malloc(sizeof(map_cell_t) * expected_size));
  if (new_map->cells == nullptr) {
    map_free(new_map);
    return false;
  }

  // Unknown and free cells are both non-obstacles, matching the Python
  // occupancy >= threshold mask used by cv_amcl_debug.
  for (std::size_t index = 0; index < expected_size; ++index) {
    new_map->cells[index].occ_state =
      map_msg.data[index] >= parameters_.occupied_threshold ? 1 : -1;
  }
  // Replace the previous map only after construction has succeeded.
  if (map_ != nullptr) {
    map_free(map_);
  }
  map_ = new_map;
  map_origin_x_ = map_msg.info.origin.position.x;
  map_origin_y_ = map_msg.info.origin.position.y;

  // OccupancyGrid origins may be rotated. Cache their yaw so scoreSad() can use
  // the inverse origin transform when converting world points to grid cells.
  const auto & orientation = map_msg.info.origin.orientation;
  const double yaw = std::atan2(
    2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
    1.0 - 2.0 *
    (orientation.y * orientation.y + orientation.z * orientation.z));
  map_origin_cos_ = std::cos(yaw);
  map_origin_sin_ = std::sin(yaw);
  return true;
}

bool CvLikelihoodModel::ready() const
{
  return map_ != nullptr;
}

CvLikelihoodModel::SadScoreResult CvLikelihoodModel::scoreSad(
  const pf_sample_set_t * set,
  const std::vector<CvTemplateCell2D> & cells) const
{
  if (!ready() || set == nullptr || cells.empty()) {
    return {};
  }

  SadScoreResult result;
  const auto sample_count = static_cast<std::size_t>(set->sample_count);
  result.normalized_sad.resize(sample_count, 0.0);

  // Accumulate foreground confidence once. Background cells are deliberately
  // ignored because this semantic detector does not distinguish a confident
  // negative from a missed or unavailable classification.
  for (const auto & cell : cells) {
    if (std::isfinite(cell.occupancy)) {
      const double occupancy = std::clamp(cell.occupancy, 0.0, 1.0);
      result.positive_mass += occupancy;
    }
  }
  if (result.positive_mass <= 0.0) {
    return result;
  }

  for (int sample_index = 0; sample_index < set->sample_count; ++sample_index) {
    const auto & sample = set->samples[sample_index];
    const double cos_yaw = std::cos(sample.pose.v[2]);
    const double sin_yaw = std::sin(sample.pose.v[2]);
    double false_positive_sum = 0.0;

    for (const auto & cell : cells) {
      const double local_positive = std::clamp(cell.occupancy, 0.0, 1.0);
      if (local_positive <= 0.0) {
        continue;
      }
      const double world_x = sample.pose.v[0] + cos_yaw * cell.x - sin_yaw * cell.y;
      const double world_y = sample.pose.v[1] + sin_yaw * cell.x + cos_yaw * cell.y;

      const double delta_x = world_x - map_origin_x_;
      const double delta_y = world_y - map_origin_y_;
      const double local_x = map_origin_cos_ * delta_x + map_origin_sin_ * delta_y;
      const double local_y = -map_origin_sin_ * delta_x + map_origin_cos_ * delta_y;
      const int column = static_cast<int>(std::floor(local_x / map_->scale));
      const int row = static_cast<int>(std::floor(local_y / map_->scale));

      // An off-map positive observation receives maximum disagreement.
      if (!MAP_VALID(map_, column, row)) {
        false_positive_sum += local_positive;
        continue;
      }
      const double static_occupancy =
        map_->cells[MAP_INDEX(map_, column, row)].occ_state > 0 ? 1.0 : 0.0;
      false_positive_sum += local_positive * (1.0 - static_occupancy);
    }

    const double normalized_sad = false_positive_sum / result.positive_mass;
    const auto output_index = static_cast<std::size_t>(sample_index);
    result.normalized_sad[output_index] = normalized_sad;
  }
  return result;
}

}  // namespace nav2_amcl
