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
#include <limits>
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
    !std::isfinite(map_msg.info.resolution) ||
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

  // Distinguish exact free space from unknown/intermediate map values.
  for (std::size_t index = 0; index < expected_size; ++index) {
    new_map->cells[index].occ_state =
      map_msg.data[index] == 0 ? -1 :
      (map_msg.data[index] >= parameters_.occupied_threshold ? 1 : 0);
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

  // Only explicit observations enter this template: occupancy 1 is an obstacle,
  // occupancy 0 is a classified road, never an absent/unclassified point.
  for (const auto & cell : cells) {
    if (std::isfinite(cell.occupancy) && std::isfinite(cell.x) && std::isfinite(cell.y)) {
      const double occupancy = std::max(0.0, std::min(cell.occupancy, 1.0));
      result.positive_mass += occupancy;
      result.negative_mass += 1.0 - occupancy;
    }
  }
  const double observation_mass = result.positive_mass + result.negative_mass;
  if (observation_mass <= 0.0) {
    return result;
  }

  for (int sample_index = 0; sample_index < set->sample_count; ++sample_index) {
    const auto & sample = set->samples[sample_index];
    const double cos_yaw = std::cos(sample.pose.v[2]);
    const double sin_yaw = std::sin(sample.pose.v[2]);
    double mismatch_sum = 0.0;

    for (const auto & cell : cells) {
      if (!std::isfinite(cell.x) || !std::isfinite(cell.y) ||
        !std::isfinite(cell.occupancy))
      {
        continue;
      }
      const double local_positive = std::max(0.0, std::min(cell.occupancy, 1.0));
      const double local_negative = 1.0 - local_positive;
      const double world_x = sample.pose.v[0] + cos_yaw * cell.x - sin_yaw * cell.y;
      const double world_y = sample.pose.v[1] + sin_yaw * cell.x + cos_yaw * cell.y;

      const double delta_x = world_x - map_origin_x_;
      const double delta_y = world_y - map_origin_y_;
      const double local_x = map_origin_cos_ * delta_x + map_origin_sin_ * delta_y;
      const double local_y = -map_origin_sin_ * delta_x + map_origin_cos_ * delta_y;
      const double column_d = std::floor(local_x / map_->scale);
      const double row_d = std::floor(local_y / map_->scale);
      if (!std::isfinite(column_d) || !std::isfinite(row_d) ||
        column_d < 0 || column_d >= map_->size_x || row_d < 0 || row_d >= map_->size_y)
      {
        mismatch_sum += local_positive + local_negative;
        continue;
      }
      const int column = static_cast<int>(column_d);
      const int row = static_cast<int>(row_d);

      const int static_state = map_->cells[MAP_INDEX(map_, column, row)].occ_state;
      mismatch_sum += local_positive * (static_state == 1 ? 0.0 : 1.0) +
        local_negative * (static_state == -1 ? 0.0 : 1.0);
    }

    const double normalized_sad = mismatch_sum / observation_mass;
    const auto output_index = static_cast<std::size_t>(sample_index);
    result.normalized_sad[output_index] = normalized_sad;
  }
  return result;
}

bool CvLikelihoodModel::assessQuality(
  const pf_sample_set_t * set, const SadScoreResult & score, double effective_gain,
  const QualityLimits & limits, QualityReport & report)
{
  report = QualityReport{};
  if (!set || !set->samples || set->sample_count <= 0 ||
    score.normalized_sad.size() != static_cast<std::size_t>(set->sample_count) ||
    !std::isfinite(effective_gain) || effective_gain <= 0.0 ||
    !std::isfinite(limits.min_information) || limits.min_information < 0.0 ||
    !std::isfinite(limits.max_position_stddev) || limits.max_position_stddev <= 0.0 ||
    !std::isfinite(limits.max_yaw_stddev) || limits.max_yaw_stddev <= 0.0)
  {
    return false;
  }
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < set->sample_count; ++i) {
    const auto & pose = set->samples[i].pose;
    if (!std::isfinite(score.normalized_sad[i]) || score.normalized_sad[i] < 0.0 ||
      !std::isfinite(pose.v[0]) || !std::isfinite(pose.v[1]) || !std::isfinite(pose.v[2]))
    {
      return false;
    }
    best = std::min(best, score.normalized_sad[i]);
  }
  std::vector<double> support(set->sample_count);
  double total = 0.0;
  for (int i = 0; i < set->sample_count; ++i) {
    support[i] = std::exp(-effective_gain * (score.normalized_sad[i] - best));
    total += support[i];
  }
  // Relative coordinates avoid subtracting large squared world coordinates.
  const double origin_x = set->samples[0].pose.v[0];
  const double origin_y = set->samples[0].pose.v[1];
  double mean_x = 0.0, mean_y = 0.0, mean_cos = 0.0, mean_sin = 0.0;
  for (int i = 0; i < set->sample_count; ++i) {
    support[i] /= total;
    const double p = support[i];
    const auto & pose = set->samples[i].pose;
    mean_x += p * (pose.v[0] - origin_x);
    mean_y += p * (pose.v[1] - origin_y);
    mean_cos += p * std::cos(pose.v[2]);
    mean_sin += p * std::sin(pose.v[2]);
    if (p > 0.0) {
      report.information += p * std::log(p * set->sample_count);
    }
  }
  double xx = 0.0, xy = 0.0, yy = 0.0;
  for (int i = 0; i < set->sample_count; ++i) {
    const double dx = (set->samples[i].pose.v[0] - origin_x) - mean_x;
    const double dy = (set->samples[i].pose.v[1] - origin_y) - mean_y;
    xx += support[i] * dx * dx;
    xy += support[i] * dx * dy;
    yy += support[i] * dy * dy;
  }
  report.information = std::max(0.0, report.information);
  report.position_stddev = std::sqrt(0.5 * (xx + yy + std::hypot(xx - yy, 2.0 * xy)));
  const double resultant = std::min(1.0, std::hypot(mean_cos, mean_sin));
  report.yaw_stddev = resultant > 0.0 ? std::sqrt(-2.0 * std::log(resultant)) :
    std::numeric_limits<double>::infinity();
  if (!std::isfinite(report.position_stddev) || !std::isfinite(report.information)) {
    return false;
  }
  if (report.information < limits.min_information || report.information <= 1e-12) {
    report.reason = "uninformative";
    return false;
  }
  if (report.position_stddev > limits.max_position_stddev) {
    report.reason = "position_spread";
    return false;
  }
  if (report.yaw_stddev > limits.max_yaw_stddev) {
    report.reason = "yaw_spread";
    return false;
  }
  report.reason = "accepted";
  return true;
}

bool CvLikelihoodModel::fuseWeights(
  pf_sample_set_t * set, const SadScoreResult & score,
  double laser_factor, double cv_factor, double gain)
{
  if (!set || set->sample_count <= 0 ||
    score.normalized_sad.size() != static_cast<std::size_t>(set->sample_count) ||
    !std::isfinite(laser_factor) || laser_factor < 0.0 ||
    !std::isfinite(cv_factor) || cv_factor < 0.0 || !std::isfinite(gain) || gain < 0.0)
  {
    return false;
  }
  std::vector<double> weights(set->sample_count);
  double maximum = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < set->sample_count; ++i) {
    if (!std::isfinite(set->samples[i].weight) || set->samples[i].weight < 0.0 ||
      !std::isfinite(score.normalized_sad[i]))
    {
      return false;
    }
    weights[i] = laser_factor * std::log(std::max(set->samples[i].weight, 1e-300)) -
      cv_factor * gain * score.normalized_sad[i];
    if (!std::isfinite(weights[i])) {
      return false;
    }
    maximum = std::max(maximum, weights[i]);
  }
  double total = 0.0;
  for (auto & weight : weights) {
    weight = std::exp(weight - maximum);
    total += weight;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return false;
  }
  for (int i = 0; i < set->sample_count; ++i) {
    set->samples[i].weight = weights[i] / total;
  }
  return true;
}

}  // namespace nav2_amcl
