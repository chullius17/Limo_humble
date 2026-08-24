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
#include <map>
#include <utility>
#include <vector>

namespace nav2_amcl
{

namespace
{

// Running sums used to compute one centroid for every occupied XY voxel.
struct VoxelAccumulator
{
  double sum_x{0.0};
  double sum_y{0.0};
  std::size_t count{0};
};

}  // namespace

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
  // Reject malformed maps before allocating a replacement for the current
  // valid distance field.
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

  // Build a private AMCL map. Its occ_dist member is populated by the same
  // c-space implementation used by the laser likelihood-field model.
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
  map_update_cspace(new_map, parameters_.max_occ_dist);

  // Replace the previous field only after construction has succeeded.
  if (map_ != nullptr) {
    map_free(map_);
  }
  map_ = new_map;
  map_origin_x_ = map_msg.info.origin.position.x;
  map_origin_y_ = map_msg.info.origin.position.y;

  // OccupancyGrid origins may be rotated. Cache their yaw so score() can use
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

double CvLikelihoodModel::distanceAt(double world_x, double world_y) const
{
  // Apply the inverse OccupancyGrid origin transform before indexing the
  // distance field. A point outside the grid is treated as maximally distant,
  // which supplies neither a semantic match nor a contradiction.
  const double delta_x = world_x - map_origin_x_;
  const double delta_y = world_y - map_origin_y_;
  const double local_x = map_origin_cos_ * delta_x + map_origin_sin_ * delta_y;
  const double local_y = -map_origin_sin_ * delta_x + map_origin_cos_ * delta_y;
  const int column = static_cast<int>(std::floor(local_x / map_->scale));
  const int row = static_cast<int>(std::floor(local_y / map_->scale));
  if (!MAP_VALID(map_, column, row)) {
    return parameters_.max_occ_dist;
  }
  return map_->cells[MAP_INDEX(map_, column, row)].occ_dist;
}

double CvLikelihoodModel::distanceStrength(double distance) const
{
  // p=2 is the original Gaussian likelihood field. Values above two preserve
  // tolerance inside sigma_hit while rejecting particles outside that band
  // more sharply, producing a more discriminative semantic template score.
  const double normalized_distance = distance / parameters_.sigma_hit;
  return std::exp(
    -0.5 * std::pow(normalized_distance, parameters_.distance_exponent));
}

std::vector<CvPoint2D> CvLikelihoodModel::voxelize(
  const std::vector<CvPoint2D> & points) const
{
  // std::map keeps voxel ordering deterministic, which also makes the optional
  // max_points subsampling repeatable across runs.
  using VoxelIndex = std::pair<long long, long long>;
  std::map<VoxelIndex, VoxelAccumulator> voxels;
  for (const auto & point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      continue;
    }
    // floor() is important for negative base-frame coordinates: truncation
    // toward zero would merge points from opposite sides of a voxel boundary.
    const VoxelIndex index{
      static_cast<long long>(std::floor(point.x / parameters_.voxel_leaf_size)),
      static_cast<long long>(std::floor(point.y / parameters_.voxel_leaf_size))};
    auto & accumulator = voxels[index];
    accumulator.sum_x += point.x;
    accumulator.sum_y += point.y;
    ++accumulator.count;
  }

  std::vector<CvPoint2D> centroids;
  centroids.reserve(voxels.size());
  for (const auto & voxel : voxels) {
    const auto & accumulator = voxel.second;
    centroids.push_back(
      {
        accumulator.sum_x / static_cast<double>(accumulator.count),
        accumulator.sum_y / static_cast<double>(accumulator.count)});
  }

  // Preserve all centroids when possible. If the cloud is still too dense,
  // retain an evenly spaced deterministic subset of the ordered voxels.
  if (centroids.size() <= parameters_.max_points) {
    return centroids;
  }
  if (parameters_.max_points == 1) {
    return {centroids[centroids.size() / 2]};
  }

  std::vector<CvPoint2D> selected;
  selected.reserve(parameters_.max_points);
  const double stride = static_cast<double>(centroids.size() - 1) /
    static_cast<double>(parameters_.max_points - 1);
  for (std::size_t index = 0; index < parameters_.max_points; ++index) {
    const auto source_index = static_cast<std::size_t>(
      std::llround(static_cast<double>(index) * stride));
    selected.push_back(centroids[source_index]);
  }
  return selected;
}

CvLikelihoodModel::ScoreResult CvLikelihoodModel::score(
  const pf_sample_set_t * set,
  const std::vector<CvPoint2D> & points,
  const CvLikelihoodModel & opposite_class_model) const
{
  if (!ready() || !opposite_class_model.ready() || set == nullptr || points.empty()) {
    return {};
  }

  ScoreResult result;
  const auto sample_count = static_cast<std::size_t>(set->sample_count);
  result.likelihoods.resize(sample_count, 1.0);
  result.mean_match_strengths.resize(sample_count, 0.0);
  result.mean_mismatch_strengths.resize(sample_count, 0.0);

  // These terms are shared by every point and particle. The small positive
  // floor makes the geometric mean well-defined even when z_rand is zero.
  const double random_likelihood = parameters_.z_rand / parameters_.sensor_max_range;
  constexpr double likelihood_floor = 1.0e-12;

  for (int sample_index = 0; sample_index < set->sample_count; ++sample_index) {
    const auto & sample = set->samples[sample_index];
    const double cos_yaw = std::cos(sample.pose.v[2]);
    const double sin_yaw = std::sin(sample.pose.v[2]);

    double log_likelihood_sum = 0.0;
    double match_strength_sum = 0.0;
    double mismatch_strength_sum = 0.0;

    for (const auto & point : points) {
      // Project this robot-frame CV observation through the hypothetical
      // particle pose to obtain a point in the global CV map frame.
      const double world_x = sample.pose.v[0] + cos_yaw * point.x - sin_yaw * point.y;
      const double world_y = sample.pose.v[1] + sin_yaw * point.x + cos_yaw * point.y;

      // The expected map supplies a soft positive match. The opposite map is
      // sampled at the same world coordinate to expose semantic conflicts.
      const double match_distance = distanceAt(world_x, world_y);
      const double mismatch_distance = opposite_class_model.distanceAt(world_x, world_y);
      const double match_strength = distanceStrength(match_distance);
      const double mismatch_strength = opposite_class_model.distanceStrength(mismatch_distance);
      const double point_likelihood =
        parameters_.z_hit * match_strength +
        random_likelihood;

      // Accumulating in log space avoids underflow. A mismatch subtracts
      // evidence rather than merely failing to add a positive match.
      log_likelihood_sum += std::log(std::max(point_likelihood, likelihood_floor)) -
        parameters_.semantic_mismatch_penalty * mismatch_strength;
      match_strength_sum += match_strength;
      mismatch_strength_sum += mismatch_strength;
    }

    // The geometric mean prevents point count and voxel density from changing
    // the scale of this semantic sensor model.
    const double inverse_point_count = 1.0 / static_cast<double>(points.size());
    const auto output_index = static_cast<std::size_t>(sample_index);
    result.likelihoods[output_index] = std::exp(
      log_likelihood_sum * inverse_point_count);
    result.mean_match_strengths[output_index] =
      match_strength_sum * inverse_point_count;
    result.mean_mismatch_strengths[output_index] =
      mismatch_strength_sum * inverse_point_count;
  }
  return result;
}

CvLikelihoodModel::SadScoreResult CvLikelihoodModel::scoreSad(
  const pf_sample_set_t * set,
  const std::vector<CvTemplateCell2D> & cells,
  double gain) const
{
  if (!ready() || set == nullptr || cells.empty() || gain < 0.0) {
    return {};
  }

  SadScoreResult result;
  const auto sample_count = static_cast<std::size_t>(set->sample_count);
  result.likelihoods.resize(sample_count, 1.0);
  result.normalized_sad.resize(sample_count, 0.0);

  // Accumulate foreground confidence once. Background cells are deliberately
  // ignored because this semantic detector does not distinguish a confident
  // negative from a missed or unavailable classification.
  for (const auto & cell : cells) {
    if (std::isfinite(cell.occupancy)) {
      const double occupancy = std::clamp(cell.occupancy, 0.0, 1.0);
      result.positive_mass += occupancy;
      result.negative_mass += 1.0 - occupancy;
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
    result.likelihoods[output_index] = std::exp(-gain * normalized_sad);
  }
  return result;
}

}  // namespace nav2_amcl
