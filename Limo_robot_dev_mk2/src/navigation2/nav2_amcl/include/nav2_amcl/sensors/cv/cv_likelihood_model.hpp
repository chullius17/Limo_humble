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

#ifndef NAV2_AMCL__SENSORS__CV__CV_LIKELIHOOD_MODEL_HPP_
#define NAV2_AMCL__SENSORS__CV__CV_LIKELIHOOD_MODEL_HPP_

#include <cstddef>
#include <vector>

#include "nav2_amcl/map/map.hpp"
#include "nav2_amcl/pf/pf.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace nav2_amcl
{

/**
 * @brief Planar obstacle observation expressed in the robot base frame.
 *
 * The CV localization model is two-dimensional, so the height and any other
 * PointCloud2 fields are deliberately discarded before scoring.
 */
struct CvPoint2D
{
  double x;
  double y;
};

/** @brief One sampled local OccupancyGrid cell expressed in the robot frame. */
struct CvTemplateCell2D
{
  double x;
  double y;
  double occupancy;
};

/**
 * @class CvLikelihoodModel
 * @brief Evaluate particle poses using soft semantic template matching.
 *
 * Each semantic OccupancyGrid is converted into a metric distance field. For
 * every particle, observations are rewarded when they approach their expected
 * semantic class and penalized when they approach the opposite class. The
 * geometric mean makes the result independent of the number of sampled points.
 */
class CvLikelihoodModel
{
public:
  /** @brief Configuration kept immutable for the lifetime of the model. */
  struct Parameters
  {
    /// Weight assigned to the Gaussian obstacle-hit component.
    double z_hit{0.5};
    /// Weight assigned to the uniform random-observation component.
    double z_rand{0.5};
    /// Standard deviation, in metres, of the obstacle-hit Gaussian.
    double sigma_hit{0.2};
    /// Exponent of the normalized distance; 2 reproduces a Gaussian field.
    double distance_exponent{3.0};
    /// Distance-field saturation value and penalty for off-map points.
    double max_occ_dist{2.0};
    /// CV sensing range used to normalize the random component.
    double sensor_max_range{10.0};
    /// XY side length, in metres, of each downsampling voxel.
    double voxel_leaf_size{0.1};
    /// Maximum number of CV observations evaluated for each particle.
    std::size_t max_points{300};
    /// OccupancyGrid value at which a cell is considered an obstacle.
    int occupied_threshold{50};
    /// Relative penalty applied to observations matching the opposite class.
    double semantic_mismatch_penalty{1.0};
  };

  /** @brief Per-particle output and diagnostics of semantic template matching. */
  struct ScoreResult
  {
    /// Positive, unnormalized likelihood consumed by AMCL log-space fusion.
    std::vector<double> likelihoods;
    /// Mean soft agreement with the expected semantic map, in [0, 1].
    std::vector<double> mean_match_strengths;
    /// Mean soft agreement with the opposite semantic map, in [0, 1].
    std::vector<double> mean_mismatch_strengths;
  };

  /** @brief Per-particle positive-class mismatch and its likelihood. */
  struct SadScoreResult
  {
    std::vector<double> likelihoods;
    std::vector<double> normalized_sad;
    /// Foreground and background masses after fractional grid downsampling.
    double positive_mass{0.0};
    double negative_mass{0.0};
  };

  explicit CvLikelihoodModel(const Parameters & parameters);
  ~CvLikelihoodModel();

  CvLikelihoodModel(const CvLikelihoodModel &) = delete;
  CvLikelihoodModel & operator=(const CvLikelihoodModel &) = delete;

  /**
   * @brief Replace the CV map and rebuild its capped metric distance field.
   * @return true when the OccupancyGrid dimensions and resolution are valid.
   */
  bool setMap(const nav_msgs::msg::OccupancyGrid & map_msg);

  /** @brief Return whether a valid CV distance field is available. */
  bool ready() const;

  /**
   * @brief Downsample points by replacing every occupied XY voxel with its centroid.
   *
   * This uses the centroid semantics described by the PCL VoxelGrid tutorial:
   * https://pointclouds.org/documentation/tutorials/voxel_grid.html
   */
  std::vector<CvPoint2D> voxelize(const std::vector<CvPoint2D> & points) const;

  /**
   * @brief Compute one raw semantic likelihood for every particle.
   *
   * Particle poses are expected in the CV map frame and points in the robot
   * base frame represented by those poses. The expected and opposite semantic
   * maps may have different origins or resolutions, but must share a frame.
   * Returned arrays preserve particle order for fusion before resampling.
   */
  ScoreResult score(
    const pf_sample_set_t * set,
    const std::vector<CvPoint2D> & points,
    const CvLikelihoodModel & opposite_class_model) const;

  /**
   * @brief Compare positive local evidence with its matching static CV map.
   *
   * Only false-positive semantic mismatches vote: a local obstacle or street
   * cell is penalized when it lands outside the corresponding static class.
   * White local cells carry no evidence. Fractional occupancy produced by
   * downsampling acts as confidence and supplies the normalization mass.
   */
  SadScoreResult scoreSad(
    const pf_sample_set_t * set,
    const std::vector<CvTemplateCell2D> & cells,
    double gain) const;

private:
  /** @brief Sample the capped distance field at one world-frame coordinate. */
  double distanceAt(double world_x, double world_y) const;

  /** @brief Convert metric distance into a nonlinear soft-match strength. */
  double distanceStrength(double distance) const;

  /// Validated model parameters supplied by AmclNode during configuration.
  Parameters parameters_;
  /// AMCL map structure whose cells store distance to the closest CV obstacle.
  map_t * map_{nullptr};
  /// Lower-left OccupancyGrid origin used for world-to-grid conversion.
  double map_origin_x_{0.0};
  double map_origin_y_{0.0};
  /// Cached yaw rotation of the OccupancyGrid origin.
  double map_origin_cos_{1.0};
  double map_origin_sin_{0.0};
};

}  // namespace nav2_amcl

#endif  // NAV2_AMCL__SENSORS__CV__CV_LIKELIHOOD_MODEL_HPP_
