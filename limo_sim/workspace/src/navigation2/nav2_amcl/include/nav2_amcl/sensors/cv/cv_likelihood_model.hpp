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

/** @brief One observed semantic voxel: occupancy 1 = obstacle, 0 = road. */
struct CvTemplateCell2D
{
  double x;
  double y;
  double occupancy;
};

/**
 * @class CvLikelihoodModel
 * @brief Evaluate obstacle and road observations against a static semantic map.
 */
class CvLikelihoodModel
{
public:
  /** @brief Configuration kept immutable for the lifetime of the model. */
  struct Parameters
  {
    /// OccupancyGrid value at which a cell is considered an obstacle.
    int occupied_threshold{50};
  };

  /** @brief Per-particle mismatch normalized by all observed voxel votes. */
  struct SadScoreResult
  {
    std::vector<double> normalized_sad;
    /// Sum of positive observation weights (one per occupied point-cloud voxel).
    double positive_mass{0.0};
    /// Sum of explicit road observation weights (not missing classifications).
    double negative_mass{0.0};
  };

  struct QualityLimits
  {
    double min_information{0.02};  // KL divergence from uniform, in nats.
    double max_position_stddev{0.5};  // Metres, along the widest XY axis.
    double max_yaw_stddev{0.5};  // Circular standard deviation, radians.
  };

  struct QualityReport
  {
    double information{0.0};
    double position_stddev{0.0};
    double yaw_stddev{0.0};
    const char * reason{"invalid"};
  };

  /// Assess CV-only support over current particle poses without changing weights.
  static bool assessQuality(
    const pf_sample_set_t * set, const SadScoreResult & score, double effective_gain,
    const QualityLimits & limits, QualityReport & report);

  explicit CvLikelihoodModel(const Parameters & parameters);
  ~CvLikelihoodModel();

  CvLikelihoodModel(const CvLikelihoodModel &) = delete;
  CvLikelihoodModel & operator=(const CvLikelihoodModel &) = delete;

  /**
   * @brief Replace the static semantic occupancy map.
   * @return true when the OccupancyGrid dimensions and resolution are valid.
   */
  bool setMap(const nav_msgs::msg::OccupancyGrid & map_msg);

  /** @brief Return whether a valid CV map is available. */
  bool ready() const;

  /**
   * @brief Compare obstacle and road evidence with the static CV map.
   *
   * Each observed obstacle is penalized when it lands outside static occupied
   * cells, including unknown and off-map locations (the Humble SAD rule).
   * Road observations are penalized unless the static cell is exactly zero.
   * Unknown and off-map cells mismatch both types of observation. Soft obstacles
   * and unclassified points are excluded before scoring. Each observed voxel
   * contributes one vote per polarity; unobserved space supplies no evidence.
   */
  SadScoreResult scoreSad(
    const pf_sample_set_t * set,
    const std::vector<CvTemplateCell2D> & cells) const;

  /// Humble fusion rule: normalized laser weight^a * exp(-b * gain * SAD).
  /// Returns false without modifying weights when inputs are invalid.
  static bool fuseWeights(
    pf_sample_set_t * set, const SadScoreResult & score,
    double laser_factor, double cv_factor, double gain);

private:
  /// Validated model parameters supplied by AmclNode during configuration.
  Parameters parameters_;
  /// AMCL map structure whose cells store static semantic occupancy.
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
