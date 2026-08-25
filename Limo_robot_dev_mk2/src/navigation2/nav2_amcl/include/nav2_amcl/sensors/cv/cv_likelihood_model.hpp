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

/** @brief One sampled local OccupancyGrid cell expressed in the robot frame. */
struct CvTemplateCell2D
{
  double x;
  double y;
  double occupancy;
};

/**
 * @class CvLikelihoodModel
 * @brief Evaluate particle poses using semantic grid SAD matching.
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

  /** @brief Per-particle normalized positive-class mismatch. */
  struct SadScoreResult
  {
    std::vector<double> normalized_sad;
    /// Foreground mass after fractional grid downsampling.
    double positive_mass{0.0};
  };

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
   * @brief Compare positive local evidence with its matching static CV map.
   *
   * Only false-positive semantic mismatches vote: a local obstacle or street
   * cell is penalized when it lands outside the corresponding static class.
   * White local cells carry no evidence. Fractional occupancy produced by
   * downsampling acts as confidence and supplies the normalization mass.
   */
  SadScoreResult scoreSad(
    const pf_sample_set_t * set,
    const std::vector<CvTemplateCell2D> & cells) const;

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
