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

/**
 * @class CvLikelihoodModel
 * @brief Evaluate particle poses against a CV-derived obstacle likelihood field.
 *
 * The model mirrors AMCL's laser likelihood-field approach. A static CV
 * OccupancyGrid is converted into a metric distance-to-nearest-obstacle field.
 * For each particle, robot-relative CV points are projected into the map and
 * their individual likelihoods are accumulated using AMCL's p += pz^3 rule.
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
   * @brief Compute one raw, unnormalized CV likelihood for every particle.
   *
   * Particle poses are expected in the CV map frame and points in the robot
   * base frame represented by those poses. The returned vector preserves the
   * particle array order so it can be fused index by index before resampling.
   */
  std::vector<double> score(
    const pf_sample_set_t * set,
    const std::vector<CvPoint2D> & points) const;

private:
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
