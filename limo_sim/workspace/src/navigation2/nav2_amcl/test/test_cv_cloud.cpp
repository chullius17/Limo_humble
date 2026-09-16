// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "nav2_amcl/sensors/cv/cv_point_cloud.hpp"

namespace
{
using nav2_amcl::CvLikelihoodModel;
using nav2_amcl::CvTemplateCell2D;

sensor_msgs::msg::PointCloud2 cloudOf(const std::vector<std::array<float, 4>> & points)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 1;
  cloud.width = points.size();
  cloud.point_step = 16;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step);
  const char * names[] = {"x", "y", "z", "class_id"};
  for (uint32_t i = 0; i < 4; ++i) {
    sensor_msgs::msg::PointField field;
    field.name = names[i];
    field.count = 1;
    field.offset = i * 4;
    field.datatype = i == 3 ? field.UINT8 : field.FLOAT32;
    cloud.fields.push_back(field);
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    std::memcpy(cloud.data.data() + i * 16, points[i].data(), 12);
    cloud.data[i * 16 + 12] = static_cast<uint8_t>(points[i][3]);
  }
  return cloud;
}

TEST(CvCloud, MergesObstacleClassesButIgnoresBothBlueClassesAndInvalidPoints)
{
  auto cloud = cloudOf(
    {
      {0.01f, 0.01f, 0, 2}, {0.03f, 0.03f, 0, 3}, {0.05f, 0.05f, 0, 4},
      {1, 1, 0, 1}, {2, 2, 0, 5}, {3, 3, 0, 0},
      {std::numeric_limits<float>::quiet_NaN(), 0, 0, 2},
      {0, 0, std::numeric_limits<float>::infinity(), 3}});
  std::vector<CvTemplateCell2D> cells;
  std::string error;
  ASSERT_TRUE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.075, cells, error));
  ASSERT_EQ(cells.size(), 1u);
  EXPECT_NEAR(cells[0].x, 0.03, 1e-7);
  EXPECT_NEAR(cells[0].y, 0.03, 1e-7);
  EXPECT_DOUBLE_EQ(cells[0].occupancy, 1.0);
}

TEST(CvCloud, AppliesFullTransformAndHandlesNegativeVoxels)
{
  auto cloud = cloudOf({{-0.01f, 0.01f, 0, 3}, {0.01f, 0.01f, 0, 4}});
  std::vector<CvTemplateCell2D> cells;
  std::string error;
  ASSERT_TRUE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.1, cells, error));
  ASSERT_EQ(cells.size(), 2u);
  tf2::Quaternion rotation;
  rotation.setRPY(0, 0, M_PI / 2);
  tf2::Transform transform(rotation, tf2::Vector3(1, 2, 0));
  ASSERT_TRUE(nav2_amcl::voxelizeCvCloud(cloud, transform, 0.1, cells, error));
  ASSERT_EQ(cells.size(), 2u);
  for (const auto & cell : cells) {
    EXPECT_NEAR(cell.x, 0.99, 1e-6);
    EXPECT_NEAR(std::abs(cell.y - 2), 0.01, 1e-6);
  }
}

TEST(CvCloud, ReadsOrganizedBigEndianCloudWithRowPadding)
{
  auto cloud = cloudOf({{1, 2, 0, 2}, {3, 4, 0, 3}});
  cloud.height = 2;
  cloud.width = 1;
  cloud.row_step = 20;
  cloud.data.resize(40);
  std::memmove(cloud.data.data() + 20, cloud.data.data() + 16, 16);
  cloud.is_bigendian = true;
  for (std::size_t start : {0u, 20u}) {
    for (int offset : {0, 4, 8}) {
      std::reverse(cloud.data.begin() + start + offset, cloud.data.begin() + start + offset + 4);
    }
  }
  std::vector<CvTemplateCell2D> cells;
  std::string error;
  ASSERT_TRUE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.1, cells, error));
  ASSERT_EQ(cells.size(), 2u);
  EXPECT_DOUBLE_EQ(cells[0].x + cells[1].x, 4);
  EXPECT_DOUBLE_EQ(cells[0].y + cells[1].y, 6);
  cloud.data.resize(25);
  EXPECT_FALSE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.1, cells, error));
  EXPECT_TRUE(cells.empty());
}

TEST(CvCloud, RejectsMissingOrWrongFieldsAndInvalidVoxelSize)
{
  auto cloud = cloudOf({{1, 2, 0, 2}});
  std::vector<CvTemplateCell2D> cells;
  std::string error;
  EXPECT_FALSE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.0, cells, error));
  cloud.fields.back().datatype = sensor_msgs::msg::PointField::FLOAT32;
  EXPECT_FALSE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.1, cells, error));
  cloud.fields.pop_back();
  EXPECT_FALSE(
    nav2_amcl::voxelizeCvCloud(
      cloud, tf2::Transform::getIdentity(), 0.1, cells, error));
}

TEST(CvCloud, ScoresAllParticlesAndFusesWithLaserBeforeResampling)
{
  nav_msgs::msg::OccupancyGrid map;
  map.info.width = 4;
  map.info.height = 1;
  map.info.resolution = 1;
  map.info.origin.orientation.w = 1;
  map.data = {100, 0, -1, 100};
  CvLikelihoodModel model(CvLikelihoodModel::Parameters{});
  ASSERT_TRUE(model.setMap(map));
  pf_sample_t samples[4]{};
  pf_sample_set_t set{};
  set.samples = samples;
  set.sample_count = 4;
  for (int i = 0; i < 4; ++i) {
    samples[i].pose.v[0] = i;
    samples[i].weight = 0.25;
  }
  const auto score = model.scoreSad(&set, {{0.2, 0.2, 1.0}});
  EXPECT_EQ(score.normalized_sad, (std::vector<double>{0, 1, 1, 0}));
  EXPECT_DOUBLE_EQ(score.positive_mass, 1.0);
  ASSERT_TRUE(CvLikelihoodModel::fuseWeights(&set, score, 2, 1, 20));
  EXPECT_NEAR(samples[0].weight, 0.5, 1e-8);
  EXPECT_NEAR(samples[3].weight, 0.5, 1e-8);
  EXPECT_LT(samples[1].weight, 1e-8);
  EXPECT_LT(samples[2].weight, 1e-8);
  double total = 0;
  for (const auto & sample : samples) {
    total += sample.weight;
  }
  EXPECT_NEAR(total, 1, 1e-12);
  EXPECT_TRUE(CvLikelihoodModel::fuseWeights(&set, score, 2, 1, 10000));
  EXPECT_TRUE(std::isfinite(samples[0].weight));
}

TEST(CvCloud, RotatedMapAndParticlePoseAreAppliedIndependently)
{
  nav_msgs::msg::OccupancyGrid map;
  map.info.width = 2;
  map.info.height = 1;
  map.info.resolution = 1;
  map.info.origin.position.x = 10;
  map.info.origin.position.y = 20;
  map.info.origin.orientation.z = std::sin(M_PI / 4);
  map.info.origin.orientation.w = std::cos(M_PI / 4);
  map.data = {100, 0};
  CvLikelihoodModel model(CvLikelihoodModel::Parameters{});
  ASSERT_TRUE(model.setMap(map));
  pf_sample_t samples[2]{};
  pf_sample_set_t set{};
  set.samples = samples;
  set.sample_count = 2;
  samples[0].pose.v[0] = 10;
  samples[0].pose.v[1] = 20;
  samples[0].pose.v[2] = M_PI / 2;
  samples[1].pose.v[0] = 100;  // Off-map observation must disagree.
  const auto score = model.scoreSad(&set, {{0.5, 0.5, 1}});
  EXPECT_EQ(score.normalized_sad, (std::vector<double>{0, 1}));
}

TEST(CvCloud, FusionUsesHumbleExponentsAndDoesNotModifyWeightsOnInvalidInput)
{
  pf_sample_t samples[2]{};
  samples[0].weight = 0.8;
  samples[1].weight = 0.2;
  pf_sample_set_t set{};
  set.samples = samples;
  set.sample_count = 2;
  CvLikelihoodModel::SadScoreResult score;
  score.normalized_sad = {0, 0};
  ASSERT_TRUE(CvLikelihoodModel::fuseWeights(&set, score, 2, 1, 20));
  EXPECT_NEAR(samples[0].weight / samples[1].weight, 16, 1e-12);
  const double old_weight = samples[0].weight;
  score.normalized_sad[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(CvLikelihoodModel::fuseWeights(&set, score, 2, 1, 20));
  EXPECT_EQ(samples[0].weight, old_weight);
}
}  // namespace
