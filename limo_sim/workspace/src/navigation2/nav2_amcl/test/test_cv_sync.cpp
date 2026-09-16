// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <cstring>
#include <memory>

#include "nav2_amcl/amcl_node.hpp"
#include "tf2_ros/buffer.h"

namespace
{
class AmclCvHarness : public nav2_amcl::AmclNode
{
public:
  void initialize()
  {
    set_parameter(rclcpp::Parameter("cv_enabled", true));
    set_parameter(rclcpp::Parameter("cv_min_points", 1.0));
    set_parameter(rclcpp::Parameter("base_frame_id", "base_link"));
    initParameters();
    initTransforms();
    initPubSub();
    active_ = true;
  }

  using AmclNode::applyCvFusion;
  using AmclNode::cvCloudReceived;
  using AmclNode::cvMapReceived;

  void addOdom(int seconds, double x)
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "odom";
    transform.child_frame_id = "base_link";
    transform.header.stamp.sec = seconds;
    transform.transform.translation.x = x;
    transform.transform.rotation.w = 1;
    ASSERT_TRUE(tf_buffer_->setTransform(transform, "test"));
  }
};

class CvSync : public ::testing::Test
{
protected:
  static void SetUpTestCase()
  {
    rclcpp::init(0, nullptr);
  }
  static void TearDownTestCase()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    node = std::make_shared<AmclCvHarness>();
    node->initialize();
    // At laser time (10.1), base has advanced 1 metre since cloud time (10).
    node->addOdom(10, 0);
    node->addOdom(11, 10);
    auto map = std::make_shared<nav_msgs::msg::OccupancyGrid>();
    map->header.frame_id = "map";
    map->info.width = 4;
    map->info.height = 1;
    map->info.resolution = 1;
    map->info.origin.orientation.w = 1;
    map->data = {0, 100, 0, 0};
    node->cvMapReceived(map);

    cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id = "base_link";
    cloud->header.stamp.sec = 10;
    cloud->height = 1;
    cloud->width = 1;
    cloud->point_step = 13;
    cloud->row_step = 13;
    cloud->data.resize(13);
    const char * names[] = {"x", "y", "z", "class_id"};
    for (uint32_t i = 0; i < 4; ++i) {
      sensor_msgs::msg::PointField field;
      field.name = names[i];
      field.count = 1;
      field.offset = i * 4;
      field.datatype = i == 3 ? field.UINT8 : field.FLOAT32;
      cloud->fields.push_back(field);
    }
    float point[] = {2.5, 0.5, 0};
    std::memcpy(cloud->data.data(), point, 12);
    cloud->data[12] = 3;
    samples[0].weight = 0.5;
    samples[1].weight = 0.5;
    samples[1].pose.v[0] = 1;
    set.sample_count = 2;
    set.samples = samples;
    stamp.sec = 10;
    stamp.nanosec = 100000000;
  }

  std::shared_ptr<AmclCvHarness> node;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud;
  pf_sample_t samples[2]{};
  pf_sample_set_t set{};
  builtin_interfaces::msg::Time stamp;
};

TEST_F(CvSync, CompensatesMotionAndDoesNotReuseCloud)
{
  node->cvCloudReceived(cloud);
  ASSERT_TRUE(node->applyCvFusion(&set, stamp));
  // 2.5 in old base -> 1.5 in new base, which matches occupied map cell 1.
  EXPECT_GT(samples[0].weight, 0.99);
  const double weight = samples[0].weight;
  EXPECT_FALSE(node->applyCvFusion(&set, stamp));
  EXPECT_DOUBLE_EQ(samples[0].weight, weight);
}

TEST_F(CvSync, MissingStaleEmptyAndUnavailableTfLeaveLaserWeightsUnchanged)
{
  EXPECT_FALSE(node->applyCvFusion(&set, stamp));
  cloud->header.stamp.sec = 9;
  node->cvCloudReceived(cloud);
  EXPECT_FALSE(node->applyCvFusion(&set, stamp));
  auto unavailable = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
  unavailable->header.stamp.sec = 10;
  unavailable->header.frame_id = "missing_camera";
  node->cvCloudReceived(unavailable);
  EXPECT_FALSE(node->applyCvFusion(&set, stamp));
  auto empty = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
  empty->header.stamp = stamp;
  empty->width = 0;
  empty->row_step = 0;
  empty->data.clear();
  node->cvCloudReceived(empty);
  EXPECT_FALSE(node->applyCvFusion(&set, stamp));
  EXPECT_DOUBLE_EQ(samples[0].weight, 0.5);
  EXPECT_DOUBLE_EQ(samples[1].weight, 0.5);
}
}  // namespace
