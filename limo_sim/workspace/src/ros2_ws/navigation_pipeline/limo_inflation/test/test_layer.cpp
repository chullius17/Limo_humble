#include <gtest/gtest.h>

#include <memory>

#include "nav2_costmap_2d/layer.hpp"
#include "pluginlib/class_loader.hpp"

TEST(BorderFollowLayerTest, IsDiscoverableByPluginlib)
{
  pluginlib::ClassLoader<nav2_costmap_2d::Layer> loader(
    "nav2_costmap_2d",
    "nav2_costmap_2d::Layer");

  auto layer = loader.createSharedInstance(
    "limo_inflation::BorderFollowLayer");
  ASSERT_NE(layer, nullptr);
}
