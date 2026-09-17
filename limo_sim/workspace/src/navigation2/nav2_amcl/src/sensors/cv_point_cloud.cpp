// Copyright (c) 2026 Giulio Cataldo
// Licensed under the Apache License, Version 2.0.
#include "nav2_amcl/sensors/cv/cv_point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace nav2_amcl
{
bool voxelizeCvCloud(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const tf2::Transform & cloud_to_base, double voxel_size,
  std::vector<CvTemplateCell2D> & cells, std::string & error)
{
  cells.clear();
  error.clear();
  if (!std::isfinite(voxel_size) || voxel_size <= 0.0) {
    error = "voxel size must be finite and positive";
    return false;
  }
  using Field = sensor_msgs::msg::PointField;
  const Field * fields[4] = {nullptr, nullptr, nullptr, nullptr};
  const char * names[4] = {"x", "y", "z", "class_id"};
  for (const auto & field : cloud.fields) {
    for (int i = 0; i < 4; ++i) {
      if (field.name == names[i]) {
        fields[i] = &field;
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    const auto * field = fields[i];
    const uint32_t size = i == 3 ? 1 : 4;
    if (!field || field->count != 1 ||
      field->datatype != (i == 3 ? Field::UINT8 : Field::FLOAT32) ||
      field->offset > cloud.point_step || size > cloud.point_step - field->offset)
    {
      error = "expected x/y/z FLOAT32 and class_id UINT8 within point_step";
      return false;
    }
  }
  if (static_cast<uint64_t>(cloud.width) * cloud.point_step > cloud.row_step ||
    static_cast<uint64_t>(cloud.row_step) * cloud.height > cloud.data.size())
  {
    error = "truncated cloud or invalid row_step";
    return false;
  }
  const uint16_t endian_probe = 1;
  const bool host_big_endian = *reinterpret_cast<const uint8_t *>(&endian_probe) == 0;
  const bool swap = host_big_endian != cloud.is_bigendian;
  const auto read_float = [swap](const uint8_t * source) {
      uint8_t bytes[4];
      std::memcpy(bytes, source, 4);
      if (swap) {
        std::reverse(bytes, bytes + 4);
      }
      float value;
      std::memcpy(&value, bytes, 4);
      return value;
    };
  struct Voxel
  {
    double x{0.0};
    double y{0.0};
    std::size_t count{0};
  };
  // Keep road and obstacle evidence separate even within the same XY voxel.
  std::unordered_map<uint64_t, std::array<Voxel, 2>> voxels;
  for (uint32_t row = 0; row < cloud.height; ++row) {
    for (uint32_t column = 0; column < cloud.width; ++column) {
      const auto * point = cloud.data.data() +
        static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      const uint8_t label = point[fields[3]->offset];
      const bool road = label == 1 || label == 5;
      if (!road && label != 2 && label != 4 && label != 6) {
        continue;
      }
      const double x = read_float(point + fields[0]->offset);
      const double y = read_float(point + fields[1]->offset);
      const double z = read_float(point + fields[2]->offset);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      const tf2::Vector3 transformed = cloud_to_base * tf2::Vector3(x, y, z);
      const double ix = std::floor(transformed.x() / voxel_size);
      const double iy = std::floor(transformed.y() / voxel_size);
      if (!std::isfinite(ix) || !std::isfinite(iy) ||
        ix < std::numeric_limits<int32_t>::min() ||
        ix > std::numeric_limits<int32_t>::max() ||
        iy < std::numeric_limits<int32_t>::min() ||
        iy > std::numeric_limits<int32_t>::max())
      {
        continue;
      }
      const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ix))) << 32) |
        static_cast<uint32_t>(static_cast<int32_t>(iy));
      auto & voxel = voxels[key][road ? 0 : 1];
      voxel.x += transformed.x();
      voxel.y += transformed.y();
      ++voxel.count;
    }
  }
  cells.reserve(voxels.size() * 2);
  for (const auto & entry : voxels) {
    for (std::size_t polarity = 0; polarity < 2; ++polarity) {
      const auto & voxel = entry.second[polarity];
      if (voxel.count > 0) {
        cells.push_back(
          {voxel.x / voxel.count, voxel.y / voxel.count, static_cast<double>(polarity)});
      }
    }
  }
  return true;
}
}  // namespace nav2_amcl
