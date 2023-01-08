#pragma once

#include "warpsense/math/math.h"
#include "warpsense/cuda/device_map.h"
#include "map/tsdf.h"

namespace cuda
{

struct DeviceMapMemWrapper
{
  explicit DeviceMapMemWrapper(size_t n_voxels);
  explicit DeviceMapMemWrapper(const DeviceMap& existing_map);

  DeviceMapMemWrapper(const DeviceMapMemWrapper &other) = delete;
  DeviceMapMemWrapper operator=(const DeviceMapMemWrapper &other) = delete;
  DeviceMapMemWrapper(DeviceMapMemWrapper&& other) = delete;
  DeviceMapMemWrapper operator=(DeviceMapMemWrapper&& other) = delete;

  ~DeviceMapMemWrapper();

  void to_device(const DeviceMap &existing_map);
  void update_params(const DeviceMap &existing_map);

  static void to_host(cuda::DeviceMap &output, cuda::DeviceMap &inner, const DeviceMap *dev, size_t n_voxels);

  static cuda::DeviceMap* to_device(const cuda::DeviceMap &input, cuda::DeviceMap &inner, size_t n_voxels);

  void to_host(const DeviceMap &existing_map);

  DeviceMap *dev() const;

  DeviceMap inner_;
  DeviceMap *dev_;
  size_t n_voxels_;
};

} // end namespace cuda
