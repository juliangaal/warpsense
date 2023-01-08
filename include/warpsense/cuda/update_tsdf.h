#pragma once

#include <vector>
#include "warpsense/cuda/device_map_wrapper.h"

namespace cuda
{

class TSDFCuda
{
public:
  explicit TSDFCuda(const cuda::DeviceMap& existing_map, int tau, int max_weight, int map_resolution);
  ~TSDFCuda();
  void update_tsdf(const std::vector<rmagine::Pointi>& scan_points, const rmagine::Pointi& scanner_pos, const rmagine::Pointi& up);
  void update_tsdf(cuda::DeviceMap& result, const std::vector<rmagine::Pointi>& scan_points, const rmagine::Pointi& scanner_pos, const rmagine::Pointi& up);
  void update_tsdf(cuda::DeviceMap& result, cuda::DeviceMap& latest_map, const std::vector<rmagine::Pointi>& scan_points, const rmagine::Pointi& scanner_pos, const rmagine::Pointi& up);
  DeviceMap *device_map();
  const DeviceMap *device_map() const;
  const cuda::DeviceMapMemWrapper& avg_map() const;
  cuda::DeviceMapMemWrapper& avg_map();
  const cuda::DeviceMapMemWrapper& new_map() const;
  cuda::DeviceMapMemWrapper& new_map();
private:
  rmagine::Pointi* scan_points_dev_;
  rmagine::Pointi* scanner_pos_dev_;
  rmagine::Pointi* up_dev_;
  int tau_;
  int max_weight_;
  int map_resolution_;
  size_t n_voxels_;
  cuda::DeviceMapMemWrapper avg_map_;
  cuda::DeviceMapMemWrapper new_map_;
  static constexpr int n_max_points_ = 1'000'000;
};

}