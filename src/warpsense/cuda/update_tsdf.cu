#include "warpsense/cuda/update_tsdf.h"
#include "warpsense/cuda/util.h"
#include "warpsense/cuda/device_map_wrapper.h"
#include "warpsense/cuda/common.cuh"
#include <vector>

namespace rm = rmagine;
using ARITH_TYPE = long;

namespace cuda
{

__global__ void
cu_avg_tsdf_krnl(cuda::DeviceMap *new_map, cuda::DeviceMap *existing_map, int max_weight, int tau)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < existing_map->size_->prod())
  {
    auto& existing_entry = existing_map->data_[ix];
    auto& new_entry = new_map->data_[ix];

    // both entries exist -> average da ting
    if (new_entry.weight() > 0 && existing_entry.weight() > 0)
    {
      existing_entry.value((existing_entry.value() * existing_entry.weight() + new_entry.value() * new_entry.weight()) / (existing_entry.weight() + new_entry.weight()));
      existing_entry.weight(min(max_weight, existing_entry.weight() + new_entry.weight()));
    }

      // if this is the first time writing to cell, overwrite with newest values
    else if (new_entry.weight() != 0 && existing_entry.weight() <= 0)
    {
      existing_entry.value(new_entry.value());
      existing_entry.weight(new_entry.weight());
    }

    // if value is different from default, reset in new_map to default for next iteration
    //if (existing_entry.weight() != existing_map->initial_tsdf_value_.weight() &&
    //existing_entry.value() != existing_map->initial_tsdf_value_.value())
    //{
    new_entry.value(tau);
    new_entry.weight(0);
  }
}

__global__ void
cu_min_tsdf_krnl(const rm::Pointi *scan_points, size_t N, const rm::Pointi *scanner_pos, const rm::Pointi *up,
                 cuda::DeviceMap *new_map, int tau, int map_resolution)
{
  float angle = 45.f / 128.f; // TODO: Scanner FoV as Parameter
  int dz_per_distance = tan(angle / 180 * M_PI) / 2.0 * MATRIX_RESOLUTION;
  int weight_epsilon = tau / 10;
  auto pos = cu_to_mm(*scanner_pos, map_resolution);

  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N && new_map->in_bounds_with_buffer_pos(cu_to_map(scan_points[ix], map_resolution), tau / map_resolution / 2))
  {
    const auto& point = scan_points[ix];
    rm::Pointi direction_vector = point - pos;
    int distance = direction_vector.l2norm();
    rm::Pointl normed_direction_vector = (direction_vector.cast<ARITH_TYPE>() * MATRIX_RESOLUTION) / distance;
    rm::Pointl interpolation_vector = (normed_direction_vector.cross(normed_direction_vector.cross((*up).cast<ARITH_TYPE>()) / MATRIX_RESOLUTION));
    auto interpolation_norm = interpolation_vector.l2norm();
    interpolation_vector = (interpolation_vector * MATRIX_RESOLUTION) / interpolation_norm;

    rm::Pointi prev;

    for (int len = 1; len <= distance + tau; len += map_resolution / 2)
    {
      rm::Pointi proj = pos + direction_vector * len / distance;
      rm::Pointi index = proj / map_resolution;
      if (index.x == prev.x && index.y == prev.y)
      {
        continue;
      }
      prev = index;
      if (!new_map->in_bounds(index.x, index.y, index.z))
      {
        continue;
      }

      rm::Pointi target_center = index * map_resolution + rm::Pointi::Constant(map_resolution / 2);
      long initial_value = (point - target_center).l2norm();
      int value = min(initial_value, (long)tau); // Investigate max tau
      if (len > distance)
      {
        value = -value;
      }

      // Calculate the corresponding weight for every TSDF value
      int weight = WEIGHT_RESOLUTION;
      if (value < -weight_epsilon)
      {
        weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
      }
      if (weight == 0)
      {
        continue;
      }
      auto object = TSDFEntry((int16_t)value, (int16_t)weight);

      int delta_z = dz_per_distance * len / MATRIX_RESOLUTION;
      auto iter_steps = (delta_z * 2) / map_resolution + 1;
      auto mid = delta_z / map_resolution;
      auto lowest = (proj - ((delta_z * interpolation_vector) / MATRIX_RESOLUTION).cast<int>());
      auto mid_index = index;

      for (auto step = 0; step < iter_steps; ++step)
      {
        index = (lowest + ((step * map_resolution * interpolation_vector) / MATRIX_RESOLUTION).cast<int>()) / map_resolution;

        if (!new_map->in_bounds(index.x, index.y, index.z))
        {
          continue;
        }

        auto tmp = object;

        //if (mid_index != index)
        if (step != mid)
        {
          tmp.weight(tmp.weight() * -1);
        }

        write_tsdf_min(new_map->data_, new_map->get_index(index), tmp);
      }
    }
  }
}

TSDFCuda::TSDFCuda(const DeviceMap &existing_map, int tau, int max_weight, int map_resolution)
: tau_(tau)
, max_weight_(max_weight)
, map_resolution_(map_resolution)
, n_voxels_(existing_map.size_->prod())
, avg_map_(existing_map)
, new_map_(existing_map)
{
  CHECK(cudaMalloc((void **) &scanner_pos_dev_, sizeof(rm::Pointi)));
  CHECK(cudaMalloc((void **) &up_dev_, sizeof(rm::Pointi)));
  CHECK(cudaMalloc((void **) &scan_points_dev_, sizeof(rm::Pointi) * n_max_points_));
}

void TSDFCuda::update_tsdf(const std::vector<rm::Pointi> &scan_points, const rm::Pointi &scanner_pos, const rm::Pointi &up)
{
  size_t n_points = scan_points.size();
  if (n_points > n_max_points_)
  {
    fprintf(stderr, "CUDA Error: %s:%d - TSDF Update with %ld points. Larger than max allowed points %d", __FILE__, __LINE__, n_points, n_max_points_);
    return;
  }

  CHECK(cudaMemcpy(scan_points_dev_, scan_points.data(), sizeof(rm::Pointi) * n_points, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(scanner_pos_dev_, &scanner_pos, sizeof(rm::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(up_dev_, &up, sizeof(rm::Pointi), cudaMemcpyHostToDevice));

  size_t blocksize = 1024;
  dim3 block(blocksize);
  dim3 grid((n_points + block.x - 1) / block.x);
  cu_min_tsdf_krnl<<<grid, block>>>(scan_points_dev_, n_points, scanner_pos_dev_, up_dev_, new_map_.dev(), tau_,
                                    map_resolution_);

  grid.x = (n_voxels_ + block.x - 1) / block.x;
  cu_avg_tsdf_krnl<<<grid, block>>>(new_map_.dev(), avg_map_.dev(), max_weight_, tau_);
  CHECK(cudaGetLastError());
  //cudaDeviceSynchronize(); // TODO necessary?
}

void
TSDFCuda::update_tsdf(DeviceMap &result, const std::vector<rmagine::Pointi> &scan_points, const rmagine::Pointi &scanner_pos,
                      const rmagine::Pointi &up)
{
  update_tsdf(scan_points, scanner_pos, up);
  avg_map_.to_host(result);
  //cudaDeviceSynchronize(); // TODO necessary?
}

TSDFCuda::~TSDFCuda()
{
  CHECK(cudaFree(scan_points_dev_));
  CHECK(cudaFree(scanner_pos_dev_));
  CHECK(cudaFree(up_dev_));
  // CHECK(cudaDeviceReset()); TODO call only once across application
}

void TSDFCuda::update_tsdf(DeviceMap &result, DeviceMap &latest_map, const std::vector<rmagine::Pointi> &scan_points,
                           const rmagine::Pointi &scanner_pos, const rmagine::Pointi &up)
{
  update_tsdf(scan_points, scanner_pos, up);
  avg_map_.to_host(result);
  new_map_.to_host(latest_map);
}

DeviceMap *TSDFCuda::device_map()
{
  return avg_map_.dev();
}

const DeviceMap *TSDFCuda::device_map() const
{
  return avg_map_.dev();
}

const cuda::DeviceMapMemWrapper &TSDFCuda::avg_map() const
{
  return avg_map_;
}

cuda::DeviceMapMemWrapper &TSDFCuda::avg_map()
{
  return avg_map_;
}

const cuda::DeviceMapMemWrapper &TSDFCuda::new_map() const
{
  return new_map_;
}

cuda::DeviceMapMemWrapper &TSDFCuda::new_map()
{
  return new_map_;
}

} // end namespace cuda