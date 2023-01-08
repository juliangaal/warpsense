#pragma once

#include "map/tsdf.h"
#include "warpsense/math/math.h"
#include "warpsense/cuda/common.cuh"
#ifndef __NVCC__
#include "map/hdf5_local_map.h"
#endif


namespace cuda
{

template<typename T>
RMAGINE_INLINE_FUNCTION
T overflow(T val, T max)
{
  if (val >= 2 * max)
  {
    return val - 2 * max;
  }
  else if (val >= max)
  {
    return val - max;
  }
  else
  {
    return val;
  }
}

struct DeviceMap
{
  DeviceMap(int* size, int* offset, TSDFEntry* data, int* pos)
  : size_(reinterpret_cast<rmagine::Pointi*>(size))
  , offset_(reinterpret_cast<rmagine::Pointi*>(offset))
  , data_(data)
  , pos_(reinterpret_cast<rmagine::Pointi*>(pos))
{}

#ifndef __NVCC__
  DeviceMap(const std::shared_ptr<HDF5LocalMap>& map)
      : size_(reinterpret_cast<rmagine::Pointi*>(map->get_size().data()))
      , offset_(reinterpret_cast<rmagine::Pointi*>(map->get_offset().data()))
      , data_(map->get_data())
      , pos_(reinterpret_cast<rmagine::Pointi*>(map->get_pos().data()))
  {
  }
#endif

  DeviceMap(rmagine::Pointi* size, rmagine::Pointi* offset, TSDFEntry* data, rmagine::Pointi* pos)
      : size_(size)
      , offset_(offset)
      , data_(data)
      , pos_(pos)
  {}

  DeviceMap() = default;

  DeviceMap(const DeviceMap& other) = delete;
  DeviceMap(DeviceMap&& other) = delete;

  DeviceMap operator=(const DeviceMap& other) = delete;
  DeviceMap operator=(DeviceMap&& other) = delete;

  ~DeviceMap() = default; // ALL CLEANUP HAPPENS IN HDF5 LOCAL MAP!

  RMAGINE_INLINE_FUNCTION
  const rmagine::Pointi* get_size() const
  {
    return size_;
  }

  RMAGINE_INLINE_FUNCTION
  const rmagine::Pointi* get_offset() const
  {
    return offset_;
  }


  RMAGINE_INLINE_FUNCTION
  const rmagine::Pointi* get_pos() const
  {
    return pos_;
  }

  RMAGINE_INLINE_FUNCTION
  bool in_bounds(int x, int y, int z) const
  {
    return in_bounds(rmagine::Vector3i(x, y, z));
  }

  RMAGINE_INLINE_FUNCTION
  int get_index(const rmagine::Vector3i& point) const
  {
    int x_offset = overflow(point.x - pos_->x + offset_->x + size_->x, size_->x) * size_->y * size_->z;
    int y_offset = overflow(point.y - pos_->y + offset_->y + size_->y, size_->y) * size_->z;
    int z_offset = overflow(point.z - pos_->z + offset_->z + size_->z, size_->z);
    int index = x_offset  + y_offset + z_offset;
    return index;
  }

  /**
   * Checks if x, y and z are within the current range
   *
   * @param p position of the index in global coordinates
   * @return true if (x, y, z) is within the area of the buffer
   */
  RMAGINE_INLINE_FUNCTION
  bool in_bounds(rmagine::Vector3i p) const
  {
    p = (p - *pos_).cwiseAbs();
    return p.x <= size_->x / 2 && p.y <= size_->y / 2 && p.z <= size_->z / 2;
  }

  RMAGINE_INLINE_FUNCTION
  bool in_bounds_with_buffer_neg(rmagine::Vector3i p, size_t buffer) const
  {
    p = (p - *pos_).cwiseAbs();
    return p.x <= ((size_->x / 2) - buffer) && p.y <= ((size_->y / 2) - buffer) && p.z <= ((size_->z / 2) - buffer);
  }

  RMAGINE_INLINE_FUNCTION
  bool in_bounds_with_buffer_pos(rmagine::Vector3i p, size_t buffer) const
  {
    p = (p - *pos_).cwiseAbs();
    return p.x <= ((size_->x / 2) + buffer) && p.y <= ((size_->y / 2) + buffer) && p.z <= ((size_->z / 2) + buffer);
  }

  RMAGINE_INLINE_FUNCTION
  TSDFEntry& value_unchecked(int x, int y, int z)
  {
    return data_[get_index(rmagine::Vector3i(x, y, z))];
  }

  RMAGINE_INLINE_FUNCTION
  TSDFEntry& value_unchecked(const rmagine::Vector3i& p)
  {
    return data_[get_index(p)];
  }

  RMAGINE_INLINE_FUNCTION
  const TSDFEntry& value_unchecked(int x, int y, int z) const
  {
    return data_[get_index(rmagine::Vector3i(x, y, z))];
  }

  /**
   * @brief see #value but without bounds checking
   *
   * @param p position of the index in global coordinates
   * @return value of the local map
   */
  RMAGINE_INLINE_FUNCTION
  const TSDFEntry& value_unchecked(const rmagine::Vector3i& p) const
  {
    return data_[get_index(p)];
  }

  rmagine::Pointi* size_;
  rmagine::Pointi* offset_;
  TSDFEntry* data_;
  rmagine::Pointi* pos_;
};

} // end namespace cuda