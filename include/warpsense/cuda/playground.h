#pragma once

#include <vector>
#include <warpsense/math/math.h>
#include <warpsense/cuda/device_map.h>

namespace cuda
{

rmagine::Pointi transform_point(const rmagine::Pointi &point, const rmagine::Matrix4x4i &mat);

void jacobi_2_h(const rmagine::Point6l &jacobi, rmagine::Matrix6x6l &h);

void cross(const rmagine::Vector3i &a, const rmagine::Vector3i &b, rmagine::Vector3i &res);

float sum(std::vector<float> &data);

float sum(std::vector<float> &data, size_t split);

rmagine::Matrix3x3f cov(const std::vector<rmagine::Vector3f>& v1, const std::vector<rmagine::Vector3f>& v2);

rmagine::Matrix6x6f cov6d(const std::vector<rmagine::Vector6f>& v1, const std::vector<rmagine::Vector6f>& v2);

void h_g_e_reduction(const std::vector<rmagine::Vector6f> &v1, rmagine::Matrix6x6f &h, rmagine::Vector6f &g, float &e,
                     float &c);

void
calc_jacobis(const cuda::DeviceMap &map, const std::vector<rmagine::Pointi> &points, std::vector<rmagine::Point6l> &jacobis,
             std::vector<TSDFEntry::ValueType> &values, std::vector<bool> &mask);

void registration(const cuda::DeviceMap &map,
                  const rmagine::Matrix4x4f* pretransform,
                  const std::vector<rmagine::Pointi> &points,
                  std::vector<rmagine::Point6l> &jacobis,
                  std::vector<TSDFEntry::ValueType> &values,
                  std::vector<bool> &mask,
                  rmagine::Matrix6x6l& h,
                  rmagine::Point6l& g,
                  int& e, int& c);

int atomic_min(std::vector<int>& buffer, const std::vector<int>& values);

TSDFEntry atomic_tsdf_min(std::vector<TSDFEntry>& buffer, const std::vector<TSDFEntry>& values);
}