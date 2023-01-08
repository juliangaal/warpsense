#pragma once

#include "warpsense/math/math.h"
#include "warpsense/cuda/device_map.h"
#include "warpsense/cuda/device_map_wrapper.h"

namespace cuda
{

class RegistrationCuda
{
public:
  explicit RegistrationCuda(const cuda::DeviceMap &map);
  ~RegistrationCuda();
  void prepare_registration(const std::vector<rmagine::Pointi> &points);
  void
  perform_registration(const cuda::DeviceMap *map_dev, const rmagine::Matrix4x4f *pretransform, rmagine::Matrix6x6l &h,
                       rmagine::Point6l &g, int &e, int &c, int map_resolution);
private:
  void reduce(rmagine::Matrix6x6l& h,
              rmagine::Point6l& g,
              int& e,
              int& c,
              rmagine::Matrix6x6l *h_res,
              rmagine::Point6l* g_res,
              int* e_res,
              int* c_res,
              int const size);
  rmagine::Vector3i* points_dev;
  rmagine::Matrix4x4f* pretransform_dev;
  rmagine::Point6l* res_jacobis_dev;
  TSDFEntry::ValueType* res_values_dev;
  bool* res_mask_dev;
  rmagine::Matrix6x6l* res_h_dev;
  rmagine::Matrix6x6l* res_h;
  rmagine::Point6l* res_g_dev;
  rmagine::Point6l* res_g;
  int* res_e_dev;
  int* res_e;
  int* res_c_dev;
  int* res_c;
  size_t max_points;
  size_t curr_n_points;
  static constexpr size_t reduction_split = 32;
};

}