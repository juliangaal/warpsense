#include "warpsense/cuda/cleanup.h"
#include "warpsense/tsdf_registration.h"
#include "warpsense/registration/util.h"
#include "params/params.h"
#include "warpsense/visualization/map.h"

#include <pcl/filters/voxel_grid.h>

using namespace std::chrono_literals;

namespace cuda
{

TSDFRegistration::TSDFRegistration(ros::NodeHandle &nh, const Params &params,
                                   ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr &pose_buffer,
                                   HDF5LocalMap::Ptr &local_map)
    : TSDFMapping(nh, params, pose_buffer, local_map)
    , reg_{std::make_unique<RegistrationCuda>(cuda_map_)}
{
}

TSDFRegistration::TSDFRegistration(const Params &params, HDF5LocalMap::Ptr &local_map)
    : TSDFMapping(params, local_map)
    , reg_{std::make_unique<RegistrationCuda>(cuda_map_)}
{
}

Eigen::Matrix4f
TSDFRegistration::register_cloud(std::vector<rmagine::Pointi> &cloud, const Eigen::Matrix4f &pretransform)
{
  Eigen::Matrix4f total_transform = pretransform;
  const rmagine::Matrix4x4f *total_transform_cu = reinterpret_cast<rmagine::Matrix4x4f *>(&total_transform);
  Point center = total_transform.block<3, 1>(0, 3).cast<int>();
  float alpha = 0;
  float previous_errors[4] = {0, 0, 0, 0};
  bool finished = false;
  Vector6d xi;

  std::vector<rmagine::Point6l> gpu_jacobis;
  std::vector<TSDFEntry::ValueType> gpu_values;
  std::vector<bool> gpu_mask;
  rmagine::Matrix6x6l gpu_h;
  rmagine::Point6l gpu_g;
  int gpu_e;
  int gpu_c;

  const auto &max_iterations = params_.registration.max_iterations;
  const auto &map_resolution = params_.map.resolution;
  const auto &epsilon = params_.registration.epsilon;
  const auto &it_weight_gradient = params_.registration.it_weight_gradient;

  reg_->prepare_registration(cloud);

  std::shared_lock lock(mutex_);
  for (int i = 0; i < max_iterations && !finished; i++)
  {
    reg_->perform_registration(tsdf_->device_map(), total_transform_cu,
                               gpu_h,
                               gpu_g,
                               gpu_e,
                               gpu_c, map_resolution);

    Matrix6d hf = (*reinterpret_cast<Matrix6l *>(&gpu_h)).cast<double>();
    Vector6d gf = (*reinterpret_cast<Point6l *>(&gpu_g)).cast<double>();

    // W Matrix
    hf += alpha * gpu_c * Matrix6d::Identity();

    xi = -hf.inverse() * gf; //-h.completeOrthogonalDecomposition().pseudoInverse() * g;

    //convert xi into transform matrix T
    Eigen::Matrix4f transform = xi_to_transform(xi, center);

    alpha += it_weight_gradient;

    // update transform
    total_transform = transform * total_transform;
    total_transform_cu = reinterpret_cast<rmagine::Matrix4x4f *>(&total_transform);

    float err = (float) gpu_e / gpu_c;
    if (fabs(err - previous_errors[2]) < epsilon && fabs(err - previous_errors[0]) < epsilon)
    {
      //std::cout << "(GPU) Stopped after " << i << " / " << max_iterations << " Iterations" << std::endl;
      finished = true;
    }

    for (int e = 1; e < 4; e++)
    {
      previous_errors[e - 1] = previous_errors[e];
    }

    previous_errors[3] = err;
  }

  return total_transform;
}


TSDFRegistration::~TSDFRegistration()
{
  reg_.reset();
}

} // end namespace cuda
