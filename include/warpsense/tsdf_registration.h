#pragma once
#include <thread>
#include <ros/node_handle.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <shared_mutex>

#include "cuda/device_map.h"
#include "cuda/update_tsdf.h"
#include "cuda/registration.h"
#include "util/concurrent_ring_buffer.h"
#include "params/params.h"
#include "warpsense/tsdf_mapping.h"

namespace cuda
{

struct TSDFRegistration : public TSDFMapping
{
  explicit TSDFRegistration(ros::NodeHandle &nh, const Params &params,
                       ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr &pose_buffer, HDF5LocalMap::Ptr &local_map);
  explicit TSDFRegistration(const Params &params, HDF5LocalMap::Ptr &local_map);
  ~TSDFRegistration() final;
  Eigen::Matrix4f register_cloud(std::vector<rmagine::Pointi> &cloud, const Eigen::Matrix4f &pretransform);
  std::unique_ptr<cuda::RegistrationCuda> reg_;
};

}
