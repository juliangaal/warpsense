#pragma once

#include "featsense/buffers.h"
#include "util/background_thread.h"
#include "params/params.h"
#include "warpsense/vgicp.h"
#include "warpsense/tsdf_mapping.h"
#include "map/hdf5_global_map.h"
#include "map/hdf5_local_map.h"

class Mapping : public BackgroundThread
{
public:
  Mapping(ros::NodeHandle& nh, const Params& params, Buffers& buffers);
  ~Mapping() override;
  void terminate(int signal = 0);
protected:
  void thread_run() final;
private:
  std::shared_ptr<HDF5GlobalMap> hdf5_global_map_;
  std::shared_ptr<HDF5LocalMap> hdf5_local_map_;
  ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr map_shift_pose_buffer_;
  cuda::TSDFMapping gpu_;
  ros::Publisher map_pub_;
  ros::Publisher path_pub_;
  ros::Publisher pre_path_pub_;
  const Params& params_;
  PCLBuffer::Ptr& cloud_buf_;
  OdomBuffer::Ptr& odom_buf_;
  HeaderBuffer::Ptr& header_buf_;
  bool initialized_;
  bool terminated_;

  void update_map_shift(const Eigen::Isometry3d &gicp_pose);
};