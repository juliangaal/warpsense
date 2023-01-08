#pragma once
#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <shared_mutex>

#include "cuda/device_map.h"
#include "cuda/update_tsdf.h"
#include "cuda/registration.h"
#include "util/concurrent_ring_buffer.h"
#include "params/params.h"
#include <ros/ros.h>

namespace cuda
{

class TSDFMapping
{
public:
  explicit TSDFMapping(ros::NodeHandle &nh, const Params &params,
                       ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr &pose_buffer, HDF5LocalMap::Ptr &local_map);
  virtual ~TSDFMapping();
  void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const rmagine::Pointi &pos_rm, const rmagine::Pointi &up_rm);
  void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const Eigen::Matrix4f &pose);
  void update_tsdf_from_ros(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose);
  void update_tsdf_from_ros(cuda::DeviceMap &result, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose);
  void preprocess_from_ros(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose, std::vector<rmagine::Pointi>& points_rm, Eigen::Matrix4f& mm_pose);
  void update_tsdf(cuda::DeviceMap &result, const std::vector<rmagine::Pointi> &scan_points,
                   const Eigen::Matrix4f &pose);
  const std::unique_ptr<cuda::TSDFCuda>& tsdf() const;
  std::unique_ptr<cuda::TSDFCuda>& tsdf();
  void join_mapping_thread();
  std::atomic<bool>& shifted();
  std::atomic<bool>& is_shifting();

private:
  void map_shift();
  ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr pose_buffer_;
  HDF5LocalMap::Ptr& hdf5_local_map_;
  std::thread map_shift_thread_;
  ros::Publisher tsdf_map_pub_;
  std::atomic<bool> map_shift_run_cond_;
  std::atomic<bool> shifted_;
  std::atomic<bool> is_shifting_;
  void get_tsdf_map(DeviceMap &dev_map, HDF5LocalMap::Ptr &cpu_map);
  void convert_pose_to_gpu(const Eigen::Matrix4f &pose, rmagine::Pointi &pos_rm, rmagine::Pointi &up_rm) const;

protected:
  explicit TSDFMapping(const Params &params, HDF5LocalMap::Ptr &local_map);
  const Params& params_;
  cuda::DeviceMap cuda_map_;
  std::unique_ptr<cuda::TSDFCuda> tsdf_;
  std::shared_mutex mutex_;
};

}
