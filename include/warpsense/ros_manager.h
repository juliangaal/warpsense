#pragma once
#include <ros/ros.h>
#include <Eigen/Dense>
#include "warpsense/map/hdf5_local_map.h"
#include "util/concurrent_ring_buffer.h"

class ROSManager
{
  ROSManager() = default;
  ~ROSManager() = default;
  void empty_queues();
  std::unique_ptr<ros::Subscriber> sub_cloud_;
  std::unique_ptr<ros::Subscriber> sub_imu_;
  ros::Publisher tsdf_map_pub_;
  ros::Publisher path_pub_;
  ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr pose_buffer_;
  ConcurrentRingBuffer<HDF5LocalMap::Ptr>::Ptr map_buffer_;
  void publish();
};

