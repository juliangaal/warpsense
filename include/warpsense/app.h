#pragma once

// warpsense
#include "tsdf_registration.h"
#include "math/math.h"
#include "util/util.h"
#include "util/filter.h"
#include "params/params_base.h"
#include "cuda/update_tsdf.h"
#include "registration/util.h"
#include "util/imu_accumulator.h"
#include "util/concurrent_ring_buffer.h"
#include "cuda/registration.h"
#include "cuda/device_map.h"
#include "map/hdf5_local_map.h"
#include "map/hdf5_global_map.h"

// ROS
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/point_cloud2_iterator.h>

// C++
#include <unordered_set>
#include <geometry_msgs/TransformStamped.h>

namespace warpsense
{

struct App
{
  explicit App(ros::NodeHandle& nh, const Params& params);
  ~App();

  void imu_callback(const sensor_msgs::ImuConstPtr &msg);

  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud);

  void update_pose_estimate(Eigen::Matrix4f &transform);

  void publish_pose_estimate(const ros::Time &timestamp);

  void preprocess(const sensor_msgs::PointCloud2ConstPtr &cloud, std::vector<rmagine::Pointi> &scan_points) const;

  void terminate(int signal = 0);

  const Params& params_;
  std::shared_ptr<HDF5GlobalMap> hdf5_global_map_;
  std::shared_ptr<HDF5LocalMap> hdf5_local_map_;
  std::unique_ptr<ros::Subscriber> sub_cloud_;
  std::unique_ptr<ros::Subscriber> sub_imu_;
  ros::Publisher path_pub_;
  ros::Publisher pcl_pub_;
  Eigen::Matrix4f pose_;
  Eigen::Matrix4f last_tsdf_pose_;
  ConcurrentRingBuffer<sensor_msgs::Imu>::Ptr imu_buffer_;
  ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr pose_buffer_;
  cuda::TSDFRegistration gpu_;
  SlidingWindowFilter<Eigen::Vector3d> filter_;
  ImuAccumulator imu_acc_;
  tf2_ros::TransformBroadcaster tfb_;
  bool initialized_;
  bool terminated_;
};

} // end namespace warpsense