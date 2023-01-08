#include "warpsense/cuda/cleanup.h"
#include "warpsense/tsdf_mapping.h"
#include "warpsense/registration/util.h"
#include "warpsense/visualization/map.h"
#include "params/params.h"

#include <pcl/filters/voxel_grid.h>

using namespace std::chrono_literals;

namespace cuda
{
TSDFMapping::TSDFMapping(ros::NodeHandle &nh, const Params &params,
                         ConcurrentRingBuffer<Eigen::Matrix4f>::Ptr &pose_buffer, HDF5LocalMap::Ptr &local_map)
    : pose_buffer_{pose_buffer}
    , hdf5_local_map_{local_map}
    , cuda_map_(hdf5_local_map_)
    , tsdf_{std::make_unique<TSDFCuda>(cuda_map_, params.map.tau, params.map.max_weight, params.map.resolution)}
    , mutex_{}
    , map_shift_thread_{}
    , tsdf_map_pub_{nh.advertise<visualization_msgs::Marker>("/tsdf", 0)}
    , params_(params)
    , map_shift_run_cond_(true)
    , shifted_(false)
    , is_shifting_(false)
{
  map_shift_thread_ = std::thread(&TSDFMapping::map_shift, this);
}

TSDFMapping::TSDFMapping(const Params &params, HDF5LocalMap::Ptr &local_map)
    : hdf5_local_map_{local_map}
    , cuda_map_(hdf5_local_map_)
    , tsdf_{std::make_unique<TSDFCuda>(cuda_map_, params.map.tau, params.map.max_weight, params.map.resolution)}
    , mutex_{}
    , map_shift_thread_{}
    , params_(params)
    , map_shift_run_cond_(true)
    , shifted_(false)
    , is_shifting_(false)
{
}

TSDFMapping::~TSDFMapping()
{
  join_mapping_thread();
  tsdf_.reset();
  cuda::cleanup();
}

void TSDFMapping::join_mapping_thread()
{
  map_shift_run_cond_ = false;

  if (pose_buffer_ != nullptr && map_shift_thread_.joinable())
  {
    pose_buffer_->clear();
    map_shift_thread_.join();
  }

}

void TSDFMapping::update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const Eigen::Matrix4f &pose)
{
  rmagine::Pointi pos_rm;
  rmagine::Pointi up_rm;
  convert_pose_to_gpu(pose, pos_rm, up_rm);
  std::unique_lock lock(mutex_);
  tsdf_->update_tsdf(scan_points, pos_rm, up_rm);
}

void TSDFMapping::update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const rmagine::Pointi &pos_rm, const rmagine::Pointi &up_rm)
{
  std::unique_lock lock(mutex_);
  tsdf_->update_tsdf(scan_points, pos_rm, up_rm);
}

void TSDFMapping::convert_pose_to_gpu(const Eigen::Matrix4f &pose, rmagine::Pointi &pos_rm, rmagine::Pointi &up_rm) const
{
  Eigen::Matrix4i rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  const Eigen::Vector3i up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);
  const Eigen::Vector3i pos = to_map(pose, params_.map.resolution);
  pos_rm= *reinterpret_cast<const rmagine::Pointi *>(&pos);
  up_rm= *reinterpret_cast<const rmagine::Pointi *>(&up);
}

void TSDFMapping::update_tsdf(cuda::DeviceMap &result, const std::vector<rmagine::Pointi> &scan_points,
                              const Eigen::Matrix4f &pose)
{
  rmagine::Pointi pos_rm;
  rmagine::Pointi up_rm;
  convert_pose_to_gpu(pose, pos_rm, up_rm);
  std::unique_lock lock(mutex_);
  tsdf_->update_tsdf(result, scan_points, pos_rm, up_rm);
}

void TSDFMapping::map_shift()
{
  Eigen::Matrix4f last_shift_pose = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f current_pose = last_shift_pose;

  while (ros::ok() && map_shift_run_cond_)
  {
    if (pose_buffer_->pop_nb(&current_pose))
    {
      auto distance_shift = ((last_shift_pose.block<3, 1>(0, 3) / 1000.f) -
                             (current_pose.block<3, 1>(0, 3) / 1000.f)).norm();

      if (distance_shift >= params_.map.shift)
      {
        is_shifting_ = true;
        auto curr_t = current_pose.block<3, 1>(0, 3);
        last_shift_pose = current_pose;
        DeviceMap existing_cuda_map(hdf5_local_map_);
        mutex_.lock_shared();
        tsdf_->avg_map().to_host(existing_cuda_map);
        mutex_.unlock_shared();
        const Eigen::Vector3i pos = to_map(current_pose, params_.map.resolution);
        hdf5_local_map_->shift(pos);
        ROS_WARN_STREAM("shifted map");
        mutex_.lock();
        tsdf_->avg_map().to_device(existing_cuda_map);
        tsdf_->new_map().update_params(existing_cuda_map);
        mutex_.unlock();
        shifted_ = true;
        is_shifting_ = false;
      }
    }
    else
    {
      std::chrono::milliseconds dura(2);
      std::this_thread::sleep_for(dura);
    }
    publish_local_map_skeleton(tsdf_map_pub_, *hdf5_local_map_, params_.map.tau, params_.map.resolution);
  }
}

void TSDFMapping::get_tsdf_map(DeviceMap &dev_map, HDF5LocalMap::Ptr &cpu_map)
{
  std::shared_lock lock(mutex_);
  DeviceMap existing_cuda_map(hdf5_local_map_);
  tsdf_->avg_map().to_host(existing_cuda_map);
}

void TSDFMapping::preprocess_from_ros(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose, std::vector<rmagine::Pointi>& points_rm, Eigen::Matrix4f& mm_pose)
{
  static pcl::VoxelGrid<pcl::PointXYZI> sor;
  static pcl::PointCloud<pcl::PointXYZI> transformed_pcl;
  sor.setInputCloud(cloud);
  sor.setLeafSize(params_.map.resolution / 1000.f, params_.map.resolution / 1000.f, params_.map.resolution / 1000.f);
  sor.filter(transformed_pcl);

  points_rm.resize(transformed_pcl.size());
#pragma omp parallel for schedule(static) default(shared)
  for (int i = 0; i < transformed_pcl.size(); ++i)
  {
    const auto& cp = transformed_pcl.points[i];
    points_rm[i] = rmagine::Pointi(cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f);
  }

  mm_pose.block<3, 3>(0, 0) = Eigen::Quaterniond(pose.rotation()).toRotationMatrix().cast<float>();
  mm_pose.block<3, 1>(0, 3) = (pose.translation() * 1000.0).cast<float>();
}

void TSDFMapping::update_tsdf_from_ros(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose)
{
  Eigen::Matrix4f mm_pose = Eigen::Matrix4f::Identity();
  std::vector<rmagine::Pointi> points_rm(cloud->size());
  preprocess_from_ros(cloud, pose, points_rm, mm_pose);

  pose_buffer_->push_nb(mm_pose);
  update_tsdf(points_rm, mm_pose);
}

void TSDFMapping::update_tsdf_from_ros(cuda::DeviceMap& result, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const Eigen::Isometry3d &pose)
{
  Eigen::Matrix4f mm_pose = Eigen::Matrix4f::Identity();
  std::vector<rmagine::Pointi> points_rm(cloud->size());
  preprocess_from_ros(cloud, pose, points_rm, mm_pose);

  pose_buffer_->push_nb(mm_pose);
  update_tsdf(result, points_rm, mm_pose);
}

const std::unique_ptr<cuda::TSDFCuda>& TSDFMapping::tsdf() const
{
  return tsdf_;
}

std::unique_ptr<cuda::TSDFCuda>& TSDFMapping::tsdf()
{
  return tsdf_;
}

std::atomic<bool>& TSDFMapping::shifted()
{
  return shifted_;
}

std::atomic<bool> &TSDFMapping::is_shifting()
{
  return is_shifting_;
}

}
