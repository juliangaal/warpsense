/*

MIT License

Copyright (c) 2023 Julian Gaal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 

*/

#include "warpsense/app.h"
#include "util/runtime_evaluator.h"
#include "util/publish.h"
#include "warpsense/cuda/cleanup.h"
#include "warpsense/preprocessing.h"

namespace warpsense
{

App::App(ros::NodeHandle &nh, const Params &params)
    : params_(params)
    , hdf5_global_map_{new HDF5GlobalMap(params.map)}
    , hdf5_local_map_{new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), hdf5_global_map_)}
    , pose_buffer_{std::make_shared<ConcurrentRingBuffer<Eigen::Matrix4f>>(1)}
    , gpu_{nh, params_, pose_buffer_, hdf5_local_map_}
    , sub_cloud_{std::make_unique<ros::Subscriber>(
          nh.subscribe<sensor_msgs::PointCloud2>(params.registration.lidar_topic, 100, &App::cloud_callback, this))}
    , sub_imu_{std::make_unique<ros::Subscriber>(
          nh.subscribe<sensor_msgs::Imu>(params.registration.imu_topic, 100, &App::imu_callback, this))}
    , path_pub_{nh.advertise<sensor_msgs::PointCloud2>("path", 0)}, pose_{Eigen::Matrix4f::Identity()}
    , pcl_pub_{nh.advertise<sensor_msgs::PointCloud2>("pcl", 0)}
    , last_tsdf_pose_{Eigen::Matrix4f::Identity()}, imu_buffer_{new ConcurrentRingBuffer<sensor_msgs::Imu>(1000)}
    , filter_{10}, imu_acc_{imu_buffer_}, tfb_()
    , initialized_(false)
    , terminated_(false)
{}

void App::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  const auto filtered_msg = filter_.update(
      Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z));
  auto imu = *msg;
  imu.angular_velocity.x = filtered_msg.x();
  imu.angular_velocity.y = filtered_msg.y();
  imu.angular_velocity.z = filtered_msg.z();
  imu_buffer_->push_nb(imu, true);
}

void App::cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
#ifdef TIME_MEASUREMENT
  auto &runtime_eval_ = RuntimeEvaluator::get_instance();
  runtime_eval_.start("total");
#endif
  std::vector<rmagine::Pointi> scan_points;
  scan_points.reserve(30'000);
  preprocess(cloud, scan_points);

  auto distance_tsdf = ((last_tsdf_pose_.block<3, 1>(0, 3) / 1000.f) - (pose_.block<3, 1>(0, 3) / 1000.f)).norm();

  if (!initialized_ || distance_tsdf > 0.3f || gpu_.shifted())
  {
    initialized_ = true;
    last_tsdf_pose_ = pose_;
#ifdef TIME_MEASUREMENT
    runtime_eval_.start("tsdf");
#endif
    //cuda::DeviceMap avg_cuda_map(hdf5_local_map_);
    gpu_.update_tsdf(scan_points, pose_);
#ifdef TIME_MEASUREMENT
    runtime_eval_.stop("tsdf");
#endif

    if (gpu_.shifted())
    {
      gpu_.shifted() = false;
    }

    pcl_pub_.publish(cloud);
  }

  Eigen::Matrix4f pretransform = imu_acc_.acc_transform(cloud->header.stamp).cast<float>();
#ifdef TIME_MEASUREMENT
  runtime_eval_.start("registration");
#endif
  auto transform = gpu_.register_cloud(scan_points, pretransform);
#ifdef TIME_MEASUREMENT
  runtime_eval_.stop("registration");
#endif
  update_pose_estimate(transform);
  publish_pose_estimate(cloud->header.stamp);
  hdf5_global_map_->write_pose(pose_, 1000.f);
#ifdef TIME_MEASUREMENT
  runtime_eval_.stop("total");
#endif
#ifdef TIME_MEASUREMENT
  std::cout << runtime_eval_;
#endif

  pose_buffer_->push_nb(pose_);
}

void
App::preprocess(const sensor_msgs::PointCloud2ConstPtr &cloud, std::vector<rmagine::Pointi> &scan_points) const
{
  std::unordered_set<rmagine::Pointi> point_set;
  point_set.reserve(30'000);
  Eigen::Matrix4i next_transform = to_int_mat(pose_);
  const auto& map_resolution = params_.map.resolution;

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  for (; iter_x != iter_x.end(); ++iter_x)
  {
    if (iter_x[0] < 0.3 && iter_x[1] < 0.3 && iter_x[2] < 0.3)
    {
      continue;
    }

    rmagine::Pointf point(iter_x[0] * 1000.f, iter_x[1] * 1000.f, iter_x[2] * 1000.f);
    rmagine::Vector3i voxel_center
        {
            static_cast<int>(std::floor(point.x / map_resolution) * map_resolution + map_resolution / 2),
            static_cast<int>(std::floor(point.y / map_resolution) * map_resolution + map_resolution / 2),
            static_cast<int>(std::floor(point.z / map_resolution) * map_resolution + map_resolution / 2)
        };

    point_set.insert(transform_point(voxel_center, next_transform));
  }

  scan_points.resize(point_set.size());
  std::copy(point_set.begin(), point_set.end(), scan_points.begin());
}

void App::publish_pose_estimate(const ros::Time &timestamp)
{
  Vector3f meter_pos = pose_.block<3, 1>(0, 3) / 1000.f;
  Eigen::Matrix3f rotation = pose_.block<3, 3>(0, 0);

  Eigen::Quaternionf q(rotation);
  q.normalize();
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = timestamp;
  transformStamped.header.frame_id = "map";
  transformStamped.child_frame_id = params_.registration.link;
  transformStamped.transform.translation.x = meter_pos.x();
  transformStamped.transform.translation.y = meter_pos.y();
  transformStamped.transform.translation.z = meter_pos.z();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();
  transformStamped.transform.rotation.w = q.w();
  tfb_.sendTransform(transformStamped);
  publish_path(transformStamped, path_pub_);
}

void App::update_pose_estimate(Eigen::Matrix4f &transform)
{
  pose_.block<3, 3>(0, 0) = transform.block<3, 3>(0, 0) * pose_.block<3, 3>(0, 0);
  pose_.block<3, 1>(0, 3) += transform.block<3, 1>(0, 3);
}

App::~App()
{
  // can't call terminate() directly, unfortunately
  // ROS may call destructor, too and therefore call terminate() is called twice (ctrl+c, RAII)
  // without checking
  if (!terminated_)
  {
    terminate();
    terminated_ = true;
  }
}

void App::terminate(int signal)
{
  // force emptying of queues, so that no more runtime_evaluator form is active and can be written to csv
  sub_cloud_.reset(nullptr);
  sub_imu_.reset(nullptr);
  imu_buffer_->clear();
  pose_buffer_->clear();

  gpu_.join_mapping_thread();
  ROS_INFO_STREAM("Joined mapping thread");

  ROS_INFO_STREAM("Waiting for all callbacks to finish...");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  if (initialized_)
  {
#ifdef TIME_MEASUREMENT
    fs::path filename = (fs::path(DATA_PATH) / params_.map.identifier()).replace_extension("csv");
    ROS_INFO_STREAM("Saving runtime evaluator data in " << filename);
    auto &runtime_eval_ = RuntimeEvaluator::get_instance();
    if (!runtime_eval_.export_results(filename))
    {
      ROS_ERROR_STREAM("Couldn't save runtime evaluator stats");
    }
#endif

    ROS_INFO_STREAM("Writing map into " << hdf5_global_map_->filename());
    cuda::pause();
    cuda::DeviceMap avg_cuda_map(hdf5_local_map_);
    gpu_.tsdf()->avg_map().to_host(avg_cuda_map);
    hdf5_local_map_->write_back();
    ROS_INFO_STREAM("Completed");
  }

  ros::shutdown();
}

} // end namespace warpsense
