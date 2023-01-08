// ros
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <util/runtime_evaluator.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <ros/callback_queue.h>

// C++
#include <thread>
#include <csignal>
#include <unordered_set>
#include <boost/filesystem.hpp>

// warpsense lib
#include "map/map.h"
#include "util/util.h"
#include "cpu/update_tsdf.h"
#include "util/publish.h"
#include "params/params.h"
#include "warpsense/registration/util.h"
#include "warpsense/registration/registration.h"
#include "warpsense/visualization/map.h"

/// Global map representation
std::shared_ptr<HDF5GlobalMap> global_map_;
/// Local map representation
std::shared_ptr<HDF5LocalMap> local_map_;

std::unique_ptr<tf2_ros::TransformBroadcaster> tfb_;

constexpr int MAP_RESOLUTION = 64;

// tsdf update thread
std::thread suv_thread_;
std::atomic<bool> suv_thread_running_ = false;
std::atomic<bool> sigint_reached = false;
std::mutex suv_mutex_;

std::unique_ptr<ros::Subscriber> sub_cloud_;
std::unique_ptr<ros::Subscriber> sub_imu_;
ros::Publisher tsdf_map_pub_;
ros::Publisher path_pub_;

std::mutex imu_buf_lock;
std::queue<sensor_msgs::ImuConstPtr> imu_buf;

std::unique_ptr<Params> params;

Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();
Eigen::Matrix4f last_updated_pose_ = Eigen::Matrix4f::Identity();

namespace fs = boost::filesystem;

void mySigintHandler(int sig)
{
  ROS_INFO_STREAM("Completing map update");
  if (suv_thread_.joinable())
  {
    suv_thread_.join();
  }

  ROS_INFO_STREAM("Writing map into " << global_map_->filename());
  local_map_->write_back();

  // force emptying of queues, so that no more runtime_evaluator form is active and can be written to csv
  sub_cloud_.reset(nullptr);
  sub_imu_.reset(nullptr);

#ifdef TIME_MEASUREMENT
  fs::path filename = (fs::path(DATA_PATH) / params->map.identifier()).replace_extension("csv");
  ROS_INFO_STREAM("Saving runtime evaluator data in " << filename);
  auto& runtime_eval_ = RuntimeEvaluator::get_instance();
  if (!runtime_eval_.export_results(filename))
  {
    ROS_ERROR_STREAM("Couldn't save runtime evaluator stats");
  }
#endif

  ROS_INFO_STREAM("Bye");

  ros::shutdown();
}

void shift_update_visualize(const Eigen::Matrix4f pose, const std::vector<Point> points, const Params params)
{
  suv_thread_running_ = true;
  // Create a local copy of the ring buffer / local map
  auto suv_buffer_ptr = std::make_shared<HDF5LocalMap>(*local_map_);

  // Shift
  Point pos = pose.block<3, 1>(0, 3).cast<int>() / MAP_RESOLUTION;
  suv_buffer_ptr->shift(pos);

  Eigen::Matrix4i rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);

  // Update
  update_tsdf(points, pos, up, *suv_buffer_ptr, params.map.tau, params.map.max_weight, params.map.resolution);

  suv_mutex_.lock();
  // Update the pointer (global member for registration) to the locally updated ring buffer / local map
  local_map_ = suv_buffer_ptr;
  last_updated_pose_ = pose;
  suv_mutex_.unlock();

  // Visualize
  std_msgs::Header header;
  header.frame_id = "map";
  header.stamp = ros::Time::now();
  publish_local_map(tsdf_map_pub_, *suv_buffer_ptr, params.map.tau, params.map.resolution);

  suv_thread_running_ = false;
}

void imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  imu_buf_lock.lock();
  imu_buf.push(msg);
  imu_buf_lock.unlock();
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud)
{
  static sensor_msgs::Imu last_imu;

#ifdef TIME_MEASUREMENT
  auto& runtime_eval_ = RuntimeEvaluator::get_instance();
  runtime_eval_.start("total");
#endif

  std::vector<Point> scan_points;
  scan_points.reserve(30'000);

#ifdef TIME_MEASUREMENT
  runtime_eval_.start("preprocess");
#endif

  std::unordered_set<Point> point_set;
  point_set.reserve(30'000);
  Eigen::Matrix4i next_transform = to_int_mat(pose_);
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  for (; iter_x != iter_x.end(); ++iter_x)
  {
    Point point;
    if (iter_x[0] < 0.3 && iter_x[1] < 0.3 && iter_x[2] < 0.3)
    {
      continue;
    }
    for (size_t i = 0; i < 3; i++)
    {
      point[i] = iter_x[i] * 1000;
    }
    if (point_set.insert(point / MAP_RESOLUTION).second)
    {
      scan_points.push_back(transform_point(point, next_transform));
    }
  }

#ifdef TIME_MEASUREMENT
  runtime_eval_.stop("preprocess");
#endif

  static bool first_iteration = true;
  if (first_iteration)
  {
    Point up(0, 0, MATRIX_RESOLUTION);
    update_tsdf(scan_points, pose_.block<3, 1>(0, 3).cast<int>(), up, *local_map_, params->map.tau, params->map.max_weight, params->map.resolution);
    first_iteration = false;
  }
  else
  {

#ifdef TIME_MEASUREMENT
    runtime_eval_.start("registration");
#endif
    imu_buf_lock.lock();
    sensor_msgs::Imu curr_imu;
    auto now = ros::Time::now();
    auto diff = now - now;

    //static int skips = 0;

    while (diff.toNSec() <= 0 && !imu_buf.empty())
    {
      auto imu_stamp = imu_buf.front()->header.stamp;
      imu_buf.pop();
      diff = imu_stamp - cloud->header.stamp;
      //skips++;
    }

    if (!imu_buf.empty())
    {
      curr_imu = *imu_buf.front();
    }
    imu_buf_lock.unlock();

    //ROS_INFO_STREAM("Skipped : " << skips);
    //skips = 0;

    auto last_q = Eigen::Quaterniond(last_imu.orientation.w, last_imu.orientation.x,
                                     last_imu.orientation.y, last_imu.orientation.z);
    auto curr_q = Eigen::Quaterniond(curr_imu.orientation.w, curr_imu.orientation.x,
                                     curr_imu.orientation.y, curr_imu.orientation.z);

    Eigen::Matrix4f pretransform = Eigen::Matrix4f::Identity();
    pretransform.block<3, 3>(0, 0) = (curr_q * last_q.inverse()).cast<float>().toRotationMatrix();

    suv_mutex_.lock();
    auto transform = register_cloud(*local_map_, scan_points,
                                    pretransform, params->registration.max_iterations,
                                    params->registration.it_weight_gradient, params->registration.epsilon, 0);
    suv_mutex_.unlock();

    auto rotation = transform.block<3, 3>(0, 0) * pose_.block<3, 3>(0, 0);
    pose_.block<3, 3>(0, 0) = rotation;
    pose_.block<3, 1>(0, 3) += transform.block<3, 1>(0, 3);


#ifdef TIME_MEASUREMENT
    runtime_eval_.stop("registration");
#endif

    // Shift, update and visualize in separate thread
    /*
    TODO: Condition
    1. After a set amount of steps
    2. After a set distance moved
    -> Parameterized
    */
    /*
    TODO: In fastsense repository use and reuse one single Runner.
    The thread / Runner should loop and wait for the calls
    */
    static size_t reg_cnt_ = 0;

    auto last_translation = last_updated_pose_.block<3, 1>(0, 3).cast<int>();
    auto current_translation = pose_.block<3, 1>(0, 3).cast<int>();
    double distance = (last_translation - current_translation).norm();
    static int rep = 0;
    static float sum_distance = 0;
    if (((++reg_cnt_ >= 100) || distance > 0.25) && !suv_thread_running_)
    {
      if (suv_thread_.joinable())
      {
        suv_thread_.join();
      }
      suv_thread_ = std::thread(shift_update_visualize, pose_, scan_points, *params);
      reg_cnt_ = 0;
    }
  }

  Vector3f meter_pos = pose_.block<3, 1>(0, 3) / 1000.f;
  Eigen::Matrix3f rotation = pose_.block<3, 3>(0, 0);
  Vector3f rot = rotation.eulerAngles(0, 1, 2);
  //suv_mutex_.lock();
  //global_map_ptr_->write_pose(meter_pos.x(), meter_pos.y(), meter_pos.z(), rot.x(), rot.y(), rot.z());
  //suv_mutex_.unlock();

  Eigen::Quaternionf q(rotation);
  q.normalize();
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "map";
  transformStamped.child_frame_id = params->registration.link;
  transformStamped.transform.translation.x = meter_pos.x();
  transformStamped.transform.translation.y = meter_pos.y();
  transformStamped.transform.translation.z = meter_pos.z();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();
  transformStamped.transform.rotation.w = q.w();
  tfb_->sendTransform(transformStamped);

#ifdef TIME_MEASUREMENT
  runtime_eval_.stop("total");
  std::cout << runtime_eval_ << std::endl;
#endif

  publish_path(transformStamped, path_pub_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "warpsense_cpu");
  ros::NodeHandle nh("~");

  params.reset(new Params(nh));

  sub_cloud_ = std::make_unique<ros::Subscriber>(
      nh.subscribe<sensor_msgs::PointCloud2>(params->registration.lidar_topic, 100, cloud_callback));
  sub_imu_ = std::make_unique<ros::Subscriber>(nh.subscribe<sensor_msgs::Imu>(params->registration.imu_topic, 100, imu_callback));
  tsdf_map_pub_ = nh.advertise<visualization_msgs::Marker>("/tsdf", 0);
  path_pub_ = nh.advertise<sensor_msgs::PointCloud2>("path", 0);
  //map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 100);
  //gicp_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/gicp_map", 100);
  //path_pub = nh.advertise<sensor_msgs::PointCloud2>("/path", 0);
  //corr_path_pub = nh.advertise<sensor_msgs::PointCloud2>("/corrected_path", 0);
  //opub = nh.advertise<nav_msgs::Odometry>("/odom_refinement", 1000);

  global_map_.reset(new HDF5GlobalMap(params->map));
  local_map_.reset(new HDF5LocalMap(params->map.size.x(), params->map.size.y(), params->map.size.z(), global_map_));
  tfb_.reset(new tf2_ros::TransformBroadcaster());

  // Override the default ros sigint handler.
  // This must be set after the first NodeHandle is created.
  signal(SIGINT, mySigintHandler);

  ros::spin();

  return 0;
}