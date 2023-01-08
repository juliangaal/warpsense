
/**
  * @file visualization_node.cpp
  * @author julian 
  * @date 7/6/22
 */
#include <warpsense/preprocessing.h>
#include <iostream>
#include <warpsense/update_tsdf.h>
#include <warpsense/visualization/local_map_vis.h>
#include <pcl/point_cloud.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

constexpr int DEFAULT_VALUE = 4;
constexpr int DEFAULT_WEIGHT = 6;

bool approx(float a, float b, float epsilon)
{
  return std::abs(a - b) < epsilon;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test");
  ros::NodeHandle nh;
  auto pub = nh.advertise<visualization_msgs::Marker>("/tsdf", 0);

  Params params;

  // tsdf parameters
  float max_distance = .3;
  int map_resolution = 10;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  float map_size_x = 20 * 1000 / map_resolution;
  float map_size_y = 20 * 1000 / map_resolution;
  float map_size_z = 5 * 1000 / map_resolution;

  std::shared_ptr<GlobalMap> global_map_ptr_;
  std::shared_ptr<LocalMap> local_map_ptr_;
  global_map_ptr_.reset(new GlobalMap(params));
  local_map_ptr_.reset(new LocalMap(map_size_x, map_size_y, map_size_z, global_map_ptr_));

  // fill points
  std::vector<Eigen::Vector3f> orig_points;
  for (float x = -0.25; x < 0.25; x += (float)map_resolution / 1000.f)
  {
    for (float z = -0.25; z < 0.25; z += (float)map_resolution / 1000.f)
    {
      Eigen::Vector3f p = Eigen::Vector3f::Zero();
      p.x() = x;
      p.y() = 1;
      p.z() = z;
      orig_points.push_back(p);
    }
  }

  // first pose
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

  // shift
  int x = (int)std::floor(pose.translation().x() * 1000 / map_resolution);
  int y = (int)std::floor(pose.translation().y() * 1000 / map_resolution);
  int z = (int)std::floor(pose.translation().z() * 1000 / map_resolution);
  Eigen::Vector3i pos(x, y, z);
  local_map_ptr_->shift(pos);

  // preprocess
  std::vector<Point> scan_points;
  reduction_filter_voxel_center(orig_points, scan_points);

  // update
  Eigen::Matrix4i rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);
  update_tsdf(scan_points,pos, up, *local_map_ptr_, tau, max_weight, map_resolution);

  // move points
  float movement = .5f;
  orig_points.clear();
  scan_points.clear();
  for (float x = -0.25; x < 0.25; x += (float)map_resolution / 1000.f)
  {
    for (float z = -0.25; z < 0.25; z += (float)map_resolution / 1000.f)
    {
      Eigen::Vector3f p = Eigen::Vector3f::Zero();
      p.x() = x + movement;
      p.y() = 1;
      p.z() = z;
      orig_points.push_back(p);
    }
  }

  // move pose
  pose.pretranslate(Eigen::Vector3d(movement, 0, 0));

  // second shift
  x = (int)std::floor(pose.translation().x() * 1000 / map_resolution);
  y = (int)std::floor(pose.translation().y() * 1000 / map_resolution);
  z = (int)std::floor(pose.translation().z() * 1000 / map_resolution);
  pos = Eigen::Vector3i(x, y, z);
  local_map_ptr_->shift(pos);

  // preprocess
  reduction_filter_voxel_center(orig_points, scan_points);

  // second update
  rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  up = transform(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);
  update_tsdf(scan_points,pos, up, *local_map_ptr_, tau, max_weight, map_resolution);

  std_msgs::Header header;
  header.frame_id = "map";
  header.stamp = ros::Time::now();
  ros::Rate r(1.0);
  while (ros::ok())
  {
    r.sleep();
    visualize_local_map(header, pub, *local_map_ptr_, tau, map_resolution);
  }
}