#pragma once
#include <pcl/point_cloud.h>
#include <pcl/pcl_config.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/publisher.h>
#include <Eigen/Dense>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <queue>

template <typename T>
inline void publish_cloud(const pcl::PointCloud<T> &cloud, ros::Publisher &pub, const std_msgs::Header &header, const std::string& frame = "map")
{
  sensor_msgs::PointCloud2 ros_pcl;
  pcl::toROSMsg(cloud, ros_pcl);
  ros_pcl.header.stamp = header.stamp;
  ros_pcl.header.frame_id = frame;
  pub.publish(ros_pcl);
}

template <typename T>
inline void clear(std::queue<T>& q)
{
  std::queue<T> empty;
  std::swap(q, empty);
}

inline void publish_path(const Eigen::Isometry3d& odom_estimation, ros::Publisher& pub, const std_msgs::Header& header)
{
  // RViz crashes if navigation_msgs::Path is longer than 16384 (~13 minutes at 20 Scans/sec)
  // see https://github.com/ros-visualization/rviz/issues/1107
  // using pointcloud instead

  pcl::PointCloud<pcl::PointXYZ>::Ptr path_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  auto pos = odom_estimation.translation();

#if PCL_VERSION_COMPARE(<, 1, 7, 4)
  path_cloud->push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
#else
  path_cloud->emplace_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
#endif
  publish_cloud(*path_cloud, pub, header);
}

inline void publish_path(const geometry_msgs::Transform& transform, ros::Publisher& pub, const std_msgs::Header& header)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr path_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  auto pos = transform.translation;
#if PCL_VERSION_COMPARE(<, 1, 7, 4)
  path_cloud->push_back(pcl::PointXYZ(pos.x, pos.y, pos.z));
#else
  path_cloud->emplace_back(pcl::PointXYZ(pos.x, pos.y, pos.z));
#endif
  publish_cloud(*path_cloud, pub, header);
}

inline void publish_path(const geometry_msgs::TransformStamped& transform, ros::Publisher& pub)
{
  publish_path(transform.transform, pub, transform.header);
}

inline void publish_path(const Eigen::Quaterniond& q, const Eigen::Vector3d& t, ros::Publisher& pub, const std_msgs::Header& header)
{
  auto pose = Eigen::Isometry3d::Identity();
  pose.rotate(q);
  pose.pretranslate(t);
  publish_path(pose, pub, header);
}

template <typename T>
inline void publish_odometry(const Eigen::Quaternion<T>& q, const Eigen::Vector3d& t, ros::Publisher& pub)
{
  nav_msgs::Odometry odom;
  odom.header.frame_id = "map";
  odom.child_frame_id = "base_link";
  odom.header.stamp = ros::Time::now();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.position.x = t.x();
  odom.pose.pose.position.y = t.y();
  odom.pose.pose.position.z = t.z();
  pub.publish(odom);
}

template <typename T>
inline void publish_odometry(const Eigen::Transform<T, 3, 1>& pose, ros::Publisher& pub)
{
  Eigen::Quaterniond q = Eigen::Quaterniond(pose.rotation());
  Eigen::Vector3d t(pose.translation());
  publish_odometry(q, t, pub);
}