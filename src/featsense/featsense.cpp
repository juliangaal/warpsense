//
// Created by julian on 14.10.22.
//

// ROS
#include <thread>
#include <csignal>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

// local lib
#include "featsense/buffers.h"
#include "featsense/mapping.h"
#include "featsense/lidar_processing.h"
#include "featsense/odom_estimation.h"
#include "params/params.h"
#include "util/runner.h"
#include "featsense/visualization.h"

Buffers buffers;

std::function<void(int)> sigint_callback;
void sigint_handler(int value)
{
  sigint_callback(value);
}

void setup_sigint_handler(Mapping& mapping)
{
  sigint_callback = std::bind(&Mapping::terminate,
                              &mapping,
                              std::placeholders::_1);

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = sigint_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  buffers.cloud_prep_buffer->push_nb(msg);

  // publish original cloud in map frame
  auto map = *msg;
  map.header.stamp = msg->header.stamp;
  map.header.frame_id = "base_link";
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "odom_estimation_node");
  ros::NodeHandle nh("~");
  Params params(nh);

  LidarProcessing lidar_processing(nh, params, buffers);
  OdomEstimation odom_estimation(nh, params, buffers);
  Mapping mapping(nh, params, buffers);
  Visualization visualization(nh, params, buffers);

  ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>(params.floam.pcl_topic, 100, cloud_callback);

  ThreadRunner lidar_runner(lidar_processing);
  ThreadRunner odom_runner(odom_estimation);
  ThreadRunner mapping_runner(mapping);
  ThreadRunner visualization_runner(visualization);

  setup_sigint_handler(mapping);

  ros::spin();

  return 0;
}