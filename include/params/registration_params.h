#pragma once

#include <string>
#include "params_base.h"

/**
  * @file params.h
  * @author julian
  * @date 10/17/21
 */


struct RegistrationParams : public ParamsBase
{
  RegistrationParams() = delete;

  explicit RegistrationParams(const ros::NodeHandle &nh)
  {
    RegistrationParams::load(nh);
  }

  void load(const ros::NodeHandle &nh)
  {
    nh.param<std::string>("registration/lidar_topic", lidar_topic, "/os_cloud_node/velodyne/points");
    nh.param<std::string>("registration/imu_topic", imu_topic, "/imu/data");
    nh.param<std::string>("registration/link", link, "os1_sensor");
    nh.param<int>("registration/max_iterations", max_iterations, 200);
    nh.param<float>("registration/it_weight_gradient", it_weight_gradient, 0.1);
    nh.param<float>("registration/epsilon", epsilon, 0.03);
  }

  int max_iterations;
  float it_weight_gradient;
  float epsilon;
  std::string lidar_topic;
  std::string imu_topic;
  std::string link;
};


