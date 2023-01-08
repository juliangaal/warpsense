#pragma once

#include "params_base.h"

/**
  * @file params.h
  * @author julian
  * @date 10/17/21
 */


struct LidarParams : public ParamsBase
{
  LidarParams() = delete;

  explicit LidarParams(const ros::NodeHandle &nh)
  {
    LidarParams::load(nh);
  }

  void load(const ros::NodeHandle &nh)
  {
    nh.param<int>("lidar/channels", channels, 128);
    nh.param<int>("lidar/reg_channels", reg_channels, 16);
    nh.param<float>("lidar/vfov", vfov, 45.f);
    nh.param<int>("lidar/frequency", frequency, 20);
    nh.param<int>("lidar/hresolution", hresolution, 1024);
  }

  int channels;
  int reg_channels;
  float vfov;
  int frequency;
  int hresolution;
};


