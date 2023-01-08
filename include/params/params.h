#pragma once

#include "params_base.h"
#include "map_params.h"
#include "registration_params.h"
#include "floam_params.h"
#include "lidar_params.h"

/**
  * @file params.h
  * @author julian
  * @date 10/17/21
 */


struct Params : public ParamsBase
{
  Params() = delete;

  explicit Params(const ros::NodeHandle &nh)
      : map(nh), registration(nh), floam(nh), lidar(nh)
  {
  }

  void load(const ros::NodeHandle &nh)
  {

  }

  MapParams map;
  RegistrationParams registration;
  FloamParams floam;
  LidarParams lidar;
};


