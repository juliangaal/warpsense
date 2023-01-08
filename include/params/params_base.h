#pragma once

#include <ros/node_handle.h>

struct ParamsBase
{
  ParamsBase() = default;
  virtual ~ParamsBase() = default;
  virtual void load(const ros::NodeHandle& nh) = 0;
};