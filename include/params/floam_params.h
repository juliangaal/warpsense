#pragma once

#include "params_base.h"

/**
  * @file params.h
  * @author julian
  * @date 10/17/21
 */


struct FloamParams : ParamsBase
{
  FloamParams() = delete;

  explicit FloamParams(const ros::NodeHandle &nh)
  {
    FloamParams::load(nh);
  }

  void load(const ros::NodeHandle &nh)
  {
    nh.param<std::string>("floam/pcl_topic", pcl_topic, "/velodyne_points");
    nh.param<float>("floam/edge_resolution", edge_resolution, 0.4);
    nh.param<float>("floam/edge_threshold", edge_threshold, 2.0);
    nh.param<float>("floam/surf_resolution", surf_resolution, 0.8);
    nh.param<float>("floam/surf_threshold", surf_threshold, 0.1);
    nh.param<float>("floam/min_distance", min_distance, 2.f);
    nh.param<float>("floam/max_distance", max_distance, 60.f);
    nh.param<int>("floam/enrich", enrich, 3);
    nh.param<int>("floam/optimization_steps", optimization_steps, 2);
    nh.param<float>("floam/vgicp_fitness_score", vgicp_fitness_score, 6.0);
  }

  std::string pcl_topic;
  float edge_resolution;
  float edge_threshold;
  float surf_resolution;
  float surf_threshold;
  int optimization_steps;
  float min_distance;
  float max_distance;
  int enrich;
  float vgicp_fitness_score;
};


