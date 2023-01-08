// Original Copyright:
// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
//
// Modification Copyright: Julian Gaal

#pragma once

// std lib
#include <string>
#include <math.h>
#include <vector>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

// ROS
#include <tf/transform_broadcaster.h>
#include <ros/node_handle.h>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Local lib
#include "lidar_optimization.h"
#include "featsense/buffers.h"
#include "util/background_thread.h"
#include "params/params.h"
#include "util/nanoflann_pcl.h"

class OdomEstimation : public BackgroundThread
{

public:
  OdomEstimation(ros::NodeHandle& nh, const Params &params, Buffers& buffers);

  void initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_in,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_in);

  void updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_in,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_in);

  void thread_run() final;

private:
  const Params params_;
  SurfBuffer::Ptr& surf_buf_;
  EdgeBuffer::Ptr& edge_buf_;
  OdomBuffer::Ptr& odom_buf_;
  OdomBuffer::Ptr& vgicp_odom_buf_;

  Eigen::Isometry3d odom;
  pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerMap;
  pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfMap;

  //optimization variable
  double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
  Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(parameters);
  Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(parameters + 4);

  Eigen::Isometry3d last_odom;

  //kd-tree
  nanoflann::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeEdgeMap;
  nanoflann::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfMap;

  //points downsampling before add to map
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterEdge;
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf;

  //local map
  pcl::CropBox<pcl::PointXYZI> cropBoxFilter;

  //optimization count 
  int optimization_count;

  //function
  void addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr &map_in, ceres::Problem &problem,
                         ceres::LossFunction *loss_function);

  void addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr &map_in, ceres::Problem &problem,
                         ceres::LossFunction *loss_function);

  void addPointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &downsampledEdgeCloud,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &downsampledSurfCloud);

  void pointAssociateToMap(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po);

  void downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_pc_in,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_pc_out,
                         const pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc_in,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_pc_out);
};
