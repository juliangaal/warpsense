// Original Copyright
// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
//
// Modification Copyright: Julian Gaal
#pragma once

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>

// local lib
#include "mypcl.h"
#include "featsense/buffers.h"
#include "util/concurrent_ring_buffer.h"
#include "util/background_thread.h"
#include "params/params.h"


struct Feature
{
  Feature() : idx(0), curvature(0) {}
  Feature(size_t idx, float curvature) : idx(idx), curvature(curvature) {}
  size_t idx;
  float curvature;
};

class LidarProcessing : public BackgroundThread
{
public:
  explicit LidarProcessing(ros::NodeHandle &nh, const Params &params, Buffers& cloud_buf);

  void process_lidar(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_in,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_cloud);

  void thread_run() final;

private:
  const Params& params_;
  CloudBuffer::Ptr& cloud_buf_;
  PCLBuffer::Ptr& vgicp_cloud_buf_;
  SurfBuffer::Ptr& surf_buf_;
  EdgeBuffer::Ptr& edge_buf_;
  HeaderBuffer::Ptr& header_buf_;
  HeaderBuffer::Ptr& vgicp_header_buf_;

  std::vector<Feature> features_;
  std::vector<float> ranges_;
  std::vector<bool> neighbor_picked_;

  ros::Publisher cloud_filtered_pub_;
  ros::Publisher edge_point_pub_;
  ros::Publisher surf_points_pub_;

  void mark_occluded_points();

  void calculate_curvature(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);

  void extract_features(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_cloud);
};
