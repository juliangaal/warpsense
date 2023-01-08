#pragma once

#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "util/concurrent_ring_buffer.h"

using CloudBuffer = ConcurrentRingBuffer<sensor_msgs::PointCloud2ConstPtr>;
using PCLBuffer = ConcurrentRingBuffer<pcl::PointCloud<pcl::PointXYZI>::Ptr>;
using SurfBuffer = PCLBuffer;
using EdgeBuffer = PCLBuffer;
using OdomBuffer = ConcurrentRingBuffer<Eigen::Isometry3d>;
using HeaderBuffer = ConcurrentRingBuffer<std_msgs::Header>;

struct Buffers
{
  Buffers()
      : cloud_prep_buffer(std::make_shared<CloudBuffer>(1000))
      , surf_buffer(std::make_shared<PCLBuffer>(1000))
      , edge_buffer(std::make_shared<PCLBuffer>(1000))
      , odom_buffer(std::make_shared<OdomBuffer>(1000))
      , header_buffer(std::make_shared<HeaderBuffer>(1000))
      , vgicp_header_buffer(std::make_shared<HeaderBuffer>(1))
      , vgicp_cloud_buffer(std::make_shared<PCLBuffer>(1))
      , vgicp_odom_buffer(std::make_shared<OdomBuffer>(1))
  {}

  ~Buffers() = default;
  Buffers(Buffers&) = delete;
  Buffers(Buffers&&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  Buffers& operator=(Buffers&&) = delete;

  CloudBuffer::Ptr cloud_prep_buffer;
  SurfBuffer::Ptr surf_buffer;
  EdgeBuffer::Ptr edge_buffer;
  OdomBuffer::Ptr odom_buffer;
  HeaderBuffer::Ptr header_buffer;
  HeaderBuffer::Ptr vgicp_header_buffer;
  PCLBuffer::Ptr vgicp_cloud_buffer;
  OdomBuffer::Ptr vgicp_odom_buffer;
};