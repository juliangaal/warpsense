// Original Copyright:
// Original Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
//
// Modification Copyright: Julian Gaal

#include <execution>
#include <omp.h>
#include "featsense/lidar_processing.h"

LidarProcessing::LidarProcessing(ros::NodeHandle &nh, const Params &params,
                                 Buffers &buffers)
    : params_(params), cloud_buf_(buffers.cloud_prep_buffer), vgicp_cloud_buf_(buffers.vgicp_cloud_buffer),
      surf_buf_(buffers.surf_buffer), edge_buf_(buffers.edge_buffer), header_buf_(buffers.header_buffer),
      vgicp_header_buf_(buffers.vgicp_header_buffer),
      features_(),
      ranges_(),
      neighbor_picked_(),
      cloud_filtered_pub_(nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_filtered", 100)),
      edge_point_pub_(nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_edge", 100)),
      surf_points_pub_(nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100))
{
  auto width = params_.lidar.hresolution;
  auto height = params_.lidar.channels;

  features_.resize(width * height);
  ranges_.resize(width * height);
  neighbor_picked_.resize(width * height);
}

void LidarProcessing::thread_run()
{
  double total_time = 0;
  int total_frame = 0;

  const int &pcl_height = params_.lidar.channels;
  const int &pcl_width = params_.lidar.hresolution;

  while (ros::ok() && running)
  {
    if (cloud_buf_ && !cloud_buf_->empty())
    {
      //read data
      pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_in(new pcl::PointCloud<pcl::PointXYZI>());
      sensor_msgs::PointCloud2ConstPtr cloud;
      cloud_buf_->pop_nb(&cloud);
      mypcl::fromROSMsg(*cloud, *pointcloud_in);
      ros::Time pointcloud_time = cloud->header.stamp;

      if (pcl_height * pcl_width != (int) pointcloud_in->size())
      {
        ROS_ERROR_STREAM("Check horizontal and vertical resolution. Height (" \
 << pcl_height << ") * Width (" << pcl_width << ") must match pointclouds size");
        continue;
      }

      pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_edge(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_surf(new pcl::PointCloud<pcl::PointXYZI>());
      pointcloud_edge->points.reserve(pointcloud_in->size());
      pointcloud_surf->points.reserve(pointcloud_in->size());

      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();
      process_lidar(pointcloud_in, pointcloud_edge, pointcloud_surf);
      end = std::chrono::system_clock::now();
      std::chrono::duration<float> elapsed_seconds = end - start;
      total_frame++;
      float time_temp = elapsed_seconds.count() * 1000;
      total_time += time_temp;
      ROS_INFO("average laser processing time %f ms", total_time / total_frame);

      // push into respective buffers
      surf_buf_->push_nb(pointcloud_surf);
      edge_buf_->push_nb(pointcloud_edge);

      // header to visualization
      header_buf_->push_nb(cloud->header);

      // header to vgcip
      vgicp_header_buf_->push_nb(cloud->header);

      // clear data for next iteration
      features_.clear();
      ranges_.clear();
      neighbor_picked_.clear();

      sensor_msgs::PointCloud2 laserCloudFilteredMsg;
      pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
      *pointcloud_filtered += *pointcloud_edge;
      *pointcloud_filtered += *pointcloud_surf;

      // **raw** pointcloud to vgicp
      vgicp_cloud_buf_->push_nb(pointcloud_in);

      pcl::toROSMsg(*pointcloud_filtered, laserCloudFilteredMsg);
      laserCloudFilteredMsg.header.stamp = pointcloud_time;
      laserCloudFilteredMsg.header.frame_id = "base_link";
      cloud_filtered_pub_.publish(laserCloudFilteredMsg);

      sensor_msgs::PointCloud2 edgePointsMsg;
      pcl::toROSMsg(*pointcloud_edge, edgePointsMsg);
      edgePointsMsg.header.stamp = pointcloud_time;
      edgePointsMsg.header.frame_id = "base_link";
      edge_point_pub_.publish(edgePointsMsg);

      sensor_msgs::PointCloud2 surfPointsMsg;
      pcl::toROSMsg(*pointcloud_surf, surfPointsMsg);
      surfPointsMsg.header.stamp = pointcloud_time;
      surfPointsMsg.header.frame_id = "base_link";
      surf_points_pub_.publish(surfPointsMsg);
    }

    //sleep 2 ms every time
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

float range(const pcl::PointXYZI &p)
{
  return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

void LidarProcessing::process_lidar(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_cloud,
                                             pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_cloud)
{
  calculate_curvature(cloud);

  mark_occluded_points();

  extract_features(cloud, edge_cloud, surf_cloud);
}

void LidarProcessing::mark_occluded_points()
{
  auto width = params_.lidar.hresolution;
  auto height = params_.lidar.channels;
  
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      // mark occluded points
      size_t i = u * width + v;
      float distance = ranges_[i];
      float next_distance = ranges_[i + 1];
      float prev_distance = ranges_[i - 1];

      if (distance < params_.floam.min_distance || distance > params_.floam.max_distance)
      {
        neighbor_picked_[i] = true;
      }

      if (distance - next_distance > 0.3)
      {
        neighbor_picked_[i - 5] = true;
        neighbor_picked_[i - 4] = true;
        neighbor_picked_[i - 3] = true;
        neighbor_picked_[i - 2] = true;
        neighbor_picked_[i - 1] = true;
        neighbor_picked_[i] = true;

      }

      float dist_temp = next_distance - distance;
      if (dist_temp > 0.3)
      {
        neighbor_picked_[i + 1] = true;
        neighbor_picked_[i + 2] = true;
        neighbor_picked_[i + 3] = true;
        neighbor_picked_[i + 4] = true;
        neighbor_picked_[i + 5] = true;
        neighbor_picked_[i + 6] = true;
      }

      // parallel beam
      float diff1 = std::abs(prev_distance - distance);
      float diff2 = std::abs(dist_temp);

      if (diff1 > 0.02 * distance && diff2 > 0.02 * distance)
      {
        neighbor_picked_[i] = true;
      }
    }
  }
}

void LidarProcessing::calculate_curvature(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
{
  auto width = params_.lidar.hresolution;
  auto height = params_.lidar.channels;
  
  // calculate curvature
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; ++v)
    {
      size_t i = u * width + v;
      float diffX = cloud->points[i - 5].x + cloud->points[i - 4].x + cloud->points[i - 3].x + cloud->points[i - 2].x +
                    cloud->points[i - 1].x - 10 * cloud->points[i].x + cloud->points[i + 1].x + cloud->points[i + 2].x +
                    cloud->points[i + 3].x + cloud->points[i + 4].x + cloud->points[i + 5].x;
      float diffY = cloud->points[i - 5].y + cloud->points[i - 4].y + cloud->points[i - 3].y + cloud->points[i - 2].y +
                    cloud->points[i - 1].y - 10 * cloud->points[i].y + cloud->points[i + 1].y + cloud->points[i + 2].y +
                    cloud->points[i + 3].y + cloud->points[i + 4].y + cloud->points[i + 5].y;
      float diffZ = cloud->points[i - 5].z + cloud->points[i - 4].z + cloud->points[i - 3].z + cloud->points[i - 2].z +
                    cloud->points[i - 1].z - 10 * cloud->points[i].z + cloud->points[i + 1].z + cloud->points[i + 2].z +
                    cloud->points[i + 3].z + cloud->points[i + 4].z + cloud->points[i + 5].z;

      ranges_[i] = range(cloud->points[i]);
      features_[i] = Feature(i, diffX * diffX + diffY * diffY + diffZ * diffZ);
      neighbor_picked_[i] = false;
    }
  }
}

void LidarProcessing::extract_features(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &edge_cloud,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr &surf_cloud)
{
  auto width = params_.lidar.hresolution;
  auto height = params_.lidar.channels;

  float edge_threshold = params_.floam.edge_threshold;
  float surf_threshold = params_.floam.surf_threshold;
  
  for (int u = 0; u < height; ++u)
  {
    for (int v = 5; v < width - 6; v += (width / 6))
    {
      auto sp = u * width + v;
      auto ep = sp + width / 6;
      if (ep >= (u + 1) * width)
      {
        ep = (u + 1) * width - 6;
      }

      std::sort(features_.begin() + sp, features_.begin() + ep, [&](const auto &f, const auto &f2)
      {
        return f.curvature < f2.curvature;
      });

      int max_edge_features = 20;
      for (int k = ep; k >= sp; --k)
      {
        size_t idx = features_[k].idx;
        float curvature = features_[k].curvature;

        if (curvature >= edge_threshold && !neighbor_picked_[idx] && max_edge_features-- != 0)
        {
          edge_cloud->push_back(cloud->points[idx]);

          for (int l = 0; l < 5; l++)
          {
            neighbor_picked_[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked_[idx + l] = true;
          }
        }
      }

      int max_plane_features = 20;
      for (int k = sp; k < ep; ++k)
      {
        size_t idx = features_[k].idx;
        float curvature = features_[k].curvature;
        if (curvature <= surf_threshold && !neighbor_picked_[idx] && max_plane_features-- != 0)
        {
          for (int l = 0; l <= 5; l++)
          {
            neighbor_picked_[idx + l] = true;
          }
          for (int l = -1; l >= -5; l--)
          {
            neighbor_picked_[idx + l] = true;
          }

          surf_cloud->push_back(cloud->points[idx]);
        }
      }
    }
  }
}
