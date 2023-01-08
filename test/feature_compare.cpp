#include "floam.h"
#include "util/publish.h"
#include "util/runtime_evaluator.h"

#include <ros/ros.h>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>

namespace fs = std::filesystem;

template <typename T>
inline void file2pcd(typename pcl::PointCloud<T>::Ptr& cloud, const std::string& filename)
{
  if (pcl::io::loadPCDFile<T>(filename, *cloud) == -1)
  {
    throw std::runtime_error("PCL IO Error");
  }
}

template <typename T>
inline void file2pcd(typename pcl::PointCloud<T>::Ptr& cloud, const fs::path& path)
{
  file2pcd<T>(cloud, path.string());
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_compare");
  ros::NodeHandle nh("~");

  fs::path data_loc(DATA_PATH);
  pcl::PointCloud<pcl::PointXYZI>::Ptr orig_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  file2pcd<pcl::PointXYZI>(orig_cloud, data_loc / "00062.pcd");
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::copyPointCloud(*orig_cloud, *cloud);

  ros::Publisher floam_surf_pub = nh.advertise<sensor_msgs::PointCloud2>("/floam_surfs", 100);
  ros::Publisher floam_edge_pub = nh.advertise<sensor_msgs::PointCloud2>("/floam_edges", 100);
  ros::Publisher myfloam_surf_pub = nh.advertise<sensor_msgs::PointCloud2>("/myfloam_surfs", 100);
  ros::Publisher myfloam_edge_pub = nh.advertise<sensor_msgs::PointCloud2>("/myfloam_edges", 100);
  ros::Publisher myfloamlio_surf_pub = nh.advertise<sensor_msgs::PointCloud2>("/myfloamlio_surfs", 100);
  ros::Publisher myfloamlio_edge_pub = nh.advertise<sensor_msgs::PointCloud2>("/myfloamlio_edges", 100);
  ros::Publisher myfloamlio_surf_pub_custom = nh.advertise<sensor_msgs::PointCloud2>("/myfloamlio_surfs_custom", 100);
  ros::Publisher myfloamlio_edge_pub_custom = nh.advertise<sensor_msgs::PointCloud2>("/myfloamlio_edges_custom", 100);
  while (ros::ok())
  {
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "base_link";

#ifdef TIME_MEASUREMENT
    auto &runtime_eval_ = RuntimeEvaluator::get_instance();
    runtime_eval_.start("total");
#endif

    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud(new pcl::PointCloud<pcl::PointXYZI>());
#ifdef TIME_MEASUREMENT
      runtime_eval_.start("floam");
#endif
      original::featureExtraction(cloud, edge_cloud, surf_cloud);
#ifdef TIME_MEASUREMENT
      runtime_eval_.stop("floam");
#endif
      publish_cloud(*surf_cloud, floam_surf_pub, header);
      publish_cloud(*edge_cloud, floam_edge_pub, header);
    }

    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud(new pcl::PointCloud<pcl::PointXYZI>());
#ifdef TIME_MEASUREMENT
      runtime_eval_.start("my_floam");
#endif
      custom::extract_features(cloud, edge_cloud, surf_cloud);
#ifdef TIME_MEASUREMENT
      runtime_eval_.stop("my_floam");
#endif
      publish_cloud(*surf_cloud, myfloam_surf_pub, header);
      publish_cloud(*edge_cloud, myfloam_edge_pub, header);
    }

    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud(new pcl::PointCloud<pcl::PointXYZI>());
#ifdef TIME_MEASUREMENT
      runtime_eval_.start("lio_sam");
#endif
      custom::extract_features_lio_sam(cloud, edge_cloud, surf_cloud);
#ifdef TIME_MEASUREMENT
      runtime_eval_.stop("lio_sam");
#endif
      publish_cloud(*surf_cloud, myfloamlio_surf_pub, header);
      publish_cloud(*edge_cloud, myfloamlio_edge_pub, header);
    }

    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud(new pcl::PointCloud<pcl::PointXYZI>());
#ifdef TIME_MEASUREMENT
      runtime_eval_.start("lio_sam_custom");
#endif
      custom::extract_features_lio_sam_intensity(cloud, edge_cloud, surf_cloud);
#ifdef TIME_MEASUREMENT
      runtime_eval_.stop("lio_sam_custom");
#endif
      publish_cloud(*surf_cloud, myfloamlio_surf_pub_custom, header);
      publish_cloud(*edge_cloud, myfloamlio_edge_pub_custom, header);
    }
#ifdef TIME_MEASUREMENT
    runtime_eval_.stop("total");
    std::cout << runtime_eval_;
#endif
  }
}
