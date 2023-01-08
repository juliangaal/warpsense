// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>


// warpsense
#include "params/params.h"
#include "warpsense/types.h"
#include "util/util.h"
#include "warpsense/math/math.h"
#include "warpsense/cuda/update_tsdf.h"
#include "map/hdf5_local_map.h"
#include "warpsense/visualization/map.h"
#include "cpu/update_tsdf.h"

namespace rm = rmagine;
namespace fs = boost::filesystem;



int main(int argc, char **argv)
{
  ros::init(argc, argv, "registration");
  ros::NodeHandle nh("~");
  Params params(nh);
  auto pub = nh.advertise<visualization_msgs::Marker>("/tsdf", 0);
  auto normal_pub = nh.advertise<visualization_msgs::Marker>("/tsdf_normal", 0);
  auto normal_rmagine_pub = nh.advertise<visualization_msgs::Marker>("/tsdf_rmagine_normal", 0);
  auto cuda_pub = nh.advertise<visualization_msgs::Marker>("/cuda_tsdf", 0);
  auto cuda_normal_pub = nh.advertise<visualization_msgs::Marker>("/cuda_tsdf_normal", 0);
  auto pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 0);

  // setup point data
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  fs::path filename = fs::path(DATA_PATH) / "frame_500.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZI> (filename.string(), *cloud) == -1)
  {
    PCL_ERROR("Couldn't read test pcd\n");
    return (-1);
  }

  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize((float)params.map.resolution / 1000.f, (float)params.map.resolution / 1000.f, (float)params.map.resolution / 1000.f);
  sor.filter(*cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr demeaned_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Vector4d centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  pcl::demeanPointCloud(*cloud, centroid, *demeaned_cloud);

  std::vector<Point> points_original;//(cloud->size());
  std::vector<rm::Pointi> points_rm_original;//(cloud->size());

  for (const auto& cp: *cloud)
  {
    points_original.emplace_back(Point(cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f));
    points_rm_original.emplace_back(rmagine::Pointi(cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f));
  }

  pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
  ne.setInputCloud(demeaned_cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
  ne.setSearchMethod(tree);

  std::cout << "built tree\n";

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.setKSearch(50);
  ne.compute(*normals);

  std::vector<rmagine::Pointf> normals_rm;
  for (const auto& n: *normals)
  {
    normals_rm.emplace_back(n.normal_x, n.normal_y, n.normal_z);
  }

  // setup local map
  std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
  std::shared_ptr<HDF5LocalMap> local_map;
  global_map_ptr_.reset(new HDF5GlobalMap(params.map));
  local_map.reset(new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), global_map_ptr_));

  // Shift
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Vector3i pos(0, 0, 0);
  local_map->shift(pos);

  auto cuda_local_map = std::make_shared<HDF5LocalMap>(*local_map);
  auto local_map_normal = std::make_shared<HDF5LocalMap>(*local_map);
  auto local_map_rmagine_normal = std::make_shared<HDF5LocalMap>(*local_map);

  Eigen::Matrix4i rot = Eigen::Matrix4i::Identity();
  rot.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rot);

  // create TSDF Volume (CPU)
  update_tsdf(points_original, pos, up, *local_map, params.map.tau, params.map.max_weight, params.map.resolution);
  update_tsdf(points_original, normals, pos, *local_map_normal, params.map.tau, params.map.max_weight, params.map.resolution);
  update_tsdf(points_rm_original, normals_rm, rmagine::Pointi(0, 0, 0), *local_map_rmagine_normal, params.map.tau, params.map.max_weight, params.map.resolution);

  // create TSDF Volume (GPU)
  rm::Pointi pos_rm = *reinterpret_cast<const rm::Pointi*>(&pos);
  rm::Pointi up_rm = *reinterpret_cast<const rm::Pointi*>(&up);
  cuda::DeviceMap cuda_map(cuda_local_map);
  cuda::TSDFCuda tsdf(cuda_map, params.map.tau, params.map.max_weight, params.map.resolution);
  tsdf.update_tsdf(cuda_map, points_rm_original, pos_rm, up_rm);

  std_msgs::Header header;
  header.frame_id = "map";
  header.stamp = ros::Time::now();

  sensor_msgs::PointCloud2 ros_pcl;
  pcl::toROSMsg(*cloud, ros_pcl);
  ros_pcl.header = header;

  ros::Rate r(1.0);
  while (ros::ok())
  {
    r.sleep();
    pcl_pub.publish(ros_pcl);
    publish_local_map(pub, *local_map, params.map.tau, params.map.resolution);
    publish_local_map(normal_pub, *local_map_normal, params.map.tau, params.map.resolution);
    publish_local_map(normal_rmagine_pub, *local_map_rmagine_normal, params.map.tau, params.map.resolution);
    publish_local_map(cuda_pub, *cuda_local_map, params.map.tau, params.map.resolution);
  }
}
