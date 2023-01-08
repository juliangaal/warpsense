#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "ikd_Tree.h"
#include "nanoflann.hpp"
#include "util/nanoflann_pcl.h"

namespace fs = boost::filesystem;

using KDTreePointType = pcl::PointXYZI;
using KDTreePointVector = KD_TREE<KDTreePointType>::PointVector;

std::string format(const KDTreePointType& data)
{
  return "(" + std::to_string(data.x) + " / " + std::to_string(data.y) + " / "  + std::to_string(data.z) + ")";
}

template <typename T, typename S = std::allocator<T>>
void print(const std::vector<T, S>& vec)
{
  std::transform(vec.begin(), vec.end(), std::ostream_iterator<std::string>(std::cout, ", "), format);
}

template <typename T, typename S = std::allocator<T>>
void println(const std::vector<T, S>& vec)
{
  print(vec);
  std::cout << "\n";
}

int main(void)
{
  // setup point data
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  fs::path filename = fs::path(DATA_PATH) / "parkplatz.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZI> (filename.string(), *cloud) == -1)
  {
    PCL_ERROR("Couldn't read test pcd\n");
    return (-1);
  }

  printf("Read pointcloud with %lu points", cloud->size());

  pcl::PointXYZI query;
  query.x = 3;
  query.y = 4;
  query.z = 5;

  int k = 5;

  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.1, 0.1, 0.1);
  sor.filter(*cloud);

  std::cout << "flann:\n";
  {
    auto flann_kdtree = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    flann_kdtree->setInputCloud(cloud);

    std::vector<int> pointSearchInd;
    pointSearchInd.reserve(k);
    std::vector<float> pointSearchSqDis;
    pointSearchSqDis.reserve(k);
    flann_kdtree->nearestKSearch(query, k, pointSearchInd, pointSearchSqDis);

    std::vector<KDTreePointType> flann_results;
    for (const auto& idx: pointSearchInd)
    {
      flann_results.push_back(cloud->points[idx]);
    }
    println(flann_results);
  }

  std::cout << "ikdtree:\n";
  {
    //KD_TREE<PointType> ikd_Tree(0.5, 0.6, 0.2); WHY THE FUCK DOES THIS SEGFAULT, BUT DOESN'T WITH POINTER
    KD_TREE<KDTreePointType>::Ptr kdtree(new KD_TREE<KDTreePointType>(0.3, 0.6, 0.2));
    kdtree->Build(cloud->points);

    KDTreePointVector search_result(5);
    std::vector<float> distances;
    kdtree->Nearest_Search(query, k, search_result, distances);
    println(search_result);
  }

  std::cout << "nanoflann:\n";
  {
    nanoflann::KdTreeFLANN<pcl::PointXYZI> nanoflann_kdtree;
    nanoflann_kdtree.setInputCloud(cloud);

    std::vector<int> pointSearchInd(k, 0);
    std::vector<float> pointSearchSqDis(k, 0);

    nanoflann_kdtree.nearestKSearch(query, k, pointSearchInd, pointSearchSqDis);

    std::vector<pcl::PointXYZI> nanoflann_results;
    for (const auto& idx: pointSearchInd)
    {
      nanoflann_results.push_back(cloud->points[idx]);
    }

    println(nanoflann_results);
  }
}
