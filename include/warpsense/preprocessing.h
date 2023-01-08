#pragma once
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <warpsense/math/math.h>
#include <util/util.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/point_cloud.h>
#include <warpsense/types.h>
#include <unordered_set>


// inline void reduction_filter_voxel_center(const pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<Point>& scan_points, const Eigen::Isometry3d& current_pose)
// {
//   std::unordered_set<Point> point_set;
//   point_set.reserve(30'000);
//   Eigen::Matrix4i next_transform = to_int_mat(current_pose);

//   for (const auto& point : cloud)
//   {
//     if ((point.x == 0 && point.y == 0 && point.z == 0) || 20 < std::abs(point.x) || 20 < std::abs(point.y) || 5 < std::abs(point.z))
//     {
//       continue;
//     }

//     Point voxel_center(
//     std::floor((float)point.x * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.y * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.z * 1000.f / map_resolution) * map_resolution + map_resolution / 2
//     );

//     point_set.insert(transform_point(voxel_center, next_transform));
//   }

//   scan_points.resize(point_set.size());
//   std::copy(point_set.begin(), point_set.end(), scan_points.begin());
// }

// inline void reduction_filter_voxel_center(const pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<Point>& scan_points)
// {
//   std::unordered_set<Point> point_set;
//   point_set.reserve(30'000);

//   for (const auto& point : cloud)
//   {
//     if ((point.x < 0.3 && point.y < 0.3 && point.z < 0.3))
//     {
//       continue;
//     }

//     Point voxel_center(
//         std::floor((float)point.x * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.y * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.z * 1000.f / map_resolution) * map_resolution + map_resolution / 2
//     );

//     point_set.insert(voxel_center);
//   }

//   scan_points.resize(point_set.size());
//   std::copy(point_set.begin(), point_set.end(), scan_points.begin());
// }

template <typename T>
inline void reduction_filter_voxel_center(const pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<rmagine::Vector3<T>>& scan_points, int map_resolution, const Eigen::Matrix4i& pretransform)
{
  std::unordered_set<rmagine::Vector3<T>> point_set;
  point_set.reserve(30'000);

  for (const auto& point : cloud)
  {
    if ((point.x < 0.3 && point.y < 0.3 && point.z < 0.3))
    {
      continue;
    }

    auto p = transform_point(rmagine::Pointi(point.x, point.y, point.z), pretransform);
    rmagine::Vector3<T> voxel_center
    {
      std::floor((float)point.x * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
      std::floor((float)point.y * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
      std::floor((float)point.z * 1000.f / map_resolution) * map_resolution + map_resolution / 2
    };

    point_set.insert(voxel_center);
  }

  scan_points.resize(point_set.size());
  std::copy(point_set.begin(), point_set.end(), scan_points.begin());
}

// template <typename T>
// inline void reduction_filter_voxel_center(const pcl::PointCloud<pcl::PointXYZ>& cloud, std::vector<rmagine::Vector3<T>>& scan_points, const Eigen::Matrix4i& transform)
// {
//   std::unordered_set<rmagine::Vector3<T>> point_set;
//   point_set.reserve(30'000);

//   for (const auto& point : cloud)
//   {
//     if ((point.x < 0.3 && point.y < 0.3 && point.z < 0.3))
//     {
//       continue;
//     }

//     rmagine::Vector3<T> voxel_center
//         {
//             std::floor((float)point.x * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//             std::floor((float)point.y * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//             std::floor((float)point.z * 1000.f / map_resolution) * map_resolution + map_resolution / 2
//         };

//     point_set.insert(transform_point(voxel_center, transform));
//   }

//   scan_points.resize(point_set.size());
//   std::copy(point_set.begin(), point_set.end(), scan_points.begin());
// }

// // TODO map boundaries configurable
// inline void reduction_filter_voxel_center(const std::vector<Eigen::Vector3f>& cloud, std::vector<Point>& scan_points)
// {
//   std::unordered_set<Point> point_set;
//   point_set.reserve(cloud.size());

//   for (const auto& point : cloud)
//   {
//     if ((point.x() == 0 && point.y() == 0 && point.z() == 0))
//     {
//       continue;
//     }

//     Point voxel_center(
//         std::floor((float)point.x() * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.y() * 1000.f / map_resolution) * map_resolution + map_resolution / 2,
//         std::floor((float)point.z() * 1000.f / map_resolution) * map_resolution + map_resolution / 2
//     );

//     point_set.insert(voxel_center);
//   }

//   scan_points.resize(point_set.size());
//   std::copy(point_set.begin(), point_set.end(), scan_points.begin());
// }

// inline void reduction_filter_average(pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<Point>& scan_points, const Eigen::Isometry3d& current_pose)
// {
//   std::unordered_map<Point, std::pair<Vector3i, int>> point_map;
//   point_map.reserve(cloud.size());
//   std::pair<Vector3i, int> default_value = std::make_pair(Point::Zero(), 0);
//   Eigen::Matrix4i next_transform = to_int_mat(current_pose);

//   for (const auto& point : cloud)
//   {
//     if ((point.x == 0 && point.y == 0 && point.z == 0) || 20 < std::abs(point.x) || 20 < std::abs(point.y) || 5 < std::abs(point.z))
//     {
//       continue;
//     }

//     Point voxel(
//         std::floor((float)point.x * 1000.f / map_resolution),
//         std::floor((float)point.y * 1000.f / map_resolution),
//         std::floor((float)point.z * 1000.f / map_resolution)
//     );

//     auto& avg_point = point_map.try_emplace(voxel, default_value).first->second;
//     avg_point.first += Eigen::Vector3f(point.x * 1000.f, point.y * 1000.f, point.z * 1000.f).cast<int>();
//     avg_point.second++;
//   }

//   scan_points.resize(point_map.size());
//   int counter = 0;
//   for (auto& avg_point : point_map)
//   {
//     scan_points[counter++] = transform_point((avg_point.second.first / avg_point.second.second).cast<int>(), next_transform);
//   }
// }

// inline void reduction_filter_average(pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<Point>& scan_points)
// {
//   std::unordered_map<Point, std::pair<Vector3i, int>> point_map;
//   point_map.reserve(cloud.size());
//   std::pair<Vector3i, int> default_value = std::make_pair(Point::Zero(), 0);

//   for (const auto& point : cloud)
//   {
//     if ((point.x == 0 && point.y == 0 && point.z == 0) || 20 < std::abs(point.x) || 20 < std::abs(point.y) || 5 < std::abs(point.z))
//     {
//       continue;
//     }

//     Point voxel(
//         std::floor((float)point.x * 1000.f / map_resolution),
//         std::floor((float)point.y * 1000.f / map_resolution),
//         std::floor((float)point.z * 1000.f / map_resolution)
//     );

//     auto& avg_point = point_map.try_emplace(voxel, default_value).first->second;
//     avg_point.first += Eigen::Vector3f(point.x * 1000.f, point.y * 1000.f, point.z * 1000.f).cast<int>();
//     avg_point.second++;
//   }

//   scan_points.resize(point_map.size());
//   int counter = 0;
//   for (auto& avg_point : point_map)
//   {
//     scan_points[counter++] = (avg_point.second.first / avg_point.second.second).cast<int>();
//   }
// }

// template <typename T>
// inline void reduction_filter_average(pcl::PointCloud<pcl::PointXYZI>& cloud, std::vector<rmagine::Vector3<T>>& scan_points)
// {
//   std::unordered_map<rmagine::Vector3<T>, std::pair<rmagine::Vector3<T>, int>> point_map;
//   point_map.reserve(cloud.size());
//   std::pair<rmagine::Vector3<T>, int> default_value = std::make_pair(rmagine::Vector3<T>{}, 0);

//   for (const auto& point : cloud)
//   {
//     if ((point.x == 0 && point.y == 0 && point.z == 0) || 20 < std::abs(point.x) || 20 < std::abs(point.y) || 5 < std::abs(point.z))
//     {
//       continue;
//     }

//     rmagine::Vector3<T> voxel{
//     std::floor((float)point.x * 1000.f / map_resolution),
//     std::floor((float)point.y * 1000.f / map_resolution),
//     std::floor((float)point.z * 1000.f / map_resolution)
//     };

//     auto& avg_point = point_map.try_emplace(voxel, default_value).first->second;
//     avg_point.first += rmagine::Vector3<T>(point.x * 1000.f, point.y * 1000.f, point.z * 1000.f);
//     avg_point.second++;
//   }

//   scan_points.resize(point_map.size());
//   int counter = 0;
//   for (auto& avg_point : point_map)
//   {
//     scan_points[counter++] = (avg_point.second.first / avg_point.second.second);
//   }
// }


