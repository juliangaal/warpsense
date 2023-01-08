#pragma once

/**
 * @author Malte Hillmann (mhillmann)
 * @author Marc Eisoldt (meisoldt)
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <warpsense/types.h>
#include <warpsense/math/math.h>
#include <map/hdf5_local_map.h>
#include <unordered_map>

// using Map = std::unordered_map<Point, std::pair<float, float>>;
using FastMap = std::unordered_map<Point, TSDFEntry>;

Point to_point(const Vector3f& vec);

void update_tsdf(const std::vector<Point> &scan_points, const Eigen::Vector3i &scanner_pos, const Eigen::Vector3i &up,
                 HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution);

void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const Eigen::Vector3i &scanner_pos, const Eigen::Vector3i &up,
                 HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution);

void update_tsdf(const std::vector<Point> &scan_points, pcl::PointCloud<pcl::Normal>::Ptr &normals,
                 const Eigen::Vector3i &scanner_pos, HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution);

void update_tsdf(const std::vector<rmagine::Pointi> &scan_points, const std::vector<rmagine::Pointf> &normals,
                 const rmagine::Pointi &scanner_pos, HDF5LocalMap &buffer, int tau, int max_weight, int map_resolution);
