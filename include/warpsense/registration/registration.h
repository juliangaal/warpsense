#pragma once

/**
 * @author Malte Hillmann
 * @author Marc Eisoldt
 * @author Julian Gaal
 */

// ros
#include <sensor_msgs/Imu.h>
#include <Eigen/Dense>

// warpsense
#include "warpsense/math/math.h"
#include "map/hdf5_local_map.h"
#include "warpsense/cuda/registration.h"

Eigen::Matrix4f register_cloud(const HDF5LocalMap &map, std::vector<Point> &cloud, const Eigen::Matrix4f &pretransform,
                               int max_iterations, float it_weight_gradient, float epsilon, int map_resolution);

namespace cuda
{
Eigen::Matrix4f
register_cloud(cuda::RegistrationCuda &cuda_reg, std::vector<rmagine::Pointi> &cloud,
               const Eigen::Matrix4f &pretransform, int max_iterations, float it_weight_gradient,
               float epsilon);
}
