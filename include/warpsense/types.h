#pragma once
#include <Eigen/Dense>
#include "consts.h"
#include <vector>
#include "warpsense/math/math.h"

using Point = Eigen::Vector3i;
using Pointl = Eigen::Matrix<long, 3, 1>;
using Vector3f = Eigen::Vector3f;
using Vector3i = Eigen::Vector3i;
using Matrix6l = Eigen::Matrix<long, 6, 6>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Point6l = Eigen::Matrix<long, 6, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

#define ID_VAL(x) printf("%s=%d\n", #x, x)

template<typename VEC_T>
using ScanPoints_t = std::vector<std::vector<VEC_T>>;

namespace std
{
template<>
struct hash<Point>
{
  std::size_t operator()(Point const &p) const noexcept
  {
    long long v = ((long long) p.x() << 32) ^ ((long long) p.y() << 16) ^ (long long) p.z();
    return std::hash<long long>()(v);
  }
};

template<>
struct hash<rmagine::Pointi>
{
  std::size_t operator()(rmagine::Pointi const &p) const noexcept
  {
    long long v = ((long long) p.x << 32) ^ ((long long) p.y << 16) ^ (long long) p.z;
    return std::hash<long long>()(v);
  }
};
}


