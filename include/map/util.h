#pragma once

#include <Eigen/Dense>

static inline Eigen::Vector3i floor_divide(const Eigen::Vector3i& a, int b)
{
  return Eigen::Vector3i(
      std::floor(static_cast<float>(a[0]) / b),
      std::floor(static_cast<float>(a[1]) / b),
      std::floor(static_cast<float>(a[2]) / b)
  );
}

