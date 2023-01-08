#pragma once
#include <vector>
#include <warpsense/math/math.h>
#include <warpsense/types.h>

#define DEBUG(x) printf("%s:%d : ", __FILE__, __LINE__); printf((x));

inline Eigen::Matrix4i to_int_mat(const Eigen::Matrix4f& mat)
{
  return (mat * MATRIX_RESOLUTION).cast<int>();
}

inline Point transform_point(const Point& input, const Eigen::Matrix4i& mat)
{
  Eigen::Vector4i v;
  v << input, 1;
  return (mat * v).block<3, 1>(0, 0) / MATRIX_RESOLUTION;
}

template <typename T>
inline rmagine::Vector3<T> transform_point(const rmagine::Vector3<T>& input, const Eigen::Matrix<T, 4, 4>& mat)
{
  rmagine::Vector3<T> out;
  out.x = mat(0, 0) * input.x + mat(0, 1) * input.y + mat(0, 2) * input.z;
  out.y = mat(1, 0) * input.x + mat(1, 1) * input.y + mat(1, 2) * input.z;
  out.z = mat(2, 0) * input.x + mat(2, 1) * input.y + mat(2, 2) * input.z;
  out.x += mat(0, 3);
  out.y += mat(1, 3);
  out.z += mat(2, 3);
  out.x /= MATRIX_RESOLUTION;
  out.y /= MATRIX_RESOLUTION;
  out.z /= MATRIX_RESOLUTION;
  return out;
}

template <typename T>
inline rmagine::Vector3<T> transform_point(const rmagine::Vector3<T>& input, const rmagine::Matrix4x4<T>& mat)
{
  rmagine::Vector3<T> out;
  out.x = mat.at(0, 0) * input.x + mat.at(0, 1) * input.y + mat.at(0, 2) * input.z;
  out.y = mat.at(1, 0) * input.x + mat.at(1, 1) * input.y + mat.at(1, 2) * input.z;
  out.z = mat.at(2, 0) * input.x + mat.at(2, 1) * input.y + mat.at(2, 2) * input.z;
  out.x += mat.at(0, 3);
  out.y += mat.at(1, 3);
  out.z += mat.at(2, 3);
  out.x /= MATRIX_RESOLUTION;
  out.y /= MATRIX_RESOLUTION;
  out.z /= MATRIX_RESOLUTION;
  return out;
}

inline Eigen::Vector3i to_map(const Eigen::Matrix4f& pose, int map_resolution)
{
  const Eigen::Vector3f& p = pose.block<3, 1>(0, 3);
  return { (int)std::floor(p.x() / (float)map_resolution), (int)std::floor(p.y() / (float)map_resolution), (int)std::floor(p.z() / (float)map_resolution) };
}

inline Eigen::Matrix4i to_int_mat(const Eigen::Isometry3d& pos)
{
  Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
  mat.block<3, 3>(0, 0) = Eigen::Quaterniond(pos.rotation()).toRotationMatrix().cast<float>();
  mat.block<3, 1>(0, 3) = (pos.translation() * 1000).cast<float>();
  return to_int_mat(mat);
}

inline void transform_point_cloud(std::vector<Point>& in_cloud, const Eigen::Matrix4f& mat)
{
#pragma omp parallel for schedule(static)
  for (auto index = 0u; index < in_cloud.size(); ++index)
  {
    auto& point = in_cloud[index];
    Eigen::Vector3f tmp = (mat.block<3, 3>(0, 0) * point.cast<float>() + mat.block<3, 1>(0, 3));
    tmp[0] < 0 ? tmp[0] -= 0.5 : tmp[0] += 0.5;
    tmp[1] < 0 ? tmp[1] -= 0.5 : tmp[1] += 0.5;
    tmp[2] < 0 ? tmp[2] -= 0.5 : tmp[2] += 0.5;

    point = tmp.cast<int>();
  }
}

inline void transform_point_cloud(std::vector<rmagine::Pointi>& in_cloud, const rmagine::Matrix4x4f& mat)
{
#pragma omp parallel for schedule(static)
  for (auto index = 0u; index < in_cloud.size(); ++index)
  {
    auto& input = in_cloud[index];
    auto inputf = rmagine::Vector3f((float)input.x, (float)input.y, (float)input.z);
    rmagine::Vector3f tmp;
    tmp.x = mat.at(0, 0) * inputf.x + mat.at(0, 1) * inputf.y + mat.at(0, 2) * inputf.z;
    tmp.y = mat.at(1, 0) * inputf.x + mat.at(1, 1) * inputf.y + mat.at(1, 2) * inputf.z;
    tmp.z = mat.at(2, 0) * inputf.x + mat.at(2, 1) * inputf.y + mat.at(2, 2) * inputf.z;
    tmp.x += mat(0, 3);
    tmp.y += mat(1, 3);
    tmp.z += mat(2, 3);
    tmp.x < 0 ? tmp.x -= 0.5 : tmp.x += 0.5;
    tmp.y < 0 ? tmp.y -= 0.5 : tmp.y += 0.5;
    tmp.z < 0 ? tmp.z -= 0.5 : tmp.z += 0.5;

    input = rmagine::Pointi((int)tmp.x, (int)tmp.y, (int)tmp.z);
  }
}
