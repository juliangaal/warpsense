#pragma once
#include <warpsense/registration/registration.h>
#include <util/util.h>

inline Eigen::Matrix4f xi_to_transform(const Vector6d& xi, Point& center)
{
  // Formula 3.9 on Page 40 of "Truncated Signed Distance Fields Applied To Robotics"

  // Rotation around an Axis.
  // Direction of axis (l) = Direction of angular_velocity
  // Angle of Rotation (theta) = Length of angular_velocity
  auto angular_velocity = xi.block<3, 1>(0, 0);
  auto theta = angular_velocity.norm();
  auto l = angular_velocity / theta;
  Eigen::Matrix3f L;
  L <<
    0, -l.z(), l.y(),
      l.z(), 0, -l.x(),
      -l.y(), l.x(), 0;

  auto rotation = Eigen::Matrix3f::Identity()
                  + sin(theta) * L + (1 - cos(theta)) * L * L;

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = rotation;

  Eigen::Vector3f old_center;

  old_center = -center.cast<float>();

  Eigen::Vector4f v;
  v << old_center, 1;

  auto shift = (transform * v).block<3, 1>(0, 0);

  transform.block<3, 1>(0, 3) = shift + center.cast<float>() + xi.block<3, 1>(3, 0).cast<float>();

  return transform;
}