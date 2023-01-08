/**
 * @file imu_accumulator.cpp
 * @author Julian Gaal, Pascal Buschermöhle
 */
#include "util/imu_accumulator.h"
#include <ros/time.h>

ImuAccumulator::ImuAccumulator(ConcurrentRingBuffer<sensor_msgs::Imu>::Ptr& buffer)
    :   buffer_{buffer},
        first_imu_msg_{true},
        last_imu_timestamp_{}
{}


bool ImuAccumulator::before(const ros::Time& ts_1, const ros::Time& ts_2)
{
  return ts_2.toSec() - ts_1.toSec() >= 0;
}

Eigen::Matrix4d ImuAccumulator::acc_transform(const ros::Time& pcl_timestamp) {

  sensor_msgs::Imu imu_msg;
  Eigen::Matrix4d acc_transform = Eigen::Matrix4d::Identity();

  auto imu_before_pcl = [&](sensor_msgs::Imu& msg){ return before(msg.header.stamp, pcl_timestamp); };

  while(buffer_->pop_nb_if(&imu_msg, imu_before_pcl))
  {
    if(first_imu_msg_)
    {
      last_imu_timestamp_ = imu_msg.header.stamp;
      first_imu_msg_ = false;
      continue;
    }

    apply_transform(acc_transform, imu_msg);
    last_imu_timestamp_ = imu_msg.header.stamp;
  }

  return acc_transform;
}

void ImuAccumulator::apply_transform(Eigen::Matrix4d& acc_transform, const sensor_msgs::Imu& imu_msg)
{
  const Eigen::Vector3d ang_vel(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
  const double acc_time = std::abs(imu_msg.header.stamp.toSec() - last_imu_timestamp_.toSec());
  Eigen::Vector3d orientation = ang_vel * acc_time; //in radiants [rad, rad, rad]

  auto rotation = Eigen::AngleAxisd(orientation.x(), Eigen::Vector3d::UnitX())
                  * Eigen::AngleAxisd(orientation.y(), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(orientation.z(), Eigen::Vector3d::UnitZ());

  Eigen::Matrix3d total = rotation.toRotationMatrix() * acc_transform.block<3, 3>(0, 0);
  acc_transform.block<3, 3>(0, 0) = total; //combine/update transforms
}

