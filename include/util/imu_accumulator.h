#pragma once
#include <Eigen/Dense>
#include <ros/time.h>
#include <mutex>
#include <sensor_msgs/Imu.h>
#include "util/concurrent_ring_buffer.h"

/**
 * @brief Accumulates multiple ImuStamped messages to create an accumulated transform
 * since the last PointCloud Message
 *
 */
class ImuAccumulator
{
public:
  /**
   * @brief Construct a new Imu Accumulator object
   *
   * @param buffer stamped imu buffer shared_ptr
   */
  explicit ImuAccumulator(ConcurrentRingBuffer<sensor_msgs::Imu>::Ptr& buffer);

  /**
   * @brief Destroy the Imu Accumulator object
   *
   */
  ~ImuAccumulator() = default;

  /// delete copy assignment operator
  ImuAccumulator& operator=(const ImuAccumulator& other) = delete;

  /// delete move assignment operator
  ImuAccumulator& operator=(ImuAccumulator&&) = delete;

  /// delete copy constructor
  ImuAccumulator(const ImuAccumulator&) = delete;

  /// delete move constructor
  ImuAccumulator(ImuAccumulator&&) = delete;

  /**
   * @brief Return acculated transform for given pointcloud timestamp
   *
   * @param pcl_timestamp timestamp of received pointcloud
   *
   * @return const Matrix4f the accumulated transform
   */
  Eigen::Matrix4d acc_transform(const ros::Time& pcl_timestamp);

  /// imu buffer to use for accumulation
  ConcurrentRingBuffer<sensor_msgs::Imu>::Ptr & buffer_;
private:
  /**
   * @brief applies single imu transform with duration since last imu message to accumulation matrix
   *
   * @param acc_transform current acculumation matrix
   * @param imu_msg stamped imu message
   */
  void apply_transform(Eigen::Matrix4d& acc_transform, const sensor_msgs::Imu& imu_msg);

  /**
   * @brief Calculates whether or not ts_1 happened before ts_2 (before in the OS timestamp sense)
   *
   * @param ts_1 timestamp 1
   * @param ts_2 timestamp 2
   * @return true if ts_1 happened before ts_2
   * @return false if ts_1 happened after ts_2
   */
  static bool before(const ros::Time& ts_1, const ros::Time& ts_2);

  /// first imu message needs to be catched to calculate diff
  bool first_imu_msg_;

  /// last timestamp of imu to calculate difference to incoming messages
  ros::Time last_imu_timestamp_;
};