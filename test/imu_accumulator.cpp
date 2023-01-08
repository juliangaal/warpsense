#include <gtest/gtest.h>
#include <math.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <util/imu_accumulator.h>
#include <iostream>


static void transform_point_cloud(std::vector<Eigen::Vector3f>& in_cloud, const Eigen::Matrix4f& mat)
{
#pragma omp parallel for schedule(static)
  for (auto i = 0u; i < in_cloud.size(); ++i)
  {
    Eigen::Vector4f v;
    v << in_cloud[i], 1;
    in_cloud[i] = (mat * v).block<3,1>(0, 0);

  }
}

TEST(warpsense, accumulator_pos)
{
  auto imu_buffer = std::make_shared<ConcurrentRingBuffer<sensor_msgs::Imu>>(10);

  auto stamp = ros::Time::now();
  auto diff = ros::Duration(250 * 0.001);

  sensor_msgs::Imu imu;
  imu.header.stamp = stamp;
  imu.angular_velocity.z = M_PI;

  for (const auto& i: {1, 2, 3, 4, 5, 6, 7})
  {
    auto imu_msg = imu;
    imu_msg.header.stamp = stamp + ros::Duration(250 * 0.001 * i);
    imu_buffer->push(imu_msg);
  }

  ImuAccumulator imu_acc{imu_buffer};

  Eigen::Matrix4d acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 5));

  // target rotation in deg
  double target = 180;
  // target rotation r in radians
  double r = target * (M_PI / 180);
  Eigen::Matrix4d rotation_mat;
  rotation_mat <<  cos(r), -sin(r), 0, 0,
  sin(r),  cos(r), 0, 0,
  0,       0,      1, 0,
  0,       0,      0, 1;

  EXPECT_NEAR(acc(0,0), rotation_mat(0,0), 0.001);
  EXPECT_NEAR(acc(0,1), rotation_mat(0,1), 0.001);
  EXPECT_NEAR(acc(0,2), rotation_mat(0,2), 0.001);
  EXPECT_NEAR(acc(1,0), rotation_mat(1,0), 0.001);
  EXPECT_NEAR(acc(1,1), rotation_mat(1,1), 0.001);
  EXPECT_NEAR(acc(1,2), rotation_mat(1,2), 0.001);
  EXPECT_NEAR(acc(2,0), rotation_mat(2,0), 0.001);
  EXPECT_NEAR(acc(2,1), rotation_mat(2,1), 0.001);
  EXPECT_NEAR(acc(2,2), rotation_mat(2,2), 0.001);

  EXPECT_TRUE(imu_buffer->size() ==  2);

  acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 7));

  // 2 imu messages left in buffer -> -90 grad rotation around z axis
  // target rotation in deg
  target = 90;
  // target rotation r in radians
  r = target * (M_PI / 180);
  rotation_mat.setIdentity();
  rotation_mat <<  cos(r), -sin(r), 0, 0,
  sin(r),  cos(r), 0, 0,
  0,       0,      1, 0,
  0,       0,      0, 1;

  EXPECT_NEAR(acc(0,0), rotation_mat(0,0), 0.001);
  EXPECT_NEAR(acc(0,1), rotation_mat(0,1), 0.001);
  EXPECT_NEAR(acc(0,2), rotation_mat(0,2), 0.001);
  EXPECT_NEAR(acc(1,0), rotation_mat(1,0), 0.001);
  EXPECT_NEAR(acc(1,1), rotation_mat(1,1), 0.001);
  EXPECT_NEAR(acc(1,2), rotation_mat(1,2), 0.001);
  EXPECT_NEAR(acc(2,0), rotation_mat(2,0), 0.001);
  EXPECT_NEAR(acc(2,1), rotation_mat(2,1), 0.001);
  EXPECT_NEAR(acc(2,2), rotation_mat(2,2), 0.001);

  EXPECT_TRUE(imu_buffer->empty());
}

TEST(warpsense, accumulator_neg)
{
  auto imu_buffer = std::make_shared<ConcurrentRingBuffer<sensor_msgs::Imu>>(10);

  auto stamp = ros::Time::now();
  auto diff = ros::Duration(250 * 0.001);

  sensor_msgs::Imu imu;
  imu.header.stamp = stamp;
  imu.angular_velocity.z = -M_PI;

  for (const auto& i: {1, 2, 3, 4, 5, 6, 7})
  {
    auto imu_msg = imu;
    imu_msg.header.stamp = stamp + ros::Duration(250 * 0.001 * i);
    imu_buffer->push(imu_msg);
  }

  ImuAccumulator imu_acc{imu_buffer};

  Eigen::Matrix4d acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 5));

  // target rotation in deg
  double target = -180;
  // target rotation r in radians
  double r = target * (M_PI / 180);
  Eigen::Matrix4d rotation_mat;
  rotation_mat <<  cos(r), -sin(r), 0, 0,
      sin(r),  cos(r), 0, 0,
      0,       0,      1, 0,
      0,       0,      0, 1;

  EXPECT_NEAR(acc(0,0), rotation_mat(0,0), 0.00001);
  EXPECT_NEAR(acc(0,1), rotation_mat(0,1), 0.00001);
  EXPECT_NEAR(acc(0,2), rotation_mat(0,2), 0.00001);
  EXPECT_NEAR(acc(1,0), rotation_mat(1,0), 0.00001);
  EXPECT_NEAR(acc(1,1), rotation_mat(1,1), 0.00001);
  EXPECT_NEAR(acc(1,2), rotation_mat(1,2), 0.00001);
  EXPECT_NEAR(acc(2,0), rotation_mat(2,0), 0.00001);
  EXPECT_NEAR(acc(2,1), rotation_mat(2,1), 0.00001);
  EXPECT_NEAR(acc(2,2), rotation_mat(2,2), 0.00001);

  EXPECT_TRUE(imu_buffer->size() ==  2);

  acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 7));

  // 2 imu messages left in buffer -> -90 grad rotation around z axis
  // target rotation in deg
  target = -90;
  // target rotation r in radians
  r = target * (M_PI / 180);
  rotation_mat.setIdentity();
  rotation_mat <<  cos(r), -sin(r), 0, 0,
      sin(r),  cos(r), 0, 0,
      0,       0,      1, 0,
      0,       0,      0, 1;

  EXPECT_NEAR(acc(0,0), rotation_mat(0,0), 0.00001);
  EXPECT_NEAR(acc(0,1), rotation_mat(0,1), 0.00001);
  EXPECT_NEAR(acc(0,2), rotation_mat(0,2), 0.00001);
  EXPECT_NEAR(acc(1,0), rotation_mat(1,0), 0.00001);
  EXPECT_NEAR(acc(1,1), rotation_mat(1,1), 0.00001);
  EXPECT_NEAR(acc(1,2), rotation_mat(1,2), 0.00001);
  EXPECT_NEAR(acc(2,0), rotation_mat(2,0), 0.00001);
  EXPECT_NEAR(acc(2,1), rotation_mat(2,1), 0.00001);
  EXPECT_NEAR(acc(2,2), rotation_mat(2,2), 0.00001);

  EXPECT_TRUE(imu_buffer->empty());
}

TEST(warpsense, accumulator_cloud)
{
  auto imu_buffer = std::make_shared<ConcurrentRingBuffer<sensor_msgs::Imu>>(10);

  auto stamp = ros::Time::now();
  auto diff = ros::Duration(250 * 0.001);

  sensor_msgs::Imu imu;
  imu.header.stamp = stamp;
  imu.angular_velocity.z = -M_PI;

  for (const auto& i: {1, 2, 3, 4, 5, 6, 7})
  {
    auto imu_msg = imu;
    imu_msg.header.stamp = stamp + ros::Duration(250 * 0.001 * i);
    imu_buffer->push(imu_msg);
  }

  ImuAccumulator imu_acc{imu_buffer};

  Eigen::Matrix4d acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 5));

  std::vector<Eigen::Vector3f> cloud(5);
  std::vector<Eigen::Vector3f> result(5);

  cloud[0] = Eigen::Vector3f{5, 0, 0};
  cloud[1] = Eigen::Vector3f{2, 2, 2};
  cloud[2] = Eigen::Vector3f{1, 2, 100};
  cloud[3] = Eigen::Vector3f{540, 244, 124};
  cloud[4] = Eigen::Vector3f{-3, -2, 2};

  transform_point_cloud(cloud, acc.cast<float>());

  result[0] = Eigen::Vector3f{-5, 0, 0};
  result[1] = Eigen::Vector3f{-2, -2, 2};
  result[2] = Eigen::Vector3f{-1, -2, 100};
  result[3] = Eigen::Vector3f{-540, -244, 124};
  result[4] = Eigen::Vector3f{3, 2, 2};

  for(size_t i=0; i < cloud.size(); i++)
  {
    EXPECT_NEAR(cloud[i].x(), result[i].x(), 0.00001);
    EXPECT_NEAR(cloud[i].y(), result[i].y(), 0.00001);
    EXPECT_NEAR(cloud[i].z(), result[i].z(), 0.00001);
  }

  acc = imu_acc.acc_transform(stamp + ros::Duration(250 * 0.001 * 7));

  transform_point_cloud(cloud, acc.cast<float>());

  result[0] = Eigen::Vector3f{0, 5, 0};
  result[1] = Eigen::Vector3f{-2, 2, 2};
  result[2] = Eigen::Vector3f{-2, 1, 100};
  result[3] = Eigen::Vector3f{-244, 540, 124};
  result[4] = Eigen::Vector3f{2, -3, 2};

  for(size_t i=0; i < cloud.size(); i++)
  {
    EXPECT_NEAR(cloud[i].x(), result[i].x(), 0.00001);
    EXPECT_NEAR(cloud[i].y(), result[i].y(), 0.00001);
    EXPECT_NEAR(cloud[i].z(), result[i].z(), 0.00001);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "params_b");
  ros::Time::init();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}