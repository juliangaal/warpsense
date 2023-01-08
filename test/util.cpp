// ros
#include <ros/ros.h>
#include <gtest/gtest.h>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

// warpsense
#include <warpsense/math/math.h>
#include <util/util.h>
#include <boost/filesystem.hpp>

namespace rm = rmagine;
namespace fs = boost::filesystem;

/// Fixed point scale
constexpr unsigned int SCALE = 1000;

/// Test Translation
constexpr float TX = 0.1 * SCALE;
constexpr float TY = 0.1 * SCALE;
constexpr float TZ = 0.0 * SCALE;

/// Test Rotation (radians)
constexpr float RY = 5 * (M_PI / 180);

void transform_compare(const std::vector<Point>& points, const std::vector<rm::Pointi>& points_rm, const Eigen::Matrix4f& mat)
{
  std::vector<Point> points_transformed(points);
  transform_point_cloud(points_transformed, mat);

  std::vector<rm::Pointi> points_rm_transformed(points_rm);
  transform_point_cloud(points_rm_transformed, *reinterpret_cast<const rm::Matrix4x4f*>(&mat));

  for (int i = 0; i < points.size(); ++i)
  {
    EXPECT_EQ(points_transformed[i].x(), points_rm_transformed[i].x);
    EXPECT_EQ(points_transformed[i].y(), points_rm_transformed[i].y);
    EXPECT_EQ(points_transformed[i].z(), points_rm_transformed[i].z);
  }
}

TEST(warpsense, util)
{
  // setup point data
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  fs::path filename = fs::path(DATA_PATH) / "frame_500.pcd";
  EXPECT_TRUE(pcl::io::loadPCDFile<pcl::PointXYZI>(filename.c_str(), *cloud) != -1);

  std::vector<Point> points_original(cloud->size());
  std::vector<rm::Pointi> points_rm_original(cloud->size());

#pragma omp parallel for schedule(static) default(shared)
  for (int i = 0; i < cloud->size(); ++i)
  {
    const auto& cp = (*cloud)[i];
    points_original[i] = Point(cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f);
    points_rm_original[i] = rm::Pointi (cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f);
  }

  for (int i = 1; i < 6; ++i)
  {
    auto i_f = static_cast<float>(i);

    Eigen::Matrix4f idle_mat;
    idle_mat << 1*i_f, 0, 0, 0,
                0, 1*i_f, 0, 0,
                0, 0, 1*i_f, 0,
                0, 0, 0, 1*i_f;

    transform_compare(points_original, points_rm_original, idle_mat);

    Eigen::Matrix4f translation_mat;
    translation_mat <<  1, 0, 0, TX*i_f,
                        0, 1, 0, TY*i_f,
                        0, 0, 1, TZ*i_f,
                        0, 0, 0, 1;

    transform_compare(points_original, points_rm_original, translation_mat);

    Eigen::Matrix4f rotation_mat;
    rotation_mat << cos(RY*i_f), -sin(RY*i_f), 0, 0,
                    sin(RY*i_f), cos(RY*i_f), 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

    transform_compare(points_original, points_rm_original, rotation_mat);

    Eigen::Matrix4f rotation_mat2;
    rotation_mat2 << cos(-RY*i_f), -sin(-RY*i_f), 0, 0,
                     sin(-RY*i_f), cos(-RY*i_f), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1;

    transform_compare(points_original, points_rm_original, rotation_mat2);

  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "util");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}