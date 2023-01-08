// ros
#include <ros/ros.h>
#include <boost/filesystem.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

// warpsense
#include "warpsense/types.h"
#include "util/util.h"
#include "warpsense/math/math.h"
#include "cpu/update_tsdf.h"
#include "map/hdf5_local_map.h"
#include "warpsense/tsdf_registration.h"
#include "warpsense/registration/registration.h"

namespace rm = rmagine;
namespace fs = boost::filesystem;

/// Fixed point scale
constexpr unsigned int SCALE = 1000;

/// Upper bounds for errors
constexpr float MAX_OFFSET = 100;
constexpr float DRIFT_OFFSET = 10;

/// Test Translation
constexpr float TX = 0.1 * SCALE;
constexpr float TY = 0.1 * SCALE;
constexpr float TZ = 0.0 * SCALE;
/// Test Rotation
constexpr float RY = 5 * (M_PI / 180); //radians


class Timer
{
private:
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;

public:
  Timer()
  {
    start();
  }

  ~Timer() = default;

  void start()
  {
    start_time = std::chrono::steady_clock::now();
  }

  void stop()
  {
    using namespace std::chrono;
    end_time = steady_clock::now();
    auto Duration = duration_cast<milliseconds>(end_time - start_time);
    std::cout << "Took " << Duration.count() << " ms." << std::endl;
  }
};

float check_computed_transform(const std::vector<Point>& points_posttransform, const std::vector<Point>& points_pretransform, bool print = true)
{
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  float average = 0;

  float average_x = 0;
  float average_y = 0;
  float average_z = 0;

  std::vector<float> dists(points_pretransform.size());

  for (size_t i = 0; i < points_pretransform.size(); i++)
  {
    Eigen::Vector3i sub = points_pretransform[i] - points_posttransform[i];
    float norm = sub.cast<float>().norm();

    if (norm < minimum)
    {
      minimum = norm;
    }

    if (norm > maximum)
    {
      maximum = norm;
    }

    average += norm;
    average_x += std::abs(sub.x());
    average_y += std::abs(sub.y());
    average_z += std::abs(sub.z());

    dists[i] = norm;
  }

  std::sort(dists.begin(), dists.end());

  average /= points_pretransform.size();
  average_x /= points_pretransform.size();
  average_y /= points_pretransform.size();
  average_z /= points_pretransform.size();

  if (print)
  {
    std::cout << std::fixed << std::setprecision(2)
              << "minimum distance: " << minimum << "\n"
              << "maximum distance: " << maximum << "\n"
              << "average distance: " << average
              << ",  (" << (int)average_x << ", " << (int)average_y << ", " << (int)average_z << ")\n"
              << "median  distance: " << dists[dists.size() / 2 + 1] << std::endl;
  }

  //CHECK(average < MAX_OFFSET);

  return average;
}

float check_computed_transform(const std::vector<rm::Pointi>& points_posttransform, const std::vector<rm::Pointi>& points_pretransform, bool print = true)
{
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  float average = 0;

  float average_x = 0;
  float average_y = 0;
  float average_z = 0;

  std::vector<float> dists(points_pretransform.size());

  for (size_t i = 0; i < points_pretransform.size(); i++)
  {
    rm::Pointi sub = points_pretransform[i] - points_posttransform[i];
    float norm = rm::Pointf((float)sub.x, (float)sub.y, (float)sub.z).l2norm();

    if (norm < minimum)
    {
      minimum = norm;
    }

    if (norm > maximum)
    {
      maximum = norm;
    }

    average += norm;
    average_x += std::abs(sub.x);
    average_y += std::abs(sub.y);
    average_z += std::abs(sub.z);

    dists[i] = norm;
  }

  std::sort(dists.begin(), dists.end());

  average /= points_pretransform.size();
  average_x /= points_pretransform.size();
  average_y /= points_pretransform.size();
  average_z /= points_pretransform.size();

  if (print)
  {
    std::cout << std::fixed << std::setprecision(2)
              << "minimum distance: " << minimum << "\n"
              << "maximum distance: " << maximum << "\n"
              << "average distance: " << average
              << ",  (" << (int)average_x << ", " << (int)average_y << ", " << (int)average_z << ")\n"
              << "median  distance: " << dists[dists.size() / 2 + 1] << std::endl;
  }

  //CHECK(average < MAX_OFFSET);

  return average;
}

float test_cpu(const Params &params, const HDF5LocalMap &map, const std::vector<Point> &points,
               const Eigen::Matrix4f &transformation_mat)
{
  // Pretransform the scan points
  std::vector<Point> points_transformed(points);
  transform_point_cloud(points_transformed, transformation_mat);

  std::cout << transformation_mat << "\n";

  // Register test scan

  Timer time;
  time.start();
  Eigen::Matrix4f result_matrix = register_cloud(map, points_transformed, Eigen::Matrix4f::Identity(), params.registration.max_iterations,
                                                 params.registration.it_weight_gradient, params.registration.epsilon, params.map.resolution);
  time.stop();

  //std::cout << result_matrix << "\n";

  // Retransform the points with the result transformation and compare them with the original
  transform_point_cloud(points_transformed, result_matrix);
  return check_computed_transform(points_transformed, points);
}

float
test_gpu(cuda::TSDFRegistration &gpu, const std::vector<Point> &points_original, const rm::Matrix4x4f &transformation_mat)
{
  // Pretransform the scan points
  std::vector<rm::Pointi> points(points_original.size());
  std::vector<rm::Pointi> points_transformed(points_original.size());
  #pragma omp parallel for schedule(static) default(shared)
  for (int i = 0; i < points_original.size(); ++i)
  {
    const auto& cp = points_original[i];
    points_transformed[i] = rm::Pointi(cp.x(), cp.y(), cp.z());
    points[i] = points_transformed[i];
  }

  transform_point_cloud(points_transformed, transformation_mat);

  auto tmp = transformation_mat;
  std::cout << *reinterpret_cast<Eigen::Matrix4f*>(&tmp) << "\n";

  Timer time;
  time.start();
  auto result_matrix = gpu.register_cloud(points_transformed, Eigen::Matrix4f::Identity());
  time.stop();

  //std::cout << result_matrix << "\n";
  // Retransform the points with the result transformation and compare them with the original
  transform_point_cloud(points_transformed, *reinterpret_cast<rm::Matrix4x4f*>(&result_matrix));
  return check_computed_transform(points_transformed, points);
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "registration");
  ros::NodeHandle nh("~");
  Params params(nh);

  // setup point data
  pcl::PointCloud<pcl::PointXYZI>::Ptr orig(new pcl::PointCloud<pcl::PointXYZI>);
  fs::path filename = fs::path(DATA_PATH) / "frame_500.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZI> (filename.string(), *orig) == -1)
  {
    PCL_ERROR("Couldn't read test pcd\n");
    return (-1);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(orig);
  sor.setLeafSize(0.064f, 0.064f, 0.064f);
  sor.filter(*cloud);

  std::vector<Point> points_original(cloud->size());
  std::vector<rm::Pointi> points_rm_original(cloud->size());

#pragma omp parallel for schedule(static) default(shared)
  for (int i = 0; i < cloud->size(); ++i)
  {
    const auto& cp = (*cloud)[i];
    points_original[i] = Point(cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f);
    points_rm_original[i] = rm::Pointi (cp.x * 1000.f, cp.y * 1000.f, cp.z * 1000.f);
  }

  params.map.tau = 1000;

  // setup local map
  std::shared_ptr<HDF5GlobalMap> global_map;
  std::shared_ptr<HDF5LocalMap> local_map_cpu;
  global_map.reset(new HDF5GlobalMap(params.map));
  local_map_cpu.reset(new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), global_map));

  // Shift
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  int x = (int)std::floor(pose.translation().x() * 1000 / params.map.resolution);
  int y = (int)std::floor(pose.translation().y() * 1000 / params.map.resolution);
  int z = (int)std::floor(pose.translation().z() * 1000 / params.map.resolution);
  Eigen::Vector3i pos(x, y, z);
  local_map_cpu->shift(pos);

  // create copy of local map before cpu performs update on it
  auto cuda_local_map = std::make_shared<HDF5LocalMap>(*local_map_cpu);

  Eigen::Matrix4i rot = Eigen::Matrix4i::Identity();
  rot.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rot);

  // create TSDF Volume
  update_tsdf(points_original, pos, up, *local_map_cpu, params.map.tau, params.map.max_weight, params.map.resolution);

  printf("%s:%d : %d %d %d\n", __FILE__, __LINE__, params.map.size.x(), params.map.size.y(), params.map.size.z());

  // create TSDF Volume (GPU)
  rm::Pointi pos_rm = *reinterpret_cast<const rm::Pointi*>(&pos);
  rm::Pointi up_rm = *reinterpret_cast<const rm::Pointi*>(&up);
  cuda::TSDFRegistration gpu(params, cuda_local_map);
  gpu.update_tsdf(points_rm_original, pos_rm, up_rm);

  Eigen::Matrix4f idle_mat;
  idle_mat << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1;

  Eigen::Matrix4f translation_mat;
  translation_mat << 1, 0, 0, TX,
      0, 1, 0, TY,
      0, 0, 1, TZ,
      0, 0, 0,  1;

  Eigen::Matrix4f rotation_mat;
  rotation_mat <<  cos(RY), -sin(RY), 0, 0,
      sin(RY),  cos(RY), 0, 0,
      0,        0, 1, 0,
      0,        0, 0, 1;

  Eigen::Matrix4f rotation_mat2;
  rotation_mat2 <<  cos(-RY), -sin(-RY), 0, 0,
      sin(-RY),  cos(-RY), 0, 0,
      0,         0, 1, 0,
      0,         0, 0, 1;

  // Calculate TSDF values for the points from the pcd and store them in the local map

  {
    std::cout << "Test Registration (CPU) No Transform TSDF (CPU)" << std::endl;
    test_cpu(params, *local_map_cpu, points_original, idle_mat);
  }

  {
    std::cout << "Test Registration (GPU) No Transform TSDF (GPU)" << std::endl;
    test_gpu(gpu, points_original, *reinterpret_cast<rm::Matrix4x4f *>(&idle_mat));
  }

  {
    std::cout << "Test Registration (CPU) Translation TSDF (CPU)" << std::endl;
    test_cpu(params, *local_map_cpu, points_original, translation_mat);
  }

  {
    std::cout << "Test Registration (GPU) Translation TSDF (GPU)" << std::endl;
    test_gpu(gpu, points_original, *reinterpret_cast<rm::Matrix4x4f *>(&translation_mat));
  }

  {
    std::cout << "Test Registration (CPU) Rotation TSDF (CPU)" << std::endl;
    test_cpu(params, *local_map_cpu, points_original, rotation_mat);
  }

  {
    std::cout << "Test Registration (GPU) Rotation TSDF (GPU)" << std::endl;
    test_gpu(gpu, points_original, *reinterpret_cast<rm::Matrix4x4f *>(&rotation_mat));
  }

}
