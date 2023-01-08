#include <warpsense/test/common.h>
#include <warpsense/math/math.h>
#include <util/util.h>
#include <warpsense/cuda/common.cuh>
#include "warpsense/cuda/playground.h"
#include "test.h"
#include <warpsense/types.h>
#include <cpu/update_tsdf.h>
#include <map/hdf5_local_map.h>
#include <warpsense/registration/registration.h>
#include <warpsense/cuda/device_map.h>
#include <iostream>
#include <random>
#include <algorithm>

namespace rm = rmagine;

#define RUN(name) std::cout << "Running " << #name; name(); std::cout << " ✓ \n"

template <typename Prob = double>
bool random_bool(const Prob p = 0.5) {
  static auto dev = std::random_device();
  static auto gen = std::mt19937{dev()};
  static auto dist = std::uniform_real_distribution<Prob>(0,1);
  return (dist(gen) < p);
}

void test_cuda_map_bounds()
{
  std::shared_ptr<HDF5GlobalMap> gm_ptr = std::make_shared<HDF5GlobalMap>("/tmp/test4_cuda.h5", DEFAULT_VALUE, DEFAULT_WEIGHT);
  std::shared_ptr<HDF5LocalMap> local_map_ptr_ = std::make_shared<HDF5LocalMap>(5, 5, 5, gm_ptr);

  /*
   * Write some tsdf values and weights into the local map,
   * that will be written to the file as one chunk (-1_0_0)
   *
   *    y
   *    ^
   *  4 | .   .   .   . | .   .
   *    | Chunk -1_0_0  | Chunk 0_0_0
   *  3 | .   .   .   . | .   .
   *    |               |
   *  2 | .   .  p0  p1 | .   .
   *    |               |
   *  1 | .   .  p2  p3 | .   .
   *    |               |
   *  0 | .   .  p4  p5 | .   .
   *    | --------------+------
   * -1 | .   .   .   . | .   .
   *    | Chunk -1_-1_0 | Chunk 0_-1_0
   * -2 | .   .   .   . | .   .
   *    +-----------------------> x
   *   / -4  -3  -2  -1   0   1
   * z=0
   */

  // Test CUDA map
  cuda::DeviceMap cuda_map((int *)&local_map_ptr_->get_size(),
                           (int *)&local_map_ptr_->get_offset(),
                           local_map_ptr_->get_data(),
                           (int *)&local_map_ptr_->get_pos());

  cuda::test_map_bounds(cuda_map, true);
  
  TSDFEntry p0(0, 0);
  TSDFEntry p1(1, 1);
  TSDFEntry p2(2, 1);
  TSDFEntry p3(3, 2);
  TSDFEntry p4(4, 3);
  TSDFEntry p5(5, 5);
  local_map_ptr_->value(-2, 2, 0) = p0;
  local_map_ptr_->value(-1, 2, 0) = p1;
  local_map_ptr_->value(-2, 1, 0) = p2;
  local_map_ptr_->value(-1, 1, 0) = p3;
  local_map_ptr_->value(-2, 0, 0) = p4;
  local_map_ptr_->value(-1, 0, 0) = p5;

  // test getter
  EXPECT_TRUE(local_map_ptr_->get_pos() == Eigen::Vector3i(0, 0, 0));
  EXPECT_TRUE(local_map_ptr_->get_size() == Eigen::Vector3i(5, 5, 5));
  EXPECT_TRUE(local_map_ptr_->get_offset() == Eigen::Vector3i(2, 2, 2));
  EXPECT_TRUE(*cuda_map.get_pos() == rm::Vector3i(0, 0, 0));
  EXPECT_TRUE(*cuda_map.get_size() == rm::Vector3i(5, 5, 5));
  EXPECT_TRUE(*cuda_map.get_offset() == rm::Vector3i(2, 2, 2));

  // test in_bounds
  EXPECT_TRUE(local_map_ptr_->in_bounds(0, 2, -2));
  EXPECT_TRUE(!local_map_ptr_->in_bounds(22, 0, 0));
  EXPECT_TRUE(cuda_map.in_bounds(0, 2, -2));
  EXPECT_TRUE(!cuda_map.in_bounds(22, 0, 0));

  // test default values
  EXPECT_TRUE(local_map_ptr_->value(0, 0, 0).value() == DEFAULT_VALUE);
  EXPECT_TRUE(local_map_ptr_->value(0, 0, 0).weight() == DEFAULT_WEIGHT);
  EXPECT_TRUE(cuda_map.value_unchecked(0, 0, 0).value() == DEFAULT_VALUE);
  EXPECT_TRUE(cuda_map.value_unchecked(0, 0, 0).weight() == DEFAULT_WEIGHT);

  // test value access
  EXPECT_TRUE(local_map_ptr_->value(-1, 2, 0).value() == 1);
  EXPECT_TRUE(local_map_ptr_->value(-1, 2, 0).weight() == 1);
  EXPECT_TRUE(cuda_map.value_unchecked(-1, 2, 0).value() == 1);
  EXPECT_TRUE(cuda_map.value_unchecked(-1, 2, 0).weight() == 1);

  cuda::test_map_bounds(cuda_map, false);
}

void test_cuda_map_write()
{
  Eigen::Vector3i size(5, 5, 5);
  std::shared_ptr<HDF5GlobalMap> global_map = std::make_shared<HDF5GlobalMap>("/tmp/test_cuda_write.h5", DEFAULT_VALUE, DEFAULT_WEIGHT);
  std::shared_ptr<HDF5LocalMap> local_map = std::make_shared<HDF5LocalMap>(size.x(), size.y(), size.z(), global_map);

  TSDFEntry entry(1, 1);

  std::vector<bool> random_idxs(size.prod());
  std::generate(random_idxs.begin(), random_idxs.end(), []() { return random_bool(); });
  std::vector<rmagine::Pointi> gpu_access;
  gpu_access.reserve(random_idxs.size() / 2);

  int start = -local_map->get_size().x() / 2;
  int end = local_map->get_size().x() / 2;
  for (int i = start, ii = 0; i < end; ++i, ++ii)
  {
    for (int j = start, jj = 0; j < end; ++j, ++jj)
    {
      for (int k = start, kk = 0; k < end; ++k, ++kk)
      {
        int idx = (ii % size.x()) * size.y() * size.z() + (jj % size.y()) * size.z() + kk % size.z();
        if (random_idxs[idx])
        {
          local_map->value(i, j, k) = entry;
          gpu_access.emplace_back(i, j, k);
        }
      }
    }
  }

  auto previously_written = [](const std::vector<rmagine::Pointi>& v, int i, int j, int k)
  {
    return std::find_if(v.begin(), v.end(), [&](const rmagine::Pointi& p)
    {
      return i == p.x && j == p.y && k == p.z;
    }
    ) != v.end();
  };


  // Test copy constructor

  {
    auto local_map_cuda = std::make_shared<HDF5LocalMap>(*local_map);

    for (int i = start; i < end; ++i)
    {
      for (int j = start; j < end; ++j)
      {
        for (int k = start; k < end; ++k)
        {
          if (previously_written(gpu_access, i, j, k))
          {
            EXPECT_EQ(local_map_cuda->value(i, j, k), entry);
            EXPECT_EQ(local_map_cuda->value(i, j, k), local_map->value(i, j, k));
          } else
          {
            EXPECT_EQ(local_map_cuda->value(i, j, k), TSDFEntry(DEFAULT_VALUE, DEFAULT_WEIGHT));
            EXPECT_EQ(local_map_cuda->value(i, j, k), local_map->value(i, j, k));
          }
        }
      }
    }
  }

  // Test CUDA map write (!DeviceMap, see cuda/map.h)
  {
    std::shared_ptr<HDF5GlobalMap> global_map_cuda = std::make_shared<HDF5GlobalMap>("/tmp/test_cuda_write2.h5", DEFAULT_VALUE,
                                                                                     DEFAULT_WEIGHT);
    std::shared_ptr<HDF5LocalMap> local_map_cuda = std::make_shared<HDF5LocalMap>(size.x(), size.y(), size.z(),
                                                                                  global_map_cuda);

    cuda::DeviceMap cuda_map((int *) &local_map_cuda->get_size(),
                             (int *) &local_map_cuda->get_offset(),
                             local_map_cuda->get_data(),
                             (int *) &local_map_cuda->get_pos());

    cuda::test_map_write(cuda_map, gpu_access, entry);

    for (int i = start; i < end; ++i)
    {
      for (int j = start; j < end; ++j)
      {
        for (int k = start; k < end; ++k)
        {
          if (previously_written(gpu_access, i, j, k))
          {
            EXPECT_EQ(local_map_cuda->value(i, j, k), entry);
            EXPECT_EQ(local_map_cuda->value(i, j, k), local_map->value(i, j, k));
          } else
          {
            EXPECT_EQ(local_map_cuda->value(i, j, k), TSDFEntry(DEFAULT_VALUE, DEFAULT_WEIGHT));
            EXPECT_EQ(local_map_cuda->value(i, j, k), local_map->value(i, j, k));
          }
        }
      }
    }
  }
}

void test_cuda_calc_jacobis()
{
  // tsdf parameters
  float max_distance = 3;
  int map_resolution = 1000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  float map_size_x = 20 * 1000 / map_resolution;
  float map_size_y = 20 * 1000 / map_resolution;
  float map_size_z = 20 * 1000 / map_resolution;

  std::vector<Point> points;
  Point p;
  p.x() = static_cast<int>(5.5 * 1000.f); // cell center in x direction
  p.y() = static_cast<int>(0.5 * 1000.f); // cell center in y direction
  p.z() = static_cast<int>(0.5 * 1000.f); // cell center in z direction
  points.push_back(p);

  std::vector<rm::Pointi> rm_points;
  rm::Pointi rm_p;
  rm_p.x = static_cast<int>(5.5 * 1000.f); // cell center in x direction
  rm_p.y = static_cast<int>(0.5 * 1000.f); // cell center in y direction
  rm_p.z = static_cast<int>(0.5 * 1000.f); // cell center in z direction
  rm_points.push_back(rm_p);

  std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
  std::shared_ptr<HDF5LocalMap> local_map_ptr_;
  global_map_ptr_.reset(new HDF5GlobalMap("/tmp/test.h5", tau, 0.0));
  local_map_ptr_.reset(new HDF5LocalMap(map_size_x, map_size_y, map_size_z, global_map_ptr_));

  // set pose
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

  // set ray marching pos
  int x = (int)std::floor(pose.translation().x()) * 1000 / map_resolution;
  int y = (int)std::floor(pose.translation().y()) * 1000 / map_resolution;
  int z = (int)std::floor(pose.translation().z()) * 1000 / map_resolution;
  Eigen::Vector3i pos(x, y, z);

  // perform tsdf update
  Eigen::Matrix4i rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);
  update_tsdf(points, pos, up, *local_map_ptr_, tau, max_weight, map_resolution);
  register_cloud(*local_map_ptr_, points, Eigen::Matrix4f::Identity(), 200, 6, 0.001, 0);


  // calc jacobis on gpu
  cuda::DeviceMap cuda_map((int *)&local_map_ptr_->get_size(),
                           (int *)&local_map_ptr_->get_offset(),
                           local_map_ptr_->get_data(),
                           (int *)&local_map_ptr_->get_pos());

  std::vector<rm::Point6l> gradients;
  std::vector<TSDFEntry::ValueType> values;
  std::vector<bool> mask;
  cuda::calc_jacobis(cuda_map, rm_points, gradients, values, mask);
}

void test_cuda_map_adapter()
{
  // tsdf parameters
  float max_distance = 3;
  int map_resolution = 1000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  float map_size_x = 20 * 1000 / map_resolution;
  float map_size_y = 20 * 1000 / map_resolution;
  float map_size_z = 20 * 1000 / map_resolution;

  std::vector<Point> points;
  Point p;
  p.x() = static_cast<int>(5.5 * 1000.f); // cell center in x direction
  p.y() = static_cast<int>(0.5 * 1000.f); // cell center in y direction
  p.z() = static_cast<int>(0.5 * 1000.f); // cell center in z direction
  points.push_back(p);

  std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
  std::shared_ptr<HDF5LocalMap> local_map_ptr_;
  global_map_ptr_.reset(new HDF5GlobalMap("/tmp/test.h5", tau, 0.0));
  local_map_ptr_.reset(new HDF5LocalMap(map_size_x, map_size_y, map_size_z, global_map_ptr_));

  // set pose
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

  // set ray marching pos
  int x = (int)std::floor(pose.translation().x()) * 1000 / map_resolution;
  int y = (int)std::floor(pose.translation().y()) * 1000 / map_resolution;
  int z = (int)std::floor(pose.translation().z()) * 1000 / map_resolution;
  Eigen::Vector3i pos(x, y, z);

  // perform tsdf update
  Eigen::Matrix4i rotation_mat = Eigen::Matrix4i::Identity();
  rotation_mat.block<3, 3>(0, 0) = to_int_mat(pose).block<3, 3>(0, 0);
  Point up = transform_point(Point(0, 0, MATRIX_RESOLUTION), rotation_mat);
  update_tsdf(points, pos, up, *local_map_ptr_, tau, max_weight, map_resolution);

  local_map_ptr_->write_back();

  auto val = local_map_ptr_->value(1, 0, 0).value();
  auto weight = local_map_ptr_->value(1, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(2, 0, 0).value();
  weight = local_map_ptr_->value(2, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(3, 0, 0).value();
  weight = local_map_ptr_->value(3, 0, 0).weight();
  EXPECT_EQ(val, 2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(4, 0, 0).value();
  weight = local_map_ptr_->value(4, 0, 0).weight();
  EXPECT_EQ(val,1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(5, 0, 0).value();
  weight = local_map_ptr_->value(5, 0, 0).weight();
  EXPECT_EQ(val, 0);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(6, 0, 0).value();
  weight = local_map_ptr_->value(6, 0, 0).weight();
  EXPECT_EQ(val, -1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(7, 0, 0).value();
  weight = local_map_ptr_->value(7, 0, 0).weight();
  EXPECT_EQ(val, -2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = local_map_ptr_->value(8, 0, 0).value();
  weight = local_map_ptr_->value(8, 0, 0).weight();
  EXPECT_EQ(val, 3000); // default val, weight == 0
  EXPECT_EQ(weight, 0);

  // Test CUDA map adapter
  cuda::DeviceMap cuda_map((int *)&local_map_ptr_->get_size(),
                           (int *)&local_map_ptr_->get_offset(),
                           local_map_ptr_->get_data(),
                           (int *)&local_map_ptr_->get_pos());

  // size
  EXPECT_EQ(cuda_map.get_size()->x, local_map_ptr_->get_size().x());
  EXPECT_EQ(cuda_map.get_size()->y, local_map_ptr_->get_size().y());
  EXPECT_EQ(cuda_map.get_size()->z, local_map_ptr_->get_size().z());
  // offset
  EXPECT_EQ(cuda_map.get_offset()->x, local_map_ptr_->get_offset().x());
  EXPECT_EQ(cuda_map.get_offset()->y, local_map_ptr_->get_offset().y());
  EXPECT_EQ(cuda_map.get_offset()->z, local_map_ptr_->get_offset().z());
  // data
  // TODO EXPECT_EQ();
  // pos
  EXPECT_EQ(cuda_map.get_pos()->x, local_map_ptr_->get_pos().x());
  EXPECT_EQ(cuda_map.get_pos()->y, local_map_ptr_->get_pos().y());
  EXPECT_EQ(cuda_map.get_pos()->z, local_map_ptr_->get_pos().z());

  // values
  val = cuda_map.value_unchecked(1, 0, 0).value();
  weight = cuda_map.value_unchecked(1, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(2, 0, 0).value();
  weight = cuda_map.value_unchecked(2, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(3, 0, 0).value();
  weight = cuda_map.value_unchecked(3, 0, 0).weight();
  EXPECT_EQ(val, 2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(4, 0, 0).value();
  weight = cuda_map.value_unchecked(4, 0, 0).weight();
  EXPECT_EQ(val,1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(5, 0, 0).value();
  weight = cuda_map.value_unchecked(5, 0, 0).weight();
  EXPECT_EQ(val, 0);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(6, 0, 0).value();
  weight = cuda_map.value_unchecked(6, 0, 0).weight();
  EXPECT_EQ(val, -1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(7, 0, 0).value();
  weight = cuda_map.value_unchecked(7, 0, 0).weight();
  EXPECT_EQ(val, -2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = cuda_map.value_unchecked(8, 0, 0).value();
  weight = cuda_map.value_unchecked(8, 0, 0).weight();
  EXPECT_EQ(val, 3000); // default val, weight == 0
  EXPECT_EQ(weight, 0);

  EXPECT_FALSE(sizeof(*local_map_ptr_) == sizeof(cuda_map));

  cuda::test_map_adapter(cuda_map, tau, weight_epsilon);
}

void test_cuda_h_g_e_reduction()
{
  using Vector6f = Eigen::Matrix<float, 6, 1>;
  using Matrix6f = Eigen::Matrix<float, 6, 6>;

  // single element
  {
    Vector6f ve;
    ve << 1.f, 2.f, 3.f, 4.f, 5.f, 6.f;

    Matrix6f cpu_h = Matrix6f::Zero();
    Vector6f cpu_g = Vector6f::Zero();
    float cpu_e = 0.f;

    cpu_h += ve * ve.transpose();
    cpu_g += ve * ve[0];
    cpu_e += 1;

    std::vector<rm::Vector6f> v1 = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    rm::Matrix6x6f gpu_h;
    rm::Vector6f gpu_g;
    float gpu_e;
    float gpu_c;
    cuda::h_g_e_reduction(v1, gpu_h, gpu_g, gpu_e, gpu_c);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_h(i, j), gpu_h.at(i, j));
      }
      EXPECT_EQ_FLOAT(cpu_g[i], gpu_g.at(i));
    }
    EXPECT_EQ_FLOAT(cpu_e, gpu_e);
  }

  // multiple elements
  {
    size_t n_elements = 1024;

    Vector6f ve;
    ve << 1.f, 2.f, 3.f, 4.f, 5.f, 6.f;
    std::vector<Vector6f> ves(n_elements);
    std::fill(ves.begin(), ves.end(), ve);

    Matrix6f cpu_h = Matrix6f::Zero();
    Vector6f cpu_g = Vector6f::Zero();
    float cpu_e = 0.f;

    for (const auto& vei: ves)
    {
      cpu_h += vei * vei.transpose();
      cpu_g += vei * vei[0];
      cpu_e += 1;
    }

    rm::Vector6f v =  {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<rm::Vector6f> v1(n_elements);
    std::fill(v1.begin(), v1.end(), v);

    rm::Matrix6x6f gpu_h;
    rm::Vector6f gpu_g;
    float gpu_e;
    float gpu_c;
    cuda::h_g_e_reduction(v1, gpu_h, gpu_g, gpu_e, gpu_c);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_h(i, j), gpu_h.at(i, j));
      }
      EXPECT_EQ_FLOAT(cpu_g[i], gpu_g.at(i));
    }
    EXPECT_EQ_FLOAT(cpu_e, gpu_e);
  }

  {
    size_t n_elements = 2000;

    Vector6f ve;
    ve << 1.f, 2.f, 3.f, 4.f, 5.f, 6.f;
    std::vector<Vector6f> ves(n_elements);
    std::fill(ves.begin(), ves.end(), ve);

    Matrix6f cpu_h = Matrix6f::Zero();
    Vector6f cpu_g = Vector6f::Zero();
    float cpu_e = 0.f;

    for (const auto& vei: ves)
    {
      cpu_h += vei * vei.transpose();
      cpu_g += vei * vei[0];
      cpu_e += 1;
    }

    rm::Vector6f v =  {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<rm::Vector6f> v1(n_elements);
    std::fill(v1.begin(), v1.end(), v);

    rm::Matrix6x6f gpu_h;
    rm::Vector6f gpu_g;
    float gpu_e;
    float gpu_c;
    cuda::h_g_e_reduction(v1, gpu_h, gpu_g, gpu_e, gpu_c);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_h(i, j), gpu_h.at(i, j));
      }
      EXPECT_EQ_FLOAT(cpu_g[i], gpu_g.at(i));
    }
    EXPECT_EQ_FLOAT(cpu_e, gpu_e);
  }
}

void test_cuda_cov()
{
  // single element
  {
    Eigen::Vector3f ve(1, 2, 3);
    auto cpu_cov = ve * ve.transpose();

    std::vector<rm::Vector3f> v1 = {{1, 2, 3}};
    rm::Matrix3x3f gpu_cov = cuda::cov(v1, v1);

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_cov(i, j), gpu_cov.at(i, j));
      }
    }
  }

  // multiple elements
  {
    Eigen::Vector3f v1e(1, 2, 3);
    Eigen::Vector3f v2e(2, 3, 4);
    Eigen::Matrix3f cpu_cov = v1e * v1e.transpose();
    cpu_cov += v2e * v2e.transpose();
    cpu_cov /= 2.0f;

    std::vector<rm::Vector3f> v1 = {{1, 2, 3}, {2, 3, 4}};
    std::vector<rm::Vector3f> v2 = {{1, 2, 3}, {2, 3, 4}};
    rm::Matrix3x3f gpu_cov = cuda::cov(v1, v2);

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_cov(i, j), gpu_cov.at(i, j));
      }
    }
  }
}

void test_cuda_cov6d()
{
  // single element
  {
    using Vector6f = Eigen::Matrix<float, 6, 1>;
    Vector6f ve;
    ve << 1.f, 2.f, 3.f, 4.f, 5.f, 6.f;

    auto cpu_cov = ve * ve.transpose();

    std::vector<rm::Vector6f> v1 = {{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}};
    rm::Matrix6x6f gpu_cov = cuda::cov6d(v1, v1);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_cov(i, j), gpu_cov.at(i, j));
      }
    }
  }

  // multiple
  {
    using Vector6f = Eigen::Matrix<float, 6, 1>;
    using Matrix6f = Eigen::Matrix<float, 6, 6>;

    size_t n_elements = 2;

    Vector6f ve;
    ve << .1f, .2f, .3f, .4f, .5f, .6f;
    std::vector<Vector6f> covariancese(n_elements);
    std::fill(covariancese.begin(), covariancese.end(), ve);

    Matrix6f cpu_cov = Matrix6f::Zero();

    for (const auto& v: covariancese)
    {
      cpu_cov += v * v.transpose();
    }

    cpu_cov /= static_cast<float>(n_elements);

    rm::Vector6f v{.1f, .2f, .3f, .4f, .5f, .6f};
    std::vector<rm::Vector6f> covariancesgpu(n_elements);
    std::fill(covariancesgpu.begin(), covariancesgpu.end(), v);

    rm::Matrix6x6f gpu_cov = cuda::cov6d(covariancesgpu, covariancesgpu);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT(cpu_cov(i, j), gpu_cov.at(i, j));
      }
    }
  }

  // multiple
  {
    using Vector6f = Eigen::Matrix<float, 6, 1>;
    using Matrix6f = Eigen::Matrix<float, 6, 6>;

    size_t n_elements = 1024;

    Vector6f ve;
    ve << .1f, .2f, .3f, .4f, .5f, .6f;
    std::vector<Vector6f> covariancese(n_elements);
    std::fill(covariancese.begin(), covariancese.end(), ve);

    Matrix6f cpu_cov = Matrix6f::Zero();

    for (const auto &v: covariancese)
    {
      cpu_cov += v * v.transpose();
    }

    cpu_cov /= static_cast<float>(n_elements);

    rm::Vector6f v{.1f, .2f, .3f, .4f, .5f, .6f};
    std::vector<rm::Vector6f> covariancesgpu(n_elements);
    std::fill(covariancesgpu.begin(), covariancesgpu.end(), v);

    rm::Matrix6x6f gpu_cov = cuda::cov6d(covariancesgpu, covariancesgpu);

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ_FLOAT_E(cpu_cov(i, j), gpu_cov.at(i, j), 0.01);
      }
    }
  }

}

void test_cuda_sum()
{
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist{0, 2};

  auto gen = [&dist, &mersenne_engine](){
    return static_cast<float>(dist(mersenne_engine));
  };

  {
    std::vector<float> vec(15000000);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }

  {
    std::vector<float> vec(2048);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }

  {
    std::vector<float> vec(1023);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }
}

void test_cuda_sum_split()
{
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist{0, 2};

  auto gen = [&dist, &mersenne_engine](){
    return static_cast<float>(dist(mersenne_engine));
  };

  {
    std::vector<float> vec(1024 * 2);
    std::fill(vec.begin(), vec.end(), 1.f);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec, 2);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }

  {
    std::vector<float> vec(15000000);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec, 8);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }

  {
    std::vector<float> vec(2048);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec, 8);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }

  {
    std::vector<float> vec(1023);
    generate(vec.begin(), vec.end(), gen);

    float cpu_sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    float gpu_sum = cuda::sum(vec, 8);
    EXPECT_EQ_FLOAT(cpu_sum, gpu_sum);
  }
}

void test_cuda_transform_point()
{
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  float theta = M_PI / 2;
  transform(0, 0) = std::cos(theta);
  transform(0, 1) = -sin(theta);
  transform(1, 0) = sin(theta);
  transform(1, 1) = std::cos(theta);
  // TODO test translation

  {
    auto mat = to_int_mat(transform);
    Point p(1, 0, 0);
    auto out = transform_point(p, mat);
    EXPECT_EQ(out, Point(0, 1, 0));
  }

  {
    auto temp = to_int_mat(transform);

    rm::Matrix4x4i mat;
    mat.setZeros();

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        mat.at(i, j) = temp(i, j);
      }
    }

    rm::Pointi p(1, 0, 0);
    rm::Pointi out = cuda::transform_point(p, mat);
    EXPECT_EQ(out, rm::Pointi(0, 1, 0));
  }

  theta = -M_PI / 2;
  transform(0, 0) = std::cos(theta);
  transform(0, 1) = -sin(theta);
  transform(1, 0) = sin(theta);
  transform(1, 1) = std::cos(theta);

  {
    auto mat = to_int_mat(transform);
    Point p(1, 0, 0);
    auto out = transform_point(p, mat);
    EXPECT_EQ(out, Point(0, -1, 0));
  }

  {
    auto temp = to_int_mat(transform);

    rm::Matrix4x4i mat;
    mat.setZeros();

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        mat.at(i, j) = temp(i, j);
      }
    }

    rm::Pointi p(1, 0, 0);
    rm::Pointi out = cuda::transform_point(p, mat);
    EXPECT_EQ(out, rm::Pointi(0, -1, 0));
  }
}

void to_rm(const Point6l& j1, rm::Point6l& j2)
{
  for (int i = 0; i < 6; ++i)
  {
    j2.at(i) = j1(i);
  }
}

void test_cuda_jacobi_device()
{
  {
    Point6l jacobi;
    jacobi << 0, 1, 2, 3, 4, 5;

    Matrix6l target = jacobi * jacobi.transpose();
    rm::Matrix6x6l cuda_target = rm::Matrix6x6l::Zero();
    rm::Point6l jacobirm = rm::Point6l::Zero();

    {
      to_rm(jacobi, jacobirm);
      cuda::jacobi_2_h(jacobirm, cuda_target);
    }

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ(target(i, j), cuda_target.at(i, j));
      }
    }
  }

  {
    Point6l jacobi;
    jacobi << -1, 1, 2, 3, 4, -5;
    Matrix6l target = jacobi * jacobi.transpose();
    rm::Matrix6x6l cuda_target = rm::Matrix6x6l::Zero();
    rm::Point6l jacobirm = rm::Point6l::Zero();

    {
      to_rm(jacobi, jacobirm);
      cuda::jacobi_2_h(jacobirm, cuda_target);
    }

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ(target(i, j), cuda_target.at(i, j));
      }
    }
  }

  {
    Point6l jacobi;
    jacobi << 0, 1, -2, 12, 4, 5;
    Matrix6l target = jacobi * jacobi.transpose();
    rm::Matrix6x6l cuda_target = rm::Matrix6x6l::Zero();
    rm::Point6l jacobirm = rm::Point6l::Zero();

    {
      to_rm(jacobi, jacobirm);
      cuda::jacobi_2_h(jacobirm, cuda_target);
    }

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ(target(i, j), cuda_target.at(i, j));
      }
    }
  }

  {
    Point6l jacobi;
    jacobi << 0, -1, 20, 3, -4, 5;
    Matrix6l target = jacobi * jacobi.transpose();
    rm::Matrix6x6l cuda_target = rm::Matrix6x6l::Zero();
    rm::Point6l jacobirm = rm::Point6l::Zero();

    {
      to_rm(jacobi, jacobirm);
      cuda::jacobi_2_h(jacobirm, cuda_target);
    }

    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        EXPECT_EQ(target(i, j), cuda_target.at(i, j));
      }
    }
  }
}

void test_cuda_cross()
{
  Point a(1, 2, 3);
  Point b(1, 2, 3);

  auto target = a.cross(b);

  rm::Pointi arm(1, 2, 3);
  rm::Pointi brm(1, 2, 3);
  rm::Pointi res;
  cuda::cross(arm, brm, res);
  EXPECT_EQ(target.x(), res.x);
  EXPECT_EQ(target.y(), res.y);
  EXPECT_EQ(target.z(), res.z);
}

void test_cuda_data_sizes()
{
  cuda::test_data_sizes();
  // if there is no check, test fails: dummy check
  EXPECT_TRUE(true);
}

void test_cuda_atomic_min()
{
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(1,10000); // distribution in range [1, 10000]

  auto n_numbers = 1000000;
  std::vector<int> random_numbers(n_numbers);
  std::generate(random_numbers.begin(), random_numbers.end(), [&]() { return dist(rng); });

  auto min_iterator = std::min_element(random_numbers.begin(), random_numbers.end());
  auto min_idx = std::distance(random_numbers.begin(), min_iterator);
  int min = random_numbers[min_idx];


  std::vector<int> buffer = { std::numeric_limits<int>::max() };
  auto min_cuda = cuda::atomic_min(buffer, random_numbers);
  EXPECT_EQ(min, min_cuda);
}

void test_cuda_atomic_tsdf_min()
{
  TSDFEntry entry(DEFAULT_VALUE, DEFAULT_WEIGHT);
  EXPECT_EQ(*((uint16_t*)&entry), entry.value());
  EXPECT_EQ(*(((uint16_t*)&entry) + 1), entry.weight());

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int16_t> dist(1,1000); // distribution in range [1, 10000]

  auto n_numbers = 100000;
  std::vector<TSDFEntry> random_numbers(n_numbers);
  std::generate(random_numbers.begin(), random_numbers.end(), [&]() { return TSDFEntry(dist(rng), 0); });

  auto comp = [](const auto& a, const auto& b) { return a.value() < b.value() || a.weight() > 0; };
  auto min_iterator = std::min_element(random_numbers.begin(), random_numbers.end(), comp);
  auto min_idx = std::distance(random_numbers.begin(), min_iterator);
  TSDFEntry min = random_numbers[min_idx];

  std::vector<TSDFEntry> buffer = { TSDFEntry(std::numeric_limits<int16_t>::max(), 0) };
  auto min_cuda = cuda::atomic_tsdf_min(buffer, random_numbers);
  EXPECT_EQ(min.value(), min_cuda.value());
}

void test_rmagine()
{
  rm::Pointi p = {1, 2, 3};
  Eigen::Vector3i pe = *reinterpret_cast<Eigen::Vector3i*>(&p);
  EXPECT_EQ(p.x, pe.x());
  EXPECT_EQ(p.y, pe.y());
  EXPECT_EQ(p.z, pe.z());
  auto p2 = *reinterpret_cast<rm::Pointi*>(&pe);
  EXPECT_EQ(p2.x, pe.x());
  EXPECT_EQ(p2.y, pe.y());
  EXPECT_EQ(p2.z, pe.z());
}

int main(int argc, char** argv)
{
  RUN(test_cuda_transform_point);
  RUN(test_cuda_jacobi_device);
  RUN(test_cuda_cross);
  RUN(test_cuda_sum);
  RUN(test_cuda_sum_split);
  RUN(test_cuda_cov);
  RUN(test_cuda_cov6d);
  RUN(test_cuda_h_g_e_reduction);
  RUN(test_cuda_map_adapter);
  RUN(test_cuda_map_bounds);
  RUN(test_cuda_map_write);
  RUN(test_cuda_calc_jacobis);
  RUN(test_cuda_data_sizes);
  RUN(test_cuda_atomic_min);
  RUN(test_cuda_atomic_tsdf_min);
  RUN(test_rmagine);
  return 0;
}
