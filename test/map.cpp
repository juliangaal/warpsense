#include <iostream>
#include <gtest/gtest.h>
#include <util/util.h>
#include <cpu/update_tsdf.h>
#include <warpsense/test/common.h>
#include <warpsense/cuda/device_map.h>
#include <map/hdf5_constants.h>

TEST(warpsense, tsdf_write)
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
}

TEST(warpsense, tsdf_read)
{
  HighFive::File f("/tmp/test.h5", HighFive::File::ReadOnly); // TODO: Path and name as command line input
  HighFive::Group g = f.getGroup(hdf5_constants::MAP_GROUP_NAME);

  // tsdf parameters
  float max_distance = 3;
  int map_resolution = 1000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  float map_size_x = 20 * 1000 / map_resolution;
  float map_size_y = 20 * 1000 / map_resolution;
  float map_size_z = 20 * 1000 / map_resolution;

  size_t pos = 0;
  std::string token;
  std::string delimiter = "_";

  /// Side length of the cube-shaped chunks (2^CHUNK_SHIFT).
  constexpr int CHUNK_SIZE = 64;

  std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
  std::shared_ptr<HDF5LocalMap> local_map_ptr_;
  global_map_ptr_.reset(new HDF5GlobalMap("/tmp/test2.h5", tau, 0.0));
  local_map_ptr_.reset(new HDF5LocalMap(map_size_x, map_size_y, map_size_z, global_map_ptr_));

  for (auto tag : g.listObjectNames())
  {
    auto orig_tag = tag;
    std::vector<int> chunk_pos;
    while ((pos = tag.find(delimiter)) != std::string::npos)
    {
      token = tag.substr(0, pos);
      chunk_pos.push_back(std::stoi(token));
      tag.erase(0, pos + delimiter.length());
    }

    chunk_pos.push_back(std::stoi(tag));

    HighFive::DataSet d = g.getDataSet(orig_tag);
    std::vector<TSDFEntry::RawType> chunk_data;
    d.read(chunk_data);

    for (int i = 0; i < CHUNK_SIZE; i++)
    {
      for (int j = 0; j < CHUNK_SIZE; j++)
      {
        for (int k = 0; k < CHUNK_SIZE; k++)
        {
          auto entry = TSDFEntry(chunk_data[CHUNK_SIZE * CHUNK_SIZE * i + CHUNK_SIZE * j + k]);

          auto tsdf_value = entry.value();
          auto weight = entry.weight();
          int x = CHUNK_SIZE * chunk_pos[0] + i;
          int y = CHUNK_SIZE * chunk_pos[1] + j;
          int z = CHUNK_SIZE * chunk_pos[2] + k;

          // Only touched cells are considered
          if (weight > 0)
          {
            local_map_ptr_->value(x, y, z) = entry;
          }
        }
      }
    }
  }

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


  Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();

  // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  double theta = M_PI/4; // The angle of rotation in radians
  rotation(0,0) = std::cos (theta);
  rotation(0,1) = -sin(theta);
  rotation(1,0) = sin (theta);
  rotation(1,1) = std::cos (theta);
  Eigen::Vector3d translation = Eigen::Vector3d::Ones();
  auto quaternion = Eigen::Quaterniond(rotation);

  // Test Poses
  g = f.getGroup(hdf5_constants::POSES_GROUP_NAME);
  for (int i = 0; i < g.listObjectNames().size(); ++i)
  {
    auto pose_g = g.getGroup(std::string(hdf5_constants::POSES_GROUP_NAME) + "/" + std::to_string(i));
    auto d = pose_g.getDataSet(hdf5_constants::POSE_DATASET_NAME);
    std::vector<float> data(hdf5_constants::POSE_DATASET_SIZE);
    d.read(data);

    target_pose.rotate(rotation);
    target_pose.pretranslate(translation);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
    pose.rotate(Eigen::Quaterniond(data[6], data[3], data[4], data[5]));

    EXPECT_EQ(pose.translation(), target_pose.translation());

    auto target_quaternion = Eigen::Quaterniond(target_pose.rotation());
    auto quaternion = Eigen::Quaterniond(pose.rotation());
    EXPECT_TRUE(approx(quaternion.x(), quaternion.x(), 0.01));
    EXPECT_TRUE(approx(quaternion.y(), quaternion.y(), 0.01));
    EXPECT_TRUE(approx(quaternion.z(), quaternion.z(), 0.01));
    EXPECT_TRUE(approx(quaternion.w(), quaternion.w(), 0.01));
  }
}

TEST(warpsense, map_raw)
{
  std::shared_ptr<HDF5GlobalMap> gm_ptr = std::make_shared<HDF5GlobalMap>("/tmp/test4.h5", DEFAULT_VALUE, DEFAULT_WEIGHT);
  HDF5LocalMap localMap{5, 5, 5, gm_ptr};

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
  TSDFEntry p0(0, 0);
  TSDFEntry p1(1, 1);
  TSDFEntry p2(2, 1);
  TSDFEntry p3(3, 2);
  TSDFEntry p4(4, 3);
  TSDFEntry p5(5, 5);
  localMap.value(-2, 2, 0) = p0;
  localMap.value(-1, 2, 0) = p1;
  localMap.value(-2, 1, 0) = p2;
  localMap.value(-1, 1, 0) = p3;
  localMap.value(-2, 0, 0) = p4;
  localMap.value(-1, 0, 0) = p5;

  // test getter
  EXPECT_TRUE(localMap.get_pos() == Eigen::Vector3i(0, 0, 0));
  EXPECT_TRUE(localMap.get_size() == Eigen::Vector3i(5, 5, 5));
  EXPECT_TRUE(localMap.get_offset() == Eigen::Vector3i(2, 2, 2));

  // test in_bounds
  EXPECT_TRUE(localMap.in_bounds(0, 2, -2));
  EXPECT_TRUE(!localMap.in_bounds(22, 0, 0));
  // test default values
  EXPECT_TRUE(localMap.value(0, 0, 0).value() == DEFAULT_VALUE);
  EXPECT_TRUE(localMap.value(0, 0, 0).weight() == DEFAULT_WEIGHT);
  // test value access
  EXPECT_TRUE(localMap.value(-1, 2, 0).value() == 1);
  EXPECT_TRUE(localMap.value(-1, 2, 0).weight() == 1);

  // ==================== shift ====================
  // shift so that the chunk gets unloaded
  // Each shift can only cover an area of one Map size
  localMap.shift(Eigen::Vector3i(5, 0, 0));
  localMap.shift(Eigen::Vector3i(10, 0, 0));
  localMap.shift(Eigen::Vector3i(15, 0, 0));
  localMap.shift(Eigen::Vector3i(20, 0, 0));
  localMap.shift(Eigen::Vector3i(24, 0, 0));

  EXPECT_TRUE(localMap.get_pos() == Eigen::Vector3i(24, 0, 0));
  EXPECT_TRUE(localMap.get_size() == Eigen::Vector3i(5, 5, 5));
  EXPECT_TRUE(localMap.get_offset() == Eigen::Vector3i(26 % 5, 2, 2));

  // test in_bounds
  EXPECT_TRUE(!localMap.in_bounds(0, 2, -2));
  EXPECT_TRUE(localMap.in_bounds(22, 0, 0));
  // test values
  EXPECT_TRUE(localMap.value(24, 0, 0).value() == DEFAULT_VALUE);
  EXPECT_TRUE(localMap.value(24, 0, 0).weight() == DEFAULT_WEIGHT);

  // ==================== shift directions ====================
  localMap.value(24, 0, 0) = TSDFEntry(24, 0);

  localMap.shift(Eigen::Vector3i(24, 5, 0));
  localMap.value(24, 5, 0) = TSDFEntry(24, 5);

  localMap.shift(Eigen::Vector3i(19, 5, 0));
  localMap.value(19, 5, 0) = TSDFEntry(19, 5);

  localMap.shift(Eigen::Vector3i(19, 0, 0));
  localMap.value(19, 0, 0) = TSDFEntry(19, 0);

  localMap.shift(Eigen::Vector3i(24, 0, 0));
  EXPECT_TRUE(localMap.value(24, 0, 0).value() == 24);
  EXPECT_TRUE(localMap.value(24, 0, 0).weight() == 0);

  localMap.shift(Eigen::Vector3i(19, 0, 0));
  EXPECT_TRUE(localMap.value(19, 0, 0).value() == 19);
  EXPECT_TRUE(localMap.value(19, 0, 0).weight() == 0);
  localMap.shift(Eigen::Vector3i(24, 5, 0));
  EXPECT_TRUE(localMap.value(24, 5, 0).value() == 24);
  EXPECT_TRUE(localMap.value(24, 5, 0).weight() == 5);
  localMap.shift(Eigen::Vector3i(19, 5, 0));
  EXPECT_TRUE(localMap.value(19, 5, 0).value() == 19);
  EXPECT_TRUE(localMap.value(19, 5, 0).weight() == 5);
  localMap.shift(Eigen::Vector3i(24, 0, 0));
  EXPECT_TRUE(localMap.value(24, 0, 0).value() == 24);
  EXPECT_TRUE(localMap.value(24, 0, 0).weight() == 0);

  // ==================== shift back ====================
  localMap.shift(Eigen::Vector3i(19, 0, 0));
  localMap.shift(Eigen::Vector3i(14, 0, 0));
  localMap.shift(Eigen::Vector3i(9, 0, 0));
  localMap.shift(Eigen::Vector3i(4, 0, 0));
  localMap.shift(Eigen::Vector3i(0, 0, 0));

  // test return correct
  EXPECT_TRUE(localMap.get_pos() == Eigen::Vector3i(0, 0, 0));
  EXPECT_TRUE(localMap.get_size() == Eigen::Vector3i(5, 5, 5));
  EXPECT_TRUE(localMap.get_offset() == Eigen::Vector3i(2, 2, 2));

  // test in_bounds
  EXPECT_TRUE(localMap.in_bounds(0, 2, -2));
  EXPECT_TRUE(!localMap.in_bounds(22, 0, 0));

  EXPECT_TRUE(localMap.value(0, 0, 0).value() == DEFAULT_VALUE);
  EXPECT_TRUE(localMap.value(0, 0, 0).weight() == DEFAULT_WEIGHT);
  EXPECT_TRUE(localMap.value(-1, 2, 0).value() == 1);
  EXPECT_TRUE(localMap.value(-1, 2, 0).weight() == 1);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "map_test_node");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}