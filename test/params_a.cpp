#include <gtest/gtest.h>
#include "params/params.h"
#include <map/hdf5_local_map.h>
#include <map/hdf5_constants.h>

TEST(warpsense, parameters_a_warp)
{
  ros::NodeHandle n;
  Params params(n);

  int max_iterations = 300;
  float it_weight_gradient = 1;
  std::string lidar_topic("/os1_cloud_node/pointsee");
  std::string imu_topic("/os1_cloud_node/imuee");
  std::string link("os1_sensoree");
  float epsilon = 0.05;

  EXPECT_EQ(max_iterations, params.registration.max_iterations);
  EXPECT_FLOAT_EQ(it_weight_gradient, params.registration.it_weight_gradient);
  EXPECT_EQ(lidar_topic, params.registration.lidar_topic);
  EXPECT_EQ(imu_topic, params.registration.imu_topic);
  EXPECT_EQ(link, params.registration.link);
  EXPECT_FLOAT_EQ(epsilon, params.registration.epsilon);

  float max_distance = 3;
  int map_resolution = 1000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  int map_size_x = 20 * 1000 / map_resolution;
  int map_size_y = 20 * 1000 / map_resolution;
  int map_size_z = 20 * 1000 / map_resolution;

  EXPECT_FLOAT_EQ(max_distance, params.map.max_distance);
  EXPECT_FLOAT_EQ(map_resolution, params.map.resolution);
  EXPECT_EQ(tau, params.map.tau);
  EXPECT_EQ(map_size_x, params.map.size.x());
  EXPECT_EQ(map_size_y, params.map.size.y());
  EXPECT_EQ(map_size_z, params.map.size.z());
  EXPECT_EQ(params.map.initial_weight, 0.0f);
  std::string filename(params.map.filename.c_str());
  EXPECT_EQ(filename, "/tmp/test_params.h5");

  {
    std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
    std::shared_ptr<HDF5LocalMap> local_map_ptr_;
    global_map_ptr_.reset(new HDF5GlobalMap(params.map));
    local_map_ptr_.reset(new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), global_map_ptr_));
    local_map_ptr_->write_back();
  }

  HighFive::File f(params.map.filename.c_str(), HighFive::File::ReadOnly); // TODO: Path and name as command line input
  HighFive::Group g = f.getGroup(hdf5_constants::MAP_GROUP_NAME);

  // Test metadata
  EXPECT_TRUE(f.exist(hdf5_constants::MAP_GROUP_NAME));
  g = f.getGroup(hdf5_constants::MAP_GROUP_NAME);


  float max_distance_test;
  int map_resolution_test;
  int tau_test;
  int max_weight_test;
  int map_size_x_test;
  int map_size_y_test;
  int map_size_z_test;

  g.getAttribute("tau").read(tau_test);
  EXPECT_EQ(tau_test, tau);
  EXPECT_EQ(tau_test, params.map.tau);

  g.getAttribute("map_resolution").read(map_resolution_test);
  EXPECT_EQ(map_resolution_test, map_resolution);
  EXPECT_EQ(map_resolution_test, params.map.resolution);

  g.getAttribute("max_distance").read(max_distance_test);
  EXPECT_FLOAT_EQ(max_distance_test, max_distance);
  EXPECT_FLOAT_EQ(max_distance_test, params.map.max_distance);

  g.getAttribute("max_weight").read(max_weight_test);
  EXPECT_EQ(max_weight_test, max_weight);
  EXPECT_EQ(max_weight_test, params.map.max_weight);

  g.getAttribute("map_size_x").read(map_size_x_test);
  EXPECT_FLOAT_EQ(map_size_x_test, map_size_x);
  EXPECT_FLOAT_EQ(map_size_x_test, params.map.size.x());

  g.getAttribute("map_size_y").read(map_size_y_test);
  EXPECT_FLOAT_EQ(map_size_y_test, params.map.size.y());

  g.getAttribute("map_size_z").read(map_size_z_test);
  EXPECT_FLOAT_EQ(map_size_z_test, params.map.size.z());
}

TEST(warpsense, parameters_a_feat)
{
  ros::NodeHandle n;
  Params params(n);

  float max_distance = 3;
  int map_resolution = 1000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  int map_size_x = 20 * 1000 / map_resolution;
  int map_size_y = 20 * 1000 / map_resolution;
  int map_size_z = 20 * 1000 / map_resolution;

  EXPECT_FLOAT_EQ(max_distance, params.map.max_distance);
  EXPECT_FLOAT_EQ(map_resolution, params.map.resolution);
  EXPECT_EQ(tau, params.map.tau);
  EXPECT_EQ(map_size_x, params.map.size.x());
  EXPECT_EQ(map_size_y, params.map.size.y());
  EXPECT_EQ(map_size_z, params.map.size.z());
  EXPECT_EQ(params.map.initial_weight, 0.0f);
  std::string filename(params.map.filename.c_str());
  EXPECT_EQ(filename, "/tmp/test_params.h5");

  {
    std::shared_ptr<HDF5GlobalMap> global_map_ptr_;
    std::shared_ptr<HDF5LocalMap> local_map_ptr_;
    global_map_ptr_.reset(new HDF5GlobalMap(params.map));
    local_map_ptr_.reset(new HDF5LocalMap(params.map.size.x(), params.map.size.y(), params.map.size.z(), global_map_ptr_));
    local_map_ptr_->write_back();
  }

  HighFive::File f(params.map.filename.c_str(), HighFive::File::ReadOnly); // TODO: Path and name as command line input
  HighFive::Group g = f.getGroup(hdf5_constants::MAP_GROUP_NAME);

  // Test metadata
  EXPECT_TRUE(f.exist(hdf5_constants::MAP_GROUP_NAME));
  g = f.getGroup(hdf5_constants::MAP_GROUP_NAME);
  

  float max_distance_test;
  int map_resolution_test;
  int tau_test;
  int max_weight_test;
  int map_size_x_test;
  int map_size_y_test;
  int map_size_z_test;

  g.getAttribute("tau").read(tau_test);
  EXPECT_EQ(tau_test, tau);
  EXPECT_EQ(tau_test, params.map.tau);

  g.getAttribute("map_resolution").read(map_resolution_test);
  EXPECT_EQ(map_resolution_test, map_resolution);
  EXPECT_EQ(map_resolution_test, params.map.resolution);

  g.getAttribute("max_distance").read(max_distance_test);
  EXPECT_FLOAT_EQ(max_distance_test, max_distance);
  EXPECT_FLOAT_EQ(max_distance_test, params.map.max_distance);

  g.getAttribute("max_weight").read(max_weight_test);
  EXPECT_EQ(max_weight_test, max_weight);
  EXPECT_EQ(max_weight_test, params.map.max_weight);

  g.getAttribute("map_size_x").read(map_size_x_test);
  EXPECT_FLOAT_EQ(map_size_x_test, map_size_x);
  EXPECT_FLOAT_EQ(map_size_x_test, params.map.size.x());

  g.getAttribute("map_size_y").read(map_size_y_test);
  EXPECT_FLOAT_EQ(map_size_y_test, params.map.size.y());

  g.getAttribute("map_size_z").read(map_size_z_test);
  EXPECT_FLOAT_EQ(map_size_z_test, params.map.size.z());
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "params_a");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}