#include <gtest/gtest.h>
#include "params/params.h"

TEST(warpsense, parameters_b)
{
  ros::NodeHandle n;
  Params params(n);

  float max_distance = 4;
  int map_resolution = 4000;
  int tau = max_distance * 1000;
  int weight_epsilon = tau / 10;
  int max_weight = 10 * WEIGHT_RESOLUTION;
  float map_size_x = 20 * 1000 / map_resolution;
  float map_size_y = 20 * 1000 / map_resolution;
  float map_size_z = 20 * 1000 / map_resolution;

  EXPECT_FLOAT_EQ(max_distance, params.map.max_distance);
  EXPECT_EQ(map_resolution, params.map.resolution);
  EXPECT_EQ(tau, params.map.tau);
  EXPECT_EQ(map_size_x, params.map.size.x());
  EXPECT_EQ(map_size_y, params.map.size.y());
  EXPECT_EQ(map_size_z, params.map.size.z());
  EXPECT_EQ(params.map.initial_weight, 0.0f);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "params_b");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}