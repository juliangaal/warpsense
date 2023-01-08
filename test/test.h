#pragma once

#include "warpsense/cuda/device_map.h"

// DO NOT REMOVE NEXT TO LINES (Unless you want to live blissfully), ASSERTS *WILL BE IGNORED*
#undef NDEBUG
#include <assert.h>
#include <vector>

// g_test swallows any output to stdout
// on cpu, use std::cerr, on GPU...not possible
// therefore redefine the code of gtest
#define EXPECT_EQ(A, B) assert((A) == (B))
#define EXPECT_EQ_FLOAT_E(A, B, E) assert(abs((A) - (B)) < (E))
#define EXPECT_EQ_FLOAT(A, B) assert(abs((A) - (B)) < 0.001)
#define EXPECT_TRUE(A) assert((A))
#define EXPECT(A) assert((A))
#define EXPECT_FALSE(A) assert(!(A))

namespace cuda
{

void test_map_adapter(const cuda::DeviceMap &map, int tau, int weight_epsilon);

void test_map_bounds(const cuda::DeviceMap &map, bool empty);

void test_map_write(cuda::DeviceMap &map, const std::vector<rmagine::Pointi> &gpu_access, const TSDFEntry &entry);

void test_data_sizes();

} // end namespace cuda