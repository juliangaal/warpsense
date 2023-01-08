#include "warpsense/math/math.h"
#include "warpsense/cuda/common.cuh"
#include "test.h"
#include "warpsense/test/common.h"
#include "warpsense/cuda/device_map.h"
#include "warpsense/cuda/device_map_wrapper.h"

#include <vector>

namespace rm = rmagine;

namespace cuda
{

__global__ void test_map_bounds_krnl(const cuda::DeviceMap* map, bool empty)
{
  if (empty)
  {
    auto n = map->get_size()->prod();
    for (int i = 0; i < n; ++i)
    {
      EXPECT_TRUE(map->data_[i].value() == DEFAULT_VALUE);
      EXPECT_TRUE(map->data_[i].weight() == DEFAULT_WEIGHT);
    }
  }
  else
  {
    EXPECT_TRUE(map->in_bounds(0, 2, -2));
    EXPECT_TRUE(!map->in_bounds(22, 0, 0));
    // test default values
    EXPECT_TRUE(map->value_unchecked(0, 0, 0).value() == DEFAULT_VALUE);
    EXPECT_TRUE(map->value_unchecked(0, 0, 0).weight() == DEFAULT_WEIGHT);
    // test value access
    EXPECT_TRUE(map->value_unchecked(-1, 2, 0).value() == 1);
    EXPECT_TRUE(map->value_unchecked(-1, 2, 0).weight() == 1);
  }
}

__global__ void test_map_write_krnl(cuda::DeviceMap* map, const rm::Pointi* gpu_access, size_t N, const TSDFEntry* entry)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    map->value_unchecked(gpu_access[idx]) = *entry;
  }
}

__global__ void test_data_sizes_krnl()

{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0)
  {
    EXPECT_EQ(sizeof(long), 8);
    EXPECT_EQ(sizeof(int), 4);
    EXPECT_EQ(sizeof(float), 4);
    EXPECT_EQ(sizeof(TSDFEntry), sizeof(int));
    EXPECT_EQ(sizeof(TSDFEntry::RawType), sizeof(int));
    EXPECT_EQ(sizeof(int), sizeof(uint32_t));
  }
}

__global__ void
test_map_adapter_krnl(const cuda::DeviceMap *map,
                      int tau,
                      int weight_epsilon,
                      const rmagine::Pointi *size,
                      const rmagine::Pointi *pos,
                      const rmagine::Pointi *offset)
{
  // size
  EXPECT_EQ(map->size_->x, 21);
  EXPECT_EQ(map->size_->y, 21);
  EXPECT_EQ(map->size_->z, 21);
  // offset
  EXPECT_EQ(map->offset_->x, 10);
  EXPECT_EQ(map->offset_->y, 10);
  EXPECT_EQ(map->offset_->z, 10);
  // data
  // TODO EXPECT_EQ();
  // pos
  EXPECT_EQ(map->pos_->x, 0);
  EXPECT_EQ(map->pos_->y, 0);
  EXPECT_EQ(map->pos_->z, 0);

  // values
  auto val = map->value_unchecked(1, 0, 0).value();
  auto weight = map->value_unchecked(1, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(2, 0, 0).value();
  weight = map->value_unchecked(2, 0, 0).weight();
  EXPECT_EQ(val, tau);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(3, 0, 0).value();
  weight = map->value_unchecked(3, 0, 0).weight();
  EXPECT_EQ(val, 2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(4, 0, 0).value();
  weight = map->value_unchecked(4, 0, 0).weight();
  EXPECT_EQ(val,1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(5, 0, 0).value();
  weight = map->value_unchecked(5, 0, 0).weight();
  EXPECT_EQ(val, 0);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(6, 0, 0).value();
  weight = map->value_unchecked(6, 0, 0).weight();
  EXPECT_EQ(val, -1000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(7, 0, 0).value();
  weight = map->value_unchecked(7, 0, 0).weight();
  EXPECT_EQ(val, -2000);
  EXPECT_EQ(weight, calc_weight(val, tau, weight_epsilon));

  val = map->value_unchecked(8, 0, 0).value();
  weight = map->value_unchecked(8, 0, 0).weight();
  EXPECT_EQ(val, 3000); // default val, weight == 0
  EXPECT_EQ(weight, 0);
}

void test_map_adapter(const cuda::DeviceMap &map, int tau, int weight_epsilon)
{
  size_t nelements = map.size_->prod();
  
  cuda::DeviceMap map_inner;
  cudaMalloc(&map_inner.size_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.offset_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.data_, nelements * sizeof(TSDFEntry));
  cudaMalloc(&map_inner.pos_, sizeof(rm::Pointi));
  cudaMemcpy(map_inner.size_, map.size_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.offset_, map.offset_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.data_, map.data_, nelements * sizeof(TSDFEntry), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.pos_, map.pos_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);

  cuda::DeviceMap* map_dev;
  CHECK(cudaMalloc((void **) &map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMemcpy(map_dev, &map_inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));

  test_map_adapter_krnl<<<1, 1>>>(map_dev, tau, weight_epsilon, map.size_, map.pos_, map.offset_);

  CHECK(cudaFree(map_inner.size_));
  CHECK(cudaFree(map_inner.offset_));
  CHECK(cudaFree(map_inner.data_));
  CHECK(cudaFree(map_inner.pos_));
  CHECK(cudaFree(map_dev));
  CHECK(cudaDeviceReset());
}

void test_map_bounds(const cuda::DeviceMap &map, bool empty)
{
  size_t nelements = map.size_->prod();

  cuda::DeviceMap map_inner;
  cudaMalloc(&map_inner.size_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.offset_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.data_, nelements * sizeof(TSDFEntry));
  cudaMalloc(&map_inner.pos_, sizeof(rm::Pointi));
  cudaMemcpy(map_inner.size_, map.size_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.offset_, map.offset_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.data_, map.data_, nelements * sizeof(TSDFEntry), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.pos_, map.pos_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);

  cuda::DeviceMap* map_dev;
  CHECK(cudaMalloc((void **) &map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMemcpy(map_dev, &map_inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));

  test_map_bounds_krnl<<<1, 1>>>(map_dev, empty);

  CHECK(cudaFree(map_inner.size_));
  CHECK(cudaFree(map_inner.offset_));
  CHECK(cudaFree(map_inner.data_));
  CHECK(cudaFree(map_inner.pos_));
  CHECK(cudaFree(map_dev));
  CHECK(cudaDeviceReset());
}

void test_map_write(cuda::DeviceMap &map, const std::vector<rmagine::Pointi> &gpu_access, const TSDFEntry &entry)
{
  size_t nelements = map.size_->prod();
  size_t n_gpu_access = gpu_access.size();

  cuda::DeviceMap map_inner;
  cudaMalloc(&map_inner.size_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.offset_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.data_, nelements * sizeof(TSDFEntry));
  cudaMalloc(&map_inner.pos_, sizeof(rm::Pointi));
  cudaMemcpy(map_inner.size_, map.size_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.offset_, map.offset_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.data_, map.data_, nelements * sizeof(TSDFEntry), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.pos_, map.pos_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);

  cuda::DeviceMap* map_dev;
  CHECK(cudaMalloc((void **) &map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMemcpy(map_dev, &map_inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));

  rmagine::Pointi* gpu_access_dev;
  CHECK(cudaMalloc((void **) &gpu_access_dev, sizeof(rmagine::Pointi) * n_gpu_access));
  CHECK(cudaMemcpy(gpu_access_dev, gpu_access.data(), sizeof(rmagine::Pointi) * n_gpu_access, cudaMemcpyHostToDevice));

  TSDFEntry* entry_dev;
  CHECK(cudaMalloc((void **) &entry_dev, sizeof(TSDFEntry)));
  CHECK(cudaMemcpy(entry_dev, &entry, sizeof(TSDFEntry), cudaMemcpyHostToDevice));

  test_map_write_krnl<<<1, map.size_->prod()>>>(map_dev, gpu_access_dev, n_gpu_access, entry_dev);

  CHECK(cudaMemcpy(&map_inner, map_dev, sizeof(cuda::DeviceMap), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(map.size_, map_inner.size_, sizeof(rm::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(map.offset_, map_inner.offset_, sizeof(rm::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(map.data_, map_inner.data_, nelements * sizeof(TSDFEntry), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(map.pos_, map_inner.pos_, sizeof(rm::Pointi), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(map_inner.size_));
  CHECK(cudaFree(map_inner.offset_));
  CHECK(cudaFree(map_inner.data_));
  CHECK(cudaFree(map_inner.pos_));
  CHECK(cudaFree(map_dev));
  CHECK(cudaFree(gpu_access_dev));
  CHECK(cudaFree(entry_dev));
  CHECK(cudaDeviceReset());
}

void test_data_sizes()
{
  test_data_sizes_krnl<<<1, 1>>>();
}

} // end namespace cuda
