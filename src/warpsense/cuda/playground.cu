#include <warpsense/math/math.h>
#include <warpsense/cuda/util.h>
#include <warpsense/cuda/common.cuh>
#include "warpsense/consts.h"
#include "warpsense/cuda/playground.h"
#include <warpsense/test/common.h>
#include <warpsense/cuda/device_map.h>
#include <vector>

namespace rm = rmagine;

__device__ const int MAP_RESOLUTION = 64;

namespace cuda
{

__forceinline__ __device__ uint16_t atomic_min(int* address,
                                               int new_val) {
  int old = *address, assumed;

  if(old <= new_val)
  {
    return old;
  }

  do
  {
    // ich denke der aktuelle Wert ist der alte
    assumed = old;
    if (assumed <= new_val)
    {
      // egal, nix zu tun ODER old aus atomicCAS aus anderem Thread ist kleiner als new_val
      break;
    }

    // Nur, wenn adresse noch das selbe ist, weiss ich, dass kein anderer Thread einen
    // kleineren Wert in die adresse geschrieben hat

    // Wenn kein anderer Thread einen kleinen Wert gefunden hat, schreibt "unser" thread
    // einen auf jeden fall kleineren Wert in adresse
    // -> old und assumed bleibt gleich UND die Schleife bricht ab

    // Wenn **ein** anderer Thread einen kleinen Wert gefunden hat, dann sind adresse und assumed ungleich
    // -> old enthaelt den neuen kleinsten Wert aus einem anderen Thread
    old = atomicCAS(address, assumed, new_val);
  } while(assumed != old);

  return old;
}

__forceinline__ __device__ void write_min(int* __restrict__ buffer,
                                           unsigned int coord,
                                           int value)
{
  atomic_min(buffer + coord, value);
}

__global__ void cu_write_min(int* __restrict__ buffer, int* values, size_t size)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < size)
  {
    write_min(buffer, 0, values[ix]);
  }
}

__global__ void cu_write_tsdf_min(TSDFEntry* buffer, TSDFEntry* values, size_t size)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < size)
  {
    write_tsdf_min(buffer, 0, values[ix]);
  }
}

__global__ void cu_transform_point_krnl(const rm::Pointi *input, const rm::Matrix4x4i *mat, rm::Pointi *out)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < 1)
  {
    cu_transform_point(*input, *mat, *out);
  }
}

__global__ void cu_jacobi_2_h_krnl(const rm::Point6l *jacobi, rm::Matrix6x6l *h)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < 1)
  {
    cu_jacobi_2_h(jacobi, h);
  }
}

__global__ void cu_cross_krnl(const rm::Vector3i *a, const rm::Vector3i *b, rm::Vector3i *res)
{
  auto ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < 1)
  {
    (*res) = (*a).cross(*b);
  }
}

template <unsigned int blockSize, typename T>
__global__ void sum_krnl(
    const T* data,
    T* res,
    unsigned int N)
{
  __shared__ T sdata[blockSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int globId = N * blockIdx.x + threadIdx.x;
  const unsigned int rows = (N + blockSize - 1) / blockSize;

  sdata[tid] *= 0.0;
  for(unsigned int i=0; i < rows; i++)
  {
    if(tid + blockSize * i < N)
    {
      sdata[threadIdx.x] += data[globId + blockSize * i];
    }
  }
  __syncthreads();

  for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid < blockSize / 2 && tid < 32)
  {
    warpReduce<blockSize>(sdata, tid);
  }

  if(tid == 0)
  {
    res[blockIdx.x] = sdata[0];
  }
}

template <unsigned int blockSize, typename S, typename T>
__global__ void
h_g_e_reduction_krnl(const rm::Vector6<S> *jacobis,
                     const TSDFEntry::ValueType *values,
                     const bool *mask,
                     rm::Matrix6x6<S> *res_h,
                     rm::Vector6<S> *res_g,
                     T *res_e,
                     T *res_c,
                     unsigned int N)
{
  __shared__ rm::Matrix6x6<S> s_h[blockSize];
  __shared__ rm::Vector6<S> s_g[blockSize];
  __shared__ T s_e[blockSize];
  __shared__ T s_c[blockSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int globId = N * blockIdx.x + threadIdx.x;
  const unsigned int rows = (N + blockSize - 1) / blockSize;

  s_h[tid].setZeros();
  s_g[tid].setZeros();
  s_e[tid] = 0;
  s_c[tid] = 0;

  for(unsigned int i=0; i < rows; i++)
  {
    if(tid + blockSize * i < N && mask[globId + blockSize * i] == true)
    {
      const rm::Vector6<S>& jacobi = jacobis[globId + blockSize * i];
      const TSDFEntry::ValueType& current_value = values[globId + blockSize * i];

      // update h
      s_h[tid].at(0, 0) += jacobi[0] * jacobi[0];
      s_h[tid].at(1, 0) += jacobi[0] * jacobi[1];
      s_h[tid].at(2, 0) += jacobi[0] * jacobi[2];
      s_h[tid].at(3, 0) += jacobi[0] * jacobi[3];
      s_h[tid].at(4, 0) += jacobi[0] * jacobi[4];
      s_h[tid].at(5, 0) += jacobi[0] * jacobi[5];
      s_h[tid].at(0, 1) += jacobi[1] * jacobi[0];
      s_h[tid].at(1, 1) += jacobi[1] * jacobi[1];
      s_h[tid].at(2, 1) += jacobi[1] * jacobi[2];
      s_h[tid].at(3, 1) += jacobi[1] * jacobi[3];
      s_h[tid].at(4, 1) += jacobi[1] * jacobi[4];
      s_h[tid].at(5, 1) += jacobi[1] * jacobi[5];
      s_h[tid].at(0, 2) += jacobi[2] * jacobi[0];
      s_h[tid].at(1, 2) += jacobi[2] * jacobi[1];
      s_h[tid].at(2, 2) += jacobi[2] * jacobi[2];
      s_h[tid].at(3, 2) += jacobi[2] * jacobi[3];
      s_h[tid].at(4, 2) += jacobi[2] * jacobi[4];
      s_h[tid].at(5, 2) += jacobi[2] * jacobi[5];
      s_h[tid].at(0, 3) += jacobi[3] * jacobi[0];
      s_h[tid].at(1, 3) += jacobi[3] * jacobi[1];
      s_h[tid].at(2, 3) += jacobi[3] * jacobi[2];
      s_h[tid].at(3, 3) += jacobi[3] * jacobi[3];
      s_h[tid].at(4, 3) += jacobi[3] * jacobi[4];
      s_h[tid].at(5, 3) += jacobi[3] * jacobi[5];
      s_h[tid].at(0, 4) += jacobi[4] * jacobi[0];
      s_h[tid].at(1, 4) += jacobi[4] * jacobi[1];
      s_h[tid].at(2, 4) += jacobi[4] * jacobi[2];
      s_h[tid].at(3, 4) += jacobi[4] * jacobi[3];
      s_h[tid].at(4, 4) += jacobi[4] * jacobi[4];
      s_h[tid].at(5, 4) += jacobi[4] * jacobi[5];
      s_h[tid].at(0, 5) += jacobi[5] * jacobi[0];
      s_h[tid].at(1, 5) += jacobi[5] * jacobi[1];
      s_h[tid].at(2, 5) += jacobi[5] * jacobi[2];
      s_h[tid].at(3, 5) += jacobi[5] * jacobi[3];
      s_h[tid].at(4, 5) += jacobi[5] * jacobi[4];
      s_h[tid].at(5, 5) += jacobi[5] * jacobi[5];

      // update g
      s_g[tid].at(0) += jacobi[0] * current_value;
      s_g[tid].at(1) += jacobi[1] * current_value;
      s_g[tid].at(2) += jacobi[2] * current_value;
      s_g[tid].at(3) += jacobi[3] * current_value;
      s_g[tid].at(4) += jacobi[4] * current_value;
      s_g[tid].at(5) += jacobi[5] * current_value;

      // update error
      s_e[tid] += abs(current_value);

      // update counter
      s_c[tid] += 1;
    }
  }
  __syncthreads();

  for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
  {
    if(tid < s)
    {
      s_h[tid] += s_h[tid + s];
      s_g[tid] += s_g[tid + s];
      s_e[tid] += s_e[tid + s];
      s_c[tid] += s_c[tid + s];
    }
    __syncthreads();
  }

  if(tid < blockSize / 2 && tid < 32)
  {
    warpReduce<blockSize>(s_h, tid);
    warpReduce<blockSize>(s_g, tid);
    warpReduce<blockSize>(s_e, tid);
    warpReduce<blockSize>(s_c, tid);
  }

  if(tid == 0)
  {
    res_h[blockIdx.x] = s_h[0];
    res_g[blockIdx.x] = s_g[0];
    res_e[blockIdx.x] = s_e[0];
    res_c[blockIdx.x] = s_c[0];
  }
}

template <unsigned int blockSize, typename T>
__global__ void cov6d_krnl(
    const rm::Vector6<T>* v1,
    const rm::Vector6<T>* v2,
    rm::Matrix6x6<T>* res,
    unsigned int N)
{
  __shared__ rm::Matrix6x6<T> sdata[blockSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int globId = N * blockIdx.x + threadIdx.x;
  const unsigned int rows = (N + blockSize - 1) / blockSize;

  sdata[tid].setZeros();
  for(unsigned int i=0; i < rows; i++)
  {
    if(tid + blockSize * i < N)
    {
      const rm::Vector6<T>& a = v1[globId + blockSize * i];
      const rm::Vector6<T>& b = v2[globId + blockSize * i];
      sdata[tid].at(0,0) += a[0] * b[0];
      sdata[tid].at(1,0) += a[0] * b[1];
      sdata[tid].at(2,0) += a[0] * b[2];
      sdata[tid].at(3,0) += a[0] * b[3];
      sdata[tid].at(4,0) += a[0] * b[4];
      sdata[tid].at(5,0) += a[0] * b[5];
      sdata[tid].at(0,1) += a[1] * b[0];
      sdata[tid].at(1,1) += a[1] * b[1];
      sdata[tid].at(2,1) += a[1] * b[2];
      sdata[tid].at(3,1) += a[1] * b[3];
      sdata[tid].at(4,1) += a[1] * b[4];
      sdata[tid].at(5,1) += a[1] * b[5];
      sdata[tid].at(0,2) += a[2] * b[0];
      sdata[tid].at(1,2) += a[2] * b[1];
      sdata[tid].at(2,2) += a[2] * b[2];
      sdata[tid].at(3,2) += a[2] * b[3];
      sdata[tid].at(4,2) += a[2] * b[4];
      sdata[tid].at(5,2) += a[2] * b[5];
      sdata[tid].at(0,3) += a[3] * b[0];
      sdata[tid].at(1,3) += a[3] * b[1];
      sdata[tid].at(2,3) += a[3] * b[2];
      sdata[tid].at(3,3) += a[3] * b[3];
      sdata[tid].at(4,3) += a[3] * b[4];
      sdata[tid].at(5,3) += a[3] * b[5];
      sdata[tid].at(0,4) += a[4] * b[0];
      sdata[tid].at(1,4) += a[4] * b[1];
      sdata[tid].at(2,4) += a[4] * b[2];
      sdata[tid].at(3,4) += a[4] * b[3];
      sdata[tid].at(4,4) += a[4] * b[4];
      sdata[tid].at(5,4) += a[4] * b[5];
      sdata[tid].at(0,5) += a[5] * b[0];
      sdata[tid].at(1,5) += a[5] * b[1];
      sdata[tid].at(2,5) += a[5] * b[2];
      sdata[tid].at(3,5) += a[5] * b[3];
      sdata[tid].at(4,5) += a[5] * b[4];
      sdata[tid].at(5,5) += a[5] * b[5];
    }
  }
  __syncthreads();

  for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid < blockSize / 2 && tid < 32)
  {
    warpReduce<blockSize>(sdata, tid);
  }

  if(tid == 0)
  {
    res[blockIdx.x] = sdata[0] / static_cast<T>(N);
  }
}

template <unsigned int blockSize, typename T>
__global__ void cov_krnl(
    const rm::Vector3<T>* v1,
    const rm::Vector3<T>* v2,
    rm::Matrix3x3<T>* res,
    unsigned int N)
{
  __shared__ rm::Matrix3x3<T> sdata[blockSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int globId = N * blockIdx.x + threadIdx.x;
  const unsigned int rows = (N + blockSize - 1) / blockSize;

  sdata[tid].setZeros();
  for(unsigned int i = 0; i < rows; i++)
  {
    if(tid + blockSize * i < N)
    {
      const rm::Vector3<T>& a = v1[globId + blockSize * i];
      const rm::Vector3<T>& b = v2[globId + blockSize * i];
      //printf("%d: tid: %d, blidx %d, v @ %d, a: %f %f %f, b: %f %f %f\n", i, tid, blockIdx.x, globId + blockSize * i, a.x, a.y, a.z, b.x, b.y, b.z);
      sdata[tid](0,0) += a.x * b.x;
      sdata[tid](1,0) += a.x * b.y;
      sdata[tid](2,0) += a.x * b.z;
      sdata[tid](0,1) += a.y * b.x;
      sdata[tid](1,1) += a.y * b.y;
      sdata[tid](2,1) += a.y * b.z;
      sdata[tid](0,2) += a.z * b.x;
      sdata[tid](1,2) += a.z * b.y;
      sdata[tid](2,2) += a.z * b.z;
    }
  }
  __syncthreads();

  for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
  {
    if(tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid < blockSize / 2 && tid < 32)
  {
    warpReduce<blockSize>(sdata, tid);
  }

  if(tid == 0)
  {
    res[blockIdx.x] = sdata[0] / static_cast<T>(N);
  }
}

template <typename T>
__global__ void
calc_jacobis_krnl(const cuda::DeviceMap *map,
                  const rmagine::Matrix4x4f *total_transform,
                  const rm::Pointi *points,
                  rm::Point6l *jacobis,
                  TSDFEntry::ValueType *values, bool *mask, size_t N)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("%d %d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, blockDim.x, blockDim.y, threadIdx.x, N);
  if (idx < N)
  {
    rm::Matrix4x4i next_transform;
    cu_to_int_mat(*total_transform, next_transform);

    rm::Pointi center((int)(*total_transform)(3, 0), (int)(*total_transform)(3, 1), (int)(*total_transform)(3, 2));

    rm::Pointi point;
    cu_transform_point(points[idx], next_transform, point);

    rm::Pointi buf = point / MAP_RESOLUTION; // TODO MAP_RESOLUTION configurable

    point -= center;
    mask[idx] = false;

    if (map->in_bounds(buf) &&
        map->in_bounds(buf.x + 1, buf.y, buf.z) &&
        map->in_bounds(buf.x - 1, buf.y, buf.z) &&
        map->in_bounds(buf.x, buf.y + 1, buf.z) &&
        map->in_bounds(buf.x, buf.y - 1, buf.z) &&
        map->in_bounds(buf.x, buf.y, buf.z + 1) &&
        map->in_bounds(buf.x, buf.y, buf.z - 1))
    {
      const auto& current = map->value_unchecked(buf);
      if (current.weight() != 0)
      {
        const auto& x_next = map->value_unchecked(buf.x + 1, buf.y, buf.z);
        const auto& x_last = map->value_unchecked(buf.x - 1, buf.y, buf.z);
        const auto& y_next = map->value_unchecked(buf.x, buf.y + 1, buf.z);
        const auto& y_last = map->value_unchecked(buf.x, buf.y - 1, buf.z);
        const auto& z_next = map->value_unchecked(buf.x, buf.y, buf.z + 1);
        const auto& z_last = map->value_unchecked(buf.x, buf.y, buf.z - 1);

        rm::Pointi gradient;
        if (x_next.weight() != 0 && x_last.weight() != 0 && !((x_next.value() > 0 && x_last.value() < 0) || (x_next.value() < 0 && x_last.value() > 0)))
        {
          gradient.x = (x_next.value() - x_last.value()) / 2;
        }

        if (y_next.weight() != 0 && y_last.weight() != 0 && !((y_next.value() > 0 && y_last.value() < 0) || (y_next.value() < 0 && y_last.value() > 0)))
        {
          gradient.y = (y_next.value() - y_last.value()) / 2;
        }

        if (z_next.weight() != 0 && z_last.weight() != 0 && !((z_next.value() > 0 && z_last.value() < 0) || (z_next.value() < 0 && z_last.value() > 0)))
        {
          gradient.z = (z_next.value() - z_last.value()) / 2;
        }

        auto cross = point.cross(gradient);
        rm::Point6l jacobi((long)cross.x, (long)cross.y, (long)cross.z, (long)gradient.x, (long)gradient.y, (long)gradient.z);
        const auto& value = current.value();
        mask[idx] = true;
        jacobis[idx] = jacobi;
        values[idx] = value;
        if (idx < 100 && !(jacobi.data[0] == 0 && jacobi.data[1] == 0 && jacobi.data[2] == 0 && jacobi.data[3] == 0 && jacobi.data[4] == 0 && jacobi.data[5] == 0))
        {
          //printf("jacobi (gpu): %ld %ld %ld %ld %ld %ld\n", jacobi.data[0], jacobi.data[1], jacobi.data[2], jacobi.data[3], jacobi.data[4], jacobi.data[5]);
        }
      }
    }
  }
}

rm::Pointi transform_point(const rm::Pointi &point, const rm::Matrix4x4i &mat)
{
  // allocate device arrays
  auto cuda_mat = reinterpret_cast<const rm::Matrix4x4i *>(&mat.data);
  rm::Pointi *point_dev;
  rm::Matrix4x4i *mat_dev;
  rm::Pointi *out_dev;
  CHECK(cudaMalloc((void **) &point_dev, sizeof(rm::Pointi)));
  CHECK(cudaMalloc((void **) &mat_dev, sizeof(rm::Matrix4x4i)));
  CHECK(cudaMalloc((void **) &out_dev, sizeof(rm::Pointi)));
  CHECK(cudaMemcpy(point_dev, &point, sizeof(rm::Pointi), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(mat_dev, cuda_mat, sizeof(rm::Matrix4x4i), cudaMemcpyHostToDevice));

  int dimx = 1;
  int dimy = 1;
  dim3 block(dimx, dimy);
  dim3 grid((1 + block.x - 1) / block.x, 1);
  cu_transform_point_krnl<<<grid, block>>>(point_dev, mat_dev, out_dev);

  rm::Pointi out;
  CHECK(cudaMemcpy(&out, out_dev, sizeof(rm::Pointi), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(point_dev));
  CHECK(cudaFree(mat_dev));
  CHECK(cudaFree(out_dev));
  CHECK(cudaDeviceReset());

  return out;
}

void jacobi_2_h(const rm::Point6l &jacobi, rm::Matrix6x6l &h)
{
  // allocate device arrays
  rm::Point6l *jacobi_dev;
  rm::Matrix6x6l *h_dev;
  CHECK(cudaMalloc((void **) &jacobi_dev, sizeof(rm::Point6l)));
  CHECK(cudaMalloc((void **) &h_dev, sizeof(rm::Matrix6x6l)));
  CHECK(cudaMemcpy(jacobi_dev, &jacobi, sizeof(rm::Point6l), cudaMemcpyHostToDevice));

  int dimx = 1;
  int dimy = 1;
  dim3 block(dimx, dimy);
  dim3 grid((1 + block.x - 1) / block.x, 1);
  cu_jacobi_2_h_krnl<<<grid, block>>>(jacobi_dev, h_dev);

  CHECK(cudaMemcpy(&h, h_dev, sizeof(rm::Matrix6x6l), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(jacobi_dev));
  CHECK(cudaFree(h_dev));
  CHECK(cudaDeviceReset());
}

void cross(const rm::Vector3i &a, const rm::Vector3i &b, rm::Vector3i &res)
{
  // allocate device arrays
  rm::Vector3i *a_dev;
  rm::Vector3i *b_dev;
  rm::Vector3i *res_dev;
  CHECK(cudaMalloc((void **) &a_dev, sizeof(rm::Vector3i)));
  CHECK(cudaMalloc((void **) &b_dev, sizeof(rm::Vector3i)));
  CHECK(cudaMalloc((void **) &res_dev, sizeof(rm::Vector3i)));
  CHECK(cudaMemcpy(a_dev, &a, sizeof(rm::Vector3i), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_dev, &b, sizeof(rm::Vector3i), cudaMemcpyHostToDevice));

  int dimx = 1;
  int dimy = 1;
  dim3 block(dimx, dimy);
  dim3 grid((1 + block.x - 1) / block.x, 1);
  cu_cross_krnl<<<grid, block>>>(a_dev, b_dev, res_dev);

  CHECK(cudaMemcpy(&res, res_dev, sizeof(rm::Vector3i), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_dev));
  CHECK(cudaFree(b_dev));
  CHECK(cudaFree(res_dev));
  CHECK(cudaDeviceReset());
}

void reduce(float& sum, float* sum_dev, size_t size)
{
  // stop condition
  if (size == 1)
  {
    sum = sum_dev[0];
    return;
  }

  // renew the stride
  size_t stride = size / 2;

  // in-place reduction
#pragma unroll
  for (int i = 0; i < stride; i++)
  {
    sum_dev[i] += sum_dev[i + stride];
  }

  // call recursively
  return reduce(sum, sum_dev, stride);
}

float sum(std::vector<float> &data)
{
  size_t n = data.size();
  float *data_dev;
  float *res_dev;
  CHECK(cudaMalloc((void **) &data_dev, sizeof(float) * n));
  CHECK(cudaMalloc((void **) &res_dev, sizeof(float)));
  CHECK(cudaMemcpy(data_dev, data.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

  sum_krnl<1024><<<1, 1024>>>(data_dev, res_dev, n);

  float result;
  CHECK(cudaMemcpy(&result, res_dev, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(data_dev));
  CHECK(cudaFree(res_dev));
  CHECK(cudaDeviceReset());
  return result;
}

float sum(std::vector<float> &data, size_t split)
{
  size_t n = data.size();
  float *data_dev;
  float *res_dev;
  CHECK(cudaMalloc((void **) &data_dev, sizeof(float) * n));
  CHECK(cudaMalloc((void **) &res_dev, sizeof(float) * split));
  CHECK(cudaMemcpy(data_dev, data.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

  constexpr int blockdim = 1024;
  sum_krnl<blockdim><<<split, blockdim>>>(data_dev, res_dev, n < blockdim ? n : n / split);

  float* result_split = (float*) malloc(sizeof(float) * split);
  CHECK(cudaMemcpy(result_split, res_dev, sizeof(float) * split, cudaMemcpyDeviceToHost));

  float result = 0.f;
  reduce(result, result_split, split);

  CHECK(cudaFree(data_dev));
  CHECK(cudaFree(res_dev));
  CHECK(cudaDeviceReset());
  return result;
}

rm::Matrix3x3f cov(const std::vector<rm::Vector3f>& v1, const std::vector<rm::Vector3f>& v2)
{
  size_t n = v1.size();

  rm::Vector3f* v1_dev;
  rm::Vector3f * v2_dev;
  rm::Matrix3x3f* res_dev;
  CHECK(cudaMalloc((void **) &v1_dev, sizeof(rm::Vector3f) * n));
  CHECK(cudaMalloc((void **) &v2_dev, sizeof(rm::Vector3f) * n));
  CHECK(cudaMalloc((void **) &res_dev, sizeof(rm::Matrix3x3f)));
  CHECK(cudaMemcpy(v1_dev, v1.data(), sizeof(rm::Vector3f) * n, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(v2_dev, v2.data(), sizeof(rm::Vector3f) * n, cudaMemcpyHostToDevice));

  cov_krnl<1024><<<1, 1024>>>(v1_dev, v2_dev, res_dev, n);

  rm::Matrix3x3f result;
  CHECK(cudaMemcpy(&result, res_dev, sizeof(rm::Matrix3x3f), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(v1_dev));
  CHECK(cudaFree(v2_dev));
  CHECK(cudaFree(res_dev));
  CHECK(cudaDeviceReset());
  return result;
}

rm::Matrix6x6f cov6d(const std::vector<rm::Vector6f>& v1, const std::vector<rm::Vector6f>& v2)
{
  size_t n = v1.size();

  rm::Vector6f* v1_dev;
  rm::Vector6f* v2_dev;
  rm::Matrix6x6f* res_dev;
  CHECK(cudaMalloc((void **) &v1_dev, sizeof(rm::Vector6f) * n));
  CHECK(cudaMalloc((void **) &v2_dev, sizeof(rm::Vector6f) * n));
  CHECK(cudaMalloc((void **) &res_dev, sizeof(rm::Matrix6x6f)));
  CHECK(cudaMemcpy(v1_dev, v1.data(), sizeof(rm::Vector6f) * n, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(v2_dev, v2.data(), sizeof(rm::Vector6f) * n, cudaMemcpyHostToDevice));

  cov6d_krnl<256><<<1, 256>>>(v1_dev, v2_dev, res_dev, n);

  rm::Matrix6x6f result;
  CHECK(cudaMemcpy(&result, res_dev, sizeof(rm::Matrix6x6f), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(v1_dev));
  CHECK(cudaFree(v2_dev));
  CHECK(cudaFree(res_dev));
  CHECK(cudaDeviceReset());
  return result;
}

void h_g_e_reduction(const std::vector<rmagine::Vector6f> &v1, rmagine::Matrix6x6f &h, rmagine::Vector6f &g, float &e,
                     float &c)
{
  size_t n = v1.size();

  std::vector<TSDFEntry::ValueType> values(v1.size());
  std::fill(values.begin(), values.end(), 1);

  std::vector<uint8_t> mask(v1.size());
  std::fill(mask.begin(), mask.end(), 1);

  rm::Vector6f* v1_dev;
  TSDFEntry::ValueType* values_dev;
  bool* mask_dev;
  rm::Matrix6x6f* res_h_dev;
  rm::Vector6f* res_g_dev;
  float* res_e_dev;
  float* res_c_dev;
  CHECK(cudaMalloc((void **) &v1_dev, sizeof(rm::Vector6f) * n));
  CHECK(cudaMalloc((void **) &values_dev, sizeof(TSDFEntry::ValueType) * n));
  CHECK(cudaMalloc((void **) &res_h_dev, sizeof(rm::Matrix6x6f)));
  CHECK(cudaMalloc((void **) &res_g_dev, sizeof(rm::Vector6f)));
  CHECK(cudaMalloc((void **) &res_e_dev, sizeof(float)));
  CHECK(cudaMalloc((void **) &res_c_dev, sizeof(float)));
  CHECK(cudaMalloc((void **) &mask_dev, sizeof(bool) * n));
  CHECK(cudaMemcpy(v1_dev, v1.data(), sizeof(rm::Vector6f) * n, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(values_dev, values.data(), sizeof(TSDFEntry::ValueType) * n, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(mask_dev, mask.data(), sizeof(bool) * n, cudaMemcpyHostToDevice));

  h_g_e_reduction_krnl<256><<<1, 256>>>(v1_dev, values_dev, mask_dev, res_h_dev, res_g_dev, res_e_dev, res_c_dev, n);

  CHECK(cudaMemcpy(&h, res_h_dev, sizeof(rm::Matrix6x6f), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&g, res_g_dev, sizeof(rm::Vector6f), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&e, res_e_dev, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&e, res_c_dev, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(v1_dev));
  CHECK(cudaFree(res_h_dev));
  CHECK(cudaFree(res_g_dev));
  CHECK(cudaFree(res_e_dev));
  CHECK(cudaDeviceReset());
}

void
calc_jacobis(const cuda::DeviceMap &map, const std::vector<rm::Pointi> &points, std::vector<rm::Point6l> &jabobis,
             std::vector<TSDFEntry::ValueType> &values, std::vector<bool> &mask)
{
  size_t n_points = points.size();
  size_t n_map_voxels = map.size_->prod();
  rm::Matrix4x4f transform;
  transform.setIdentity();

  cuda::DeviceMap map_inner;
  cudaMalloc(&map_inner.size_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.offset_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.data_, n_map_voxels * sizeof(TSDFEntry));
  cudaMalloc(&map_inner.pos_, sizeof(rm::Pointi));
  cudaMemcpy(map_inner.size_, map.size_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.offset_, map.offset_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.data_, map.data_, n_map_voxels * sizeof(TSDFEntry), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.pos_, map.pos_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);

  rm::Vector3i* points_dev;
  cuda::DeviceMap* map_dev;
  rm::Matrix4x4f* transform_dev;
  rm::Point6l* gradients_dev;
  TSDFEntry::ValueType* values_dev;
  bool* mask_dev;
  CHECK(cudaMalloc((void **) &points_dev, sizeof(rm::Pointi) * n_points));
  CHECK(cudaMalloc((void **) &map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMalloc((void **) &transform_dev, sizeof(rm::Matrix4x4f)));
  CHECK(cudaMalloc((void **) &gradients_dev, sizeof(rm::Point6l) * n_points));
  CHECK(cudaMalloc((void **) &values_dev, sizeof(TSDFEntry::ValueType) * n_points));
  CHECK(cudaMalloc((void **) &mask_dev, sizeof(bool) * n_points));
  CHECK(cudaMemcpy(points_dev, points.data(), sizeof(rm::Pointi) * n_points, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(map_dev, &map_inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(transform_dev, &transform, sizeof(rm::Matrix4x4f), cudaMemcpyHostToDevice));

  calc_jacobis_krnl<int><<<128, 1024>>>(map_dev, transform_dev, points_dev, gradients_dev, values_dev, mask_dev, n_points);

  CHECK(cudaFree(points_dev));
  CHECK(cudaFree(map_dev));
  CHECK(cudaFree(transform_dev));
  CHECK(cudaFree(map_inner.size_));
  CHECK(cudaFree(map_inner.offset_));
  CHECK(cudaFree(map_inner.data_));
  CHECK(cudaFree(map_inner.pos_));
  CHECK(cudaFree(gradients_dev));
  CHECK(cudaFree(values_dev));
  CHECK(cudaFree(mask_dev));
  CHECK(cudaDeviceReset());
}

void registration(const cuda::DeviceMap &map, const rm::Matrix4x4f* pretransform,
                  const std::vector<rm::Pointi> &points, std::vector<rm::Point6l> &jacobis,
                  std::vector<TSDFEntry::ValueType> &values, std::vector<bool> &mask, rm::Matrix6x6l& h, rm::Point6l& g, int& e, int& c)
{
  size_t n_points = points.size();
  size_t n_map_voxels = map.size_->prod();

  // Cuda Allocations Jacobi Kernel
  cuda::DeviceMap map_inner;
  cudaMalloc(&map_inner.size_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.offset_, sizeof(rm::Pointi));
  cudaMalloc(&map_inner.data_, n_map_voxels * sizeof(TSDFEntry));
  cudaMalloc(&map_inner.pos_, sizeof(rm::Pointi));
  cudaMemcpy(map_inner.size_, map.size_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.offset_, map.offset_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.data_, map.data_, n_map_voxels * sizeof(TSDFEntry), cudaMemcpyHostToDevice);
  cudaMemcpy(map_inner.pos_, map.pos_, sizeof(rm::Pointi), cudaMemcpyHostToDevice);

  rm::Vector3i* points_dev;
  cuda::DeviceMap* map_dev;
  rm::Matrix4x4f* pretransform_dev;
  rm::Point6l* res_jacobis_dev;
  TSDFEntry::ValueType* res_values_dev;
  bool* res_mask_dev;
  CHECK(cudaMalloc((void **) &points_dev, sizeof(rm::Pointi) * n_points));
  CHECK(cudaMalloc((void **) &map_dev, sizeof(cuda::DeviceMap)));
  CHECK(cudaMalloc((void **) &pretransform_dev, sizeof(rm::Matrix4x4f)));
  CHECK(cudaMalloc((void **) &res_jacobis_dev, sizeof(rm::Point6l) * n_points));
  CHECK(cudaMalloc((void **) &res_values_dev, sizeof(TSDFEntry::ValueType) * n_points));
  CHECK(cudaMalloc((void **) &res_mask_dev, sizeof(bool) * n_points));
  CHECK(cudaMemcpy(points_dev, points.data(), sizeof(rm::Pointi) * n_points, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(map_dev, &map_inner, sizeof(cuda::DeviceMap), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(pretransform_dev, pretransform, sizeof(rm::Matrix4x4f), cudaMemcpyHostToDevice));

  // Cuda Allocations h += local_h
  rm::Matrix6x6l* res_h_dev;
  rm::Point6l* res_g_dev;
  int* res_e_dev;
  int* res_c_dev;
  CHECK(cudaMalloc((void **) &res_h_dev, sizeof(rm::Matrix6x6l)));
  CHECK(cudaMalloc((void **) &res_g_dev, sizeof(rm::Point6l)));
  CHECK(cudaMalloc((void **) &res_e_dev, sizeof(int)));
  CHECK(cudaMalloc((void **) &res_c_dev, sizeof(int)));

  calc_jacobis_krnl<int><<<128, 1024>>>(map_dev, pretransform_dev, points_dev, res_jacobis_dev, res_values_dev, res_mask_dev,
                                        n_points);
  h_g_e_reduction_krnl<128><<<1, 128>>>(res_jacobis_dev, res_values_dev, res_mask_dev, res_h_dev, res_g_dev, res_e_dev, res_c_dev, n_points);

  // return results
  CHECK(cudaMemcpy(&h, res_h_dev, sizeof(rm::Matrix6x6l), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&g, res_g_dev, sizeof(rm::Point6l), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&e, res_e_dev, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&c, res_c_dev, sizeof(int), cudaMemcpyDeviceToHost));

  // free jacobi kernel resources
  CHECK(cudaFree(points_dev));
  CHECK(cudaFree(map_dev));
  CHECK(cudaFree(map_inner.size_));
  CHECK(cudaFree(map_inner.offset_));
  CHECK(cudaFree(map_inner.data_));
  CHECK(cudaFree(map_inner.pos_));
  CHECK(cudaFree(res_jacobis_dev));
  CHECK(cudaFree(res_values_dev));
  CHECK(cudaFree(res_mask_dev));

  // h += local_h ressources
  CHECK(cudaFree(res_h_dev));
  CHECK(cudaFree(res_g_dev));
  CHECK(cudaFree(res_e_dev));
  CHECK(cudaFree(res_c_dev));

  CHECK(cudaDeviceReset());
}

int atomic_min(std::vector<int>& buffer, const std::vector<int>& values)
{
  size_t n_values = values.size();
  size_t n_buffer = buffer.size();

  int* buffer_dev;
  int* values_dev;

  CHECK(cudaMalloc((void **) &buffer_dev, sizeof(int) * n_buffer));
  CHECK(cudaMalloc((void **) &values_dev, sizeof(int) * n_values));
  CHECK(cudaMemcpy(buffer_dev, buffer.data(), sizeof(int) * n_buffer, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(values_dev, values.data(), sizeof(int) * n_values, cudaMemcpyHostToDevice));

  cu_write_min<<<100, 1000>>>(buffer_dev, values_dev, n_values);

  CHECK(cudaMemcpy(buffer.data(), buffer_dev, sizeof(int) * n_buffer, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(buffer_dev));
  CHECK(cudaFree(values_dev));

  CHECK(cudaDeviceReset());

  return buffer[0];
}

TSDFEntry atomic_tsdf_min(std::vector<TSDFEntry>& buffer, const std::vector<TSDFEntry>& values)
{
  size_t n_values = values.size();
  size_t n_buffer = buffer.size();

  TSDFEntry* buffer_dev;
  TSDFEntry* values_dev;
  CHECK(cudaMalloc((void **) &buffer_dev, sizeof(TSDFEntry) * n_buffer));
  CHECK(cudaMalloc((void **) &values_dev, sizeof(TSDFEntry) * n_values));
  CHECK(cudaMemcpy(buffer_dev, buffer.data(), sizeof(TSDFEntry) * n_buffer, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(values_dev, values.data(), sizeof(TSDFEntry) * n_values, cudaMemcpyHostToDevice));

  size_t blocksize = 1024;
  dim3 block(blocksize);
  dim3 grid((n_values + block.x - 1) / block.x);
  cu_write_tsdf_min<<<grid, block>>>(buffer_dev, values_dev, n_values);
  CHECK(cudaMemcpy(buffer.data(), buffer_dev, sizeof(TSDFEntry) * n_buffer, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(buffer_dev));
  CHECK(cudaFree(values_dev));
  CHECK(cudaDeviceReset());

  return buffer[0];
}

} // end namespace cuda
