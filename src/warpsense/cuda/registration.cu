#include <vector>
#include <cuda_runtime.h>
#include "warpsense/consts.h"
#include "warpsense/cuda/common.cuh"
#include "warpsense/cuda/util.h"
#include "warpsense/cuda/registration.h"
#include "warpsense/math/math.h"

namespace rm = rmagine;

namespace cuda
{

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
  __shared__ rm::Vector6<S> s_h_diag[blockSize];
  __shared__ rm::VectorN<15, S> s_h_upper[blockSize];
  __shared__ rm::Vector6<S> s_g[blockSize];
  __shared__ T s_e[blockSize];
  __shared__ T s_c[blockSize];

  const unsigned int tid = threadIdx.x;
  const unsigned int globId = N * blockIdx.x + threadIdx.x;
  const unsigned int rows = (N + blockSize - 1) / blockSize;

  s_h_diag[tid].setZeros();
  s_h_upper[tid].setZeros();
  s_g[tid].setZeros();
  s_e[tid] = 0;
  s_c[tid] = 0;

  for(unsigned int i=0; i < rows; i++)
  {
    if(tid + blockSize * i < N && mask[globId + blockSize * i] == true)
    {
      const rm::Vector6<S>& jacobi = jacobis[globId + blockSize * i];
      const TSDFEntry::ValueType& current_value = values[globId + blockSize * i];

      // update h, only diagonal!
      // aa  ..  ..  ..  ..  ..
      // ..  bb  ..  ..  ..  ..
      // ..  ..  cc  ..  ..  ..
      // ..  ..  ..  dd  ..  ..
      // ..  ..  ..  ..  ee  ..
      // ..  ..  ..  ..  ..  ff
      s_h_diag[tid].at(0) += jacobi[0] * jacobi[0];
      s_h_diag[tid].at(1) += jacobi[1] * jacobi[1];
      s_h_diag[tid].at(2) += jacobi[2] * jacobi[2];
      s_h_diag[tid].at(3) += jacobi[3] * jacobi[3];
      s_h_diag[tid].at(4) += jacobi[4] * jacobi[4];
      s_h_diag[tid].at(5) += jacobi[5] * jacobi[5];

      // aa  ab  ac  ad  ae  af
      // ab  bb  bc  bd  be  bf
      // ..  ..  cc  cd  ce  cf
      // ..  ..  ..  dd  de  df
      // ..  ..  ..  ..  ee  df
      // ..  ..  ..  ..  ..  ff

      // ab - ac - ad - ae - af
      s_h_upper[tid](0) += jacobi[0] * jacobi[1];
      s_h_upper[tid](1) += jacobi[0] * jacobi[2];
      s_h_upper[tid](2) += jacobi[0] * jacobi[3];
      s_h_upper[tid](3) += jacobi[0] * jacobi[4];
      s_h_upper[tid](4) += jacobi[0] * jacobi[5];

      // bc - bd - be - bf
      s_h_upper[tid](5) += jacobi[1] * jacobi[2];
      s_h_upper[tid](6) += jacobi[1] * jacobi[3];
      s_h_upper[tid](7) += jacobi[1] * jacobi[4];
      s_h_upper[tid](8) += jacobi[1] * jacobi[5];

      // cd - ce - cf
      s_h_upper[tid](9) += jacobi[2] * jacobi[3];
      s_h_upper[tid](10) += jacobi[2] * jacobi[4];
      s_h_upper[tid](11) += jacobi[2] * jacobi[5];

      // de - df
      s_h_upper[tid](12) += jacobi[3] * jacobi[4];
      s_h_upper[tid](13) += jacobi[3] * jacobi[5];

      // ef
      s_h_upper[tid](14) += jacobi[4] * jacobi[5];

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
      s_h_diag[tid] += s_h_diag[tid + s];
      s_h_upper[tid] += s_h_upper[tid + s];
      s_g[tid] += s_g[tid + s];
      s_e[tid] += s_e[tid + s];
      s_c[tid] += s_c[tid + s];
    }
    __syncthreads();
  }

  if(tid < blockSize / 2 && tid < 32)
  {
    warpReduce<blockSize>(s_h_diag, tid);
    warpReduce<blockSize>(s_h_upper, tid);
    warpReduce<blockSize>(s_g, tid);
    warpReduce<blockSize>(s_e, tid);
    warpReduce<blockSize>(s_c, tid);
  }

  if(tid == 0)
  {
    //res_h[blockIdx.x] = s_h[0];
    res_h[blockIdx.x].at(0, 0) = s_h_diag[0].at(0);
    res_h[blockIdx.x].at(1, 1) = s_h_diag[0].at(1);
    res_h[blockIdx.x].at(2, 2) = s_h_diag[0].at(2);
    res_h[blockIdx.x].at(3, 3) = s_h_diag[0].at(3);
    res_h[blockIdx.x].at(4, 4) = s_h_diag[0].at(4);
    res_h[blockIdx.x].at(5, 5) = s_h_diag[0].at(5);

    res_h[blockIdx.x].at(1, 0) = s_h_upper[0](0);
    res_h[blockIdx.x].at(0, 1) = s_h_upper[0](0);

    res_h[blockIdx.x].at(2, 0) = s_h_upper[0](1);
    res_h[blockIdx.x].at(0, 2) = s_h_upper[0](1);

    res_h[blockIdx.x].at(3, 0) = s_h_upper[0](2);
    res_h[blockIdx.x].at(0, 3) = s_h_upper[0](2);

    res_h[blockIdx.x].at(4, 0) = s_h_upper[0](3);
    res_h[blockIdx.x].at(0, 4) = s_h_upper[0](3);

    res_h[blockIdx.x].at(5, 0) = s_h_upper[0](4);
    res_h[blockIdx.x].at(0, 5) = s_h_upper[0](4);

    res_h[blockIdx.x].at(2, 1) = s_h_upper[0](5);
    res_h[blockIdx.x].at(1, 2) = s_h_upper[0](5);

    res_h[blockIdx.x].at(3, 1) = s_h_upper[0](6);
    res_h[blockIdx.x].at(1, 3) = s_h_upper[0](6);

    res_h[blockIdx.x].at(4, 1) = s_h_upper[0](7);
    res_h[blockIdx.x].at(1, 4) = s_h_upper[0](7);

    res_h[blockIdx.x].at(5, 1) = s_h_upper[0](8);
    res_h[blockIdx.x].at(1, 5) = s_h_upper[0](8);

    res_h[blockIdx.x].at(3, 2) = s_h_upper[0](9);
    res_h[blockIdx.x].at(2, 3) = s_h_upper[0](9);

    res_h[blockIdx.x].at(4, 2) = s_h_upper[0](10);
    res_h[blockIdx.x].at(2, 4) = s_h_upper[0](10);

    res_h[blockIdx.x].at(5, 2) = s_h_upper[0](11);
    res_h[blockIdx.x].at(2, 5) = s_h_upper[0](11);

    res_h[blockIdx.x].at(4, 3) = s_h_upper[0](12);
    res_h[blockIdx.x].at(3, 4) = s_h_upper[0](12);

    res_h[blockIdx.x].at(5, 3) = s_h_upper[0](13);
    res_h[blockIdx.x].at(3, 5) = s_h_upper[0](13);

    res_h[blockIdx.x].at(5, 4) = s_h_upper[0](14);
    res_h[blockIdx.x].at(4, 5) = s_h_upper[0](14);

    res_g[blockIdx.x] = s_g[0];
    res_e[blockIdx.x] = s_e[0];
    res_c[blockIdx.x] = s_c[0];
  }
}

template <typename T>
__global__ void
calc_jacobis_krnl(const cuda::DeviceMap *map,
                  const rmagine::Matrix4x4f *total_transform,
                  const rm::Pointi *points,
                  rm::Point6l *jacobis,
                  TSDFEntry::ValueType *values, bool *mask, int map_resolution, size_t N)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("%d %d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, blockDim.x, blockDim.y, threadIdx.x, N);
  if (idx < N)
  {
    rm::Matrix4x4i next_transform;
    cu_to_int_mat(*total_transform, next_transform);

    rm::Pointi center((int)(*total_transform)(0, 3), (int)(*total_transform)(1, 3), (int)(*total_transform)(2, 3)); // TODO floor!

    rm::Pointi point;
    cu_transform_point(points[idx], next_transform, point);

    rm::Pointi buf = point / map_resolution; // TODO variable map resolution, TODO why flooring worse result

    point -= center;

    mask[idx] = false;

    if (map->in_bounds_with_buffer_neg(buf, 1))
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
      }
    }
  }
}

RegistrationCuda::RegistrationCuda(const cuda::DeviceMap &map)
{
  max_points = 128 * 1024;
  curr_n_points = max_points;

  // reduction kernel, sizes never change
  CHECK(cudaMalloc((void **) &res_h_dev, sizeof(rm::Matrix6x6l) * reduction_split));
  CHECK(cudaMalloc((void **) &res_g_dev, sizeof(rm::Point6l) * reduction_split));
  CHECK(cudaMalloc((void **) &res_e_dev, sizeof(int) * reduction_split));
  CHECK(cudaMalloc((void **) &res_c_dev, sizeof(int) * reduction_split));

  res_h = (rm::Matrix6x6l*) malloc(reduction_split * sizeof(rm::Matrix6x6l));
  res_g = (rm::Point6l*) malloc(reduction_split * sizeof(rm::Point6l));
  res_e = (int*) malloc(reduction_split * sizeof(int));
  res_c = (int*) malloc(reduction_split * sizeof(int));

  // Jacobian Kernel
  CHECK(cudaMalloc((void **) &points_dev, sizeof(rm::Pointi) * max_points));
  CHECK(cudaMalloc((void **) &pretransform_dev, sizeof(rm::Matrix4x4f)));
  CHECK(cudaMalloc((void **) &res_jacobis_dev, sizeof(rm::Point6l) * max_points));
  CHECK(cudaMalloc((void **) &res_values_dev, sizeof(TSDFEntry::ValueType) * max_points));
  CHECK(cudaMalloc((void **) &res_mask_dev, sizeof(bool) * max_points));
}

RegistrationCuda::~RegistrationCuda()
{
  // free jacobi kernel resources
  CHECK(cudaFree(res_jacobis_dev));
  CHECK(cudaFree(res_values_dev));
  CHECK(cudaFree(res_mask_dev));
  CHECK(cudaFree(pretransform_dev));
  CHECK(cudaFree(points_dev));

  // free reduction resources
  free(res_h);
  free(res_g);
  free(res_e);
  free(res_c);
  CHECK(cudaFree(res_h_dev));
  CHECK(cudaFree(res_g_dev));
  CHECK(cudaFree(res_e_dev));
  CHECK(cudaFree(res_c_dev));
}

void RegistrationCuda::prepare_registration(const std::vector<rm::Pointi> &points)
{
  curr_n_points = points.size();
  // Jacobian Kernel
  CHECK(cudaMemcpy(points_dev, points.data(), sizeof(rm::Pointi) * curr_n_points, cudaMemcpyHostToDevice));
}

void RegistrationCuda::reduce(rm::Matrix6x6l& h,
           rm::Point6l& g,
           int& e,
           int& c,
           rm::Matrix6x6l *h_res,
           rm::Point6l* g_res,
           int* e_res,
           int* c_res,
           int const size)
{
  // stop condition
  if (size == 1)
  {
    h = h_res[0];
    g = g_res[0];
    e = e_res[0];
    c = c_res[0];
    return;
  }

  // renew the stride
  int const stride = size / 2;

  // in-place reduction
#pragma unroll
  for (int i = 0; i < stride; i++)
  {
    h_res[i] += h_res[i + stride];
    g_res[i] += g_res[i + stride];
    e_res[i] += e_res[i + stride];
    c_res[i] += c_res[i + stride];
  }

  // call recursively
  return reduce(h, g, e, c, h_res, g_res, e_res, c_res, stride);
}

void RegistrationCuda::perform_registration(const cuda::DeviceMap *map_dev, const rmagine::Matrix4x4f *pretransform,
                                            rmagine::Matrix6x6l &h,
                                            rmagine::Point6l &g, int &e, int &c, int map_resolution)
{
  CHECK(cudaMemcpy(pretransform_dev, pretransform, sizeof(rm::Matrix4x4f), cudaMemcpyHostToDevice));

  calc_jacobis_krnl<int><<<128, 512>>>(map_dev, pretransform_dev, points_dev, res_jacobis_dev, res_values_dev, res_mask_dev,
                                       map_resolution, curr_n_points);
  constexpr int blockdim = 128;
  h_g_e_reduction_krnl<blockdim><<<reduction_split, blockdim>>>(res_jacobis_dev, res_values_dev, res_mask_dev, res_h_dev, res_g_dev, res_e_dev, res_c_dev, curr_n_points < blockdim ? curr_n_points : curr_n_points / reduction_split);

  CHECK(cudaMemcpy(res_h, res_h_dev, sizeof(rm::Matrix6x6l) * reduction_split, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(res_g, res_g_dev, sizeof(rm::Point6l) * reduction_split, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(res_e, res_e_dev, sizeof(int) * reduction_split, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(res_c, res_c_dev, sizeof(int) * reduction_split, cudaMemcpyDeviceToHost));

  h.setZeros();
  g.setZeros();
  e = 0;
  c = 0;
  reduce(h, g, e, c, res_h, res_g, res_e, res_c, reduction_split);
}

} // end namespace cuda

