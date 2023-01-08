#pragma once

#include "map/tsdf.h"
#include "warpsense/consts.h"
#include "warpsense/math/math.h"
#include <stdio.h>

namespace cuda
{

__forceinline__ __device__ void cu_transform_point(const rmagine::Pointi& input, const rmagine::Matrix4x4i& mat, rmagine::Pointi& out)
{
  out.x = mat.at(0, 0) * input.x + mat.at(0, 1) * input.y + mat.at(0, 2) * input.z;
  out.y = mat.at(1, 0) * input.x + mat.at(1, 1) * input.y + mat.at(1, 2) * input.z;
  out.z = mat.at(2, 0) * input.x + mat.at(2, 1) * input.y + mat.at(2, 2) * input.z;
  out.x += mat.at(0, 3);
  out.y += mat.at(1, 3);
  out.z += mat.at(2, 3);
  out.x /= MATRIX_RESOLUTION;
  out.y /= MATRIX_RESOLUTION;
  out.z /= MATRIX_RESOLUTION;
}

__forceinline__ __device__ __host__ void cu_to_int_mat(const rmagine::Matrix4x4f& in, rmagine::Matrix4x4i& out)
{
#pragma unroll
  for (int i = 0; i < 4; ++i)
  {
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
      out(i, j) = (int)(in(i, j) * MATRIX_RESOLUTION);
    }
  }
}

__forceinline__ __device__ void cu_jacobi_2_h(const rmagine::Point6l* jacobian, rmagine::Matrix6x6l* h)
{
#pragma unroll
  for (int i = 0; i < 6; ++i)
  {
#pragma unroll
    for (int j = 0; j < 6; ++j)
    {
      (*h).at(i, j) = (int)(*jacobian).at(i) * (int)(*jacobian).at(j);
    }
  }
}

template <unsigned int blockSize, typename T>
__forceinline__ __device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
  if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if(blockSize >= 16) sdata[tid] += sdata[tid +  8];
  if(blockSize >=  8) sdata[tid] += sdata[tid +  4];
  if(blockSize >=  4) sdata[tid] += sdata[tid +  2];
  if(blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

/**
 * See https://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf page 14 and 40
 * atomicCAS: uint64 atomicCAS(T *data, T oldval, T newval)
 *                If “*data” is equal to “oldval”, replace it with “newval”
 *                Always returns original value of “*data”"
 * @param address
 * @param new_entry_raw
 * @return
 */
__forceinline__ __device__ int16_t atomic_tsdf_min(int* address,
                                                    const int new_entry_raw) {
  int old = *address, assumed;

  if((abs(*((int16_t*)&old))) < abs(*((int16_t*)&new_entry_raw)) || *(((int16_t*)&old) + 1) > 0) // TODO compare > 0, < 0, >=
  {
    return *((int16_t *)&old);
  }

  do
  {
    // ich denke der aktuelle Wert ist der alte
    assumed = old;
    if (abs(*((int16_t *)&assumed)) < abs(*((int16_t *)&new_entry_raw)) || *(((int16_t*)&assumed) + 1) > 0)
    {
      // egal, nix zu tun ODER old aus atomicCAS aus anderem Thread ist kleiner als new_entry_raw
      break;
    }

    // Nur, wenn adresse noch das selbe ist, weiss ich, dass kein anderer Thread einen
    // kleineren Wert in die adresse geschrieben hat

    // Wenn kein anderer Thread einen kleinen Wert gefunden hat, schreibt "unser" thread
    // einen auf jeden fall kleineren Wert in adresse
    // -> old und assumed bleibt gleich UND die Schleife bricht ab

    // Wenn **ein** anderer Thread einen kleinen Wert gefunden hat, dann sind adresse und assumed ungleich
    // -> old enthaelt den neuen kleinsten Wert aus einem anderen Thread
    old = atomicCAS(address, assumed, new_entry_raw);
  } while(assumed != old);

  return *((int16_t *)&old);
}

__forceinline__ __device__ void write_tsdf_min(TSDFEntry* __restrict__ depth_buffer,
                                               unsigned int coord,
                                               const TSDFEntry& entry)
{
  atomic_tsdf_min(reinterpret_cast<int*>(depth_buffer + coord), *reinterpret_cast<const int*>(&entry));
}

__forceinline__ __device__ rmagine::Pointi cu_to_map(const rmagine::Pointi& p, int map_resolution)
{
  return { (int)floorf((float)p.x / (float)map_resolution), (int)floor((float)p.y / (float)map_resolution), (int)floor((float)p.z / (float)map_resolution) } ;
}

__forceinline__ __device__ rmagine::Pointi cu_to_mm(const rmagine::Pointi& p, int map_resolution)
{
  rmagine::Pointi out;
  out.x = (p.x * map_resolution) + map_resolution / 2;
  out.y = (p.y * map_resolution) + map_resolution / 2;
  out.z = (p.z * map_resolution) + map_resolution / 2;
  return out;
}

} // end namespace cuda
