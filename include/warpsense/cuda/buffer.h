#pragma once

#include <cmath>

namespace cuda
{

template<typename T>
struct Buffer
{
  Buffer(size_t N = 1);

  ~Buffer();

  void malloc();

  void free();

  void to_device(T &input);

  void to_device(const T &input);

  void to_device(T &input, size_t currN);

  void to_device(const T &input, size_t currN);

  void to_device(T *input);

  void to_device(const T *input);

  void to_device(T *input, size_t currN);

  void to_device(const T *input, size_t currN);

  void to_host(T *input);

  T *dev();

  const T *dev() const;

  T *dev_;
  size_t N;
};

} // end namespace cuda

#ifdef __CUDA_CC__

#include "warpsense/cuda/buffer.cu"

#endif