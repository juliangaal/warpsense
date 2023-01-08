#include "warpsense/cuda/buffer.h"
#include "warpsense/cuda/common.cuh"

using namespace cuda;

template <typename T>
Buffer<T>::Buffer(size_t N)
: N(N)
{
  malloc();
}

template<typename T>
Buffer<T>::~Buffer()
{
  free();
}

template<typename T>
void Buffer<T>::malloc()
{
  CHECK(cudaMalloc((void **) &dev_, sizeof(T) * N));
}

template<typename T>
void Buffer<T>::free()
{
  CHECK(cudaFree(dev_));
}

template<typename T>
T *Buffer<T>::dev()
{
  return dev_;
}

template<typename T>
const T *Buffer<T>::dev() const
{
  return dev_;
}

template<typename T>
void Buffer<T>::to_device(T& input)
{
  CHECK(cudaMemcpy(dev_, &input, sizeof(T) * N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(const T &input)
{
  CHECK(cudaMemcpy(dev_, &input, sizeof(T) * N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(T &input, size_t currN)
{
  CHECK(cudaMemcpy(dev_, &input, sizeof(T) * currN, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(const T &input, size_t currN)
{
  CHECK(cudaMemcpy(dev_, &input, sizeof(T) * currN, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(T* input)
{
  CHECK(cudaMemcpy(dev_, input, sizeof(T) * N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(const T* input)
{
  CHECK(cudaMemcpy(dev_, input, sizeof(T) * N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(T* input, size_t curr_N)
{
  CHECK(cudaMemcpy(dev_, input, sizeof(T) * curr_N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_device(const T* input, size_t curr_N)
{
  CHECK(cudaMemcpy(dev_, input, sizeof(T) * curr_N, cudaMemcpyHostToDevice));
}

template<typename T>
void Buffer<T>::to_host(T* result)
{
  CHECK(cudaMemcpy(result, dev_, sizeof(int) * N, cudaMemcpyDeviceToHost));
}

