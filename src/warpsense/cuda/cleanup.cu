#include "warpsense/cuda/cleanup.h"

void cuda::pause()
{
  cudaDeviceSynchronize();
}

void cuda::cleanup()
{
  cudaDeviceReset();
}
