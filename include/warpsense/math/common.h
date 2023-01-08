#pragma once

#include <math.h>
#include <stdint.h>

#ifdef __CUDA_ARCH__
#define RMAGINE_FUNCTION __host__ __device__
#define RMAGINE_INLINE_FUNCTION __forceinline__ __host__ __device__
#else
#define RMAGINE_FUNCTION
#define RMAGINE_INLINE_FUNCTION inline
#endif