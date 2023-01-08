#pragma once

#ifdef __CUDA_ARCH__
#define COMPILE_TIME_CONSTANT __constant__ constexpr
#else
#define COMPILE_TIME_CONSTANT constexpr
#endif

COMPILE_TIME_CONSTANT int WEIGHT_SHIFT = 6;                         // bitshift for a faster way to apply WEIGHT_RESOLUTION
COMPILE_TIME_CONSTANT int WEIGHT_RESOLUTION = 1 << WEIGHT_SHIFT;    // Resolution of the Weights. A weight of 1.0f is represented as WEIGHT_RESOLUTION

COMPILE_TIME_CONSTANT int MATRIX_SHIFT = 15;                        // bitshift for a faster way to apply MATRIX_RESOLUTION
COMPILE_TIME_CONSTANT int MATRIX_RESOLUTION = 1 << MATRIX_SHIFT;    // Resolution of calculations (Matrices, division, ...)
