#pragma once

#include "warpsense/consts.h"
#include "warpsense/math/common.h"

constexpr int DEFAULT_VALUE = 4;
constexpr int DEFAULT_WEIGHT = 6;

template <typename T>
RMAGINE_INLINE_FUNCTION
bool approx(T a, T b, T epsilon)
{
  return abs(a - b) < epsilon;
}

template <typename T, typename F>
RMAGINE_INLINE_FUNCTION
int calc_weight(T value, F tau, F weight_epsilon)
{
  int weight = WEIGHT_RESOLUTION;
  if (value < -weight_epsilon)
  {
    weight = WEIGHT_RESOLUTION * (tau + value) / (tau - weight_epsilon);
  }
  return weight;
}
