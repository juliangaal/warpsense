#pragma once

#include <warpsense/math/vector3.h>
#include <warpsense/preprocessing.h>

namespace std
{
template<>
struct hash<rmagine::Pointi>
{
  std::size_t operator()(rmagine::Pointi const &p) const noexcept
  {
    long long v = ((long long) p.x << 32) ^ ((long long) p.y << 16) ^ (long long) p.z;
    return std::hash<long long>()(v);
  }
};
}
