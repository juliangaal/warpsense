#pragma once

#include "common.h"

namespace rmagine
{

/**
 * @brief Vector3 type
 * 
 */
template <typename T>
struct Vector3
{
  T x;
  T y;
  T z;

  RMAGINE_INLINE_FUNCTION
  Vector3()
  : x(0), y(0), z(0)
  {}

  RMAGINE_INLINE_FUNCTION
  Vector3(T x, T y, T z)
  : x(x), y(y), z(z)
  {}

  RMAGINE_INLINE_FUNCTION
  Vector3(const Vector3<T> &other)
  {
    x = other.x;
    y = other.y;
    z = other.z;
  }

  RMAGINE_INLINE_FUNCTION
  static const Vector3<T> Constant(T val);

  // FUNCTIONS
  RMAGINE_INLINE_FUNCTION
  Vector3<T> add(const Vector3<T> &b) const;

  RMAGINE_INLINE_FUNCTION
  void addInplace(const Vector3<T> &b);

  RMAGINE_INLINE_FUNCTION
  void addInplace(volatile Vector3<T> &b) volatile;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> sub(const Vector3<T> &b) const;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> negation() const;

  template <typename S>
  RMAGINE_INLINE_FUNCTION
  Vector3<S> cast() const;

  RMAGINE_INLINE_FUNCTION
  void negate();

  RMAGINE_INLINE_FUNCTION
  void subInplace(const Vector3<T> &b);

  RMAGINE_INLINE_FUNCTION
  T dot(const Vector3<T> &b) const;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> cross(const Vector3<T> &b) const;

  RMAGINE_INLINE_FUNCTION
  T mult(const Vector3<T> &b) const;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> mult(const T &s) const;

  RMAGINE_INLINE_FUNCTION
  void multInplace(const T &s);

  RMAGINE_INLINE_FUNCTION
  Vector3<T> div(const T &s) const;

  RMAGINE_INLINE_FUNCTION
  void divInplace(const T &s);

  RMAGINE_INLINE_FUNCTION
  T l2normSquared() const;



  RMAGINE_INLINE_FUNCTION
  T l2norm() const;

  RMAGINE_INLINE_FUNCTION
  T sum() const;

  RMAGINE_INLINE_FUNCTION
  T prod() const;

  RMAGINE_INLINE_FUNCTION
  T l1norm() const;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> cwiseAbs() const;

  RMAGINE_INLINE_FUNCTION
  Vector3<T> normalized() const;

  RMAGINE_INLINE_FUNCTION
  void normalize();

  RMAGINE_INLINE_FUNCTION
  void setZeros();

  // OPERATORS
  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator+(const Vector3<T> &b) const
  {
    return add(b);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> &operator+=(const Vector3<T> &b)
  {
    addInplace(b);
    return *this;
  }

  RMAGINE_INLINE_FUNCTION
  volatile Vector3<T> &operator+=(volatile Vector3<T> &b) volatile
  {
    addInplace(b);
    return *this;
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator-(const Vector3<T> &b) const
  {
    return sub(b);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> &operator-=(const Vector3<T> &b)
  {
    subInplace(b);
    return *this;
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator-() const
  {
    return negation();
  }

  RMAGINE_INLINE_FUNCTION
  T operator*(const Vector3<T> &b) const
  {
    return mult(b);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator*(const T &s) const
  {
    return mult(s);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> &operator*=(const T &s)
  {
    multInplace(s);
    return *this;
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator/(const T &s) const
  {
    return div(s);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3<T> operator/=(const T &s)
  {
    divInplace(s);
    return *this;
  }

  RMAGINE_INLINE_FUNCTION
  bool operator==(const Vector3<T> &other)
  {
    return other.x == x && other.y == y && other.z == z;
  }
};

//////////////////////////////
///// INLINE IMPLEMENTATIONS
///////////////////////////////


/////////////////////
///// Vector3 ///////
/////////////////////

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::add(const Vector3<T> &b) const
{
  return {x + b.x, y + b.y, z + b.z};
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::addInplace(const Vector3<T> &b)
{
  x += b.x;
  y += b.y;
  z += b.z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::addInplace(volatile Vector3<T> &b) volatile
{
  x += b.x;
  y += b.y;
  z += b.z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::sub(const Vector3<T> &b) const
{
  return {x - b.x, y - b.y, z - b.z};
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::negation() const
{
  return {-x, -y, -z};
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::negate()
{
  x = -x;
  y = -y;
  z = -z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::subInplace(const Vector3<T> &b)
{
  x -= b.x;
  y -= b.y;
  z -= b.z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::dot(const Vector3<T> &b) const
{
  return x * b.x + y * b.y + z * b.z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::cross(const Vector3<T> &b) const
{
  return {
      y * b.z - z * b.y,
      z * b.x - x * b.z,
      x * b.y - y * b.x
  };
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::mult(const Vector3<T> &b) const
{
  return dot(b);
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::mult(const T &s) const
{
  return {x * s, y * s, z * s};
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::multInplace(const T &s)
{
  x *= s;
  y *= s;
  z *= s;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::div(const T &s) const
{
  return {x / s, y / s, z / s};
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::divInplace(const T &s)
{
  x /= s;
  y /= s;
  z /= s;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::l2normSquared() const
{
  return x * x + y * y + z * z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::l2norm() const
{
  return sqrtf(l2normSquared());
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::sum() const
{
  return x + y + z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::prod() const
{
  return x * y * z;
}

template <typename T>
RMAGINE_INLINE_FUNCTION
T Vector3<T>::l1norm() const
{
  return fabs(x) + fabs(y) + fabs(z);
}

template <typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::normalized() const
{
  return div(l2norm());
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::normalize()
{
  divInplace(l2norm());
}

template <typename T>
RMAGINE_INLINE_FUNCTION
void Vector3<T>::setZeros()
{
  x = 0.0;
  y = 0.0;
  z = 0.0;
}

template <typename T>
template <typename S>
RMAGINE_INLINE_FUNCTION
Vector3<S> Vector3<T>::cast() const
{
  return { static_cast<S>(x), static_cast<S>(y), static_cast<S>(z) };
}

template<typename T>
RMAGINE_INLINE_FUNCTION
Vector3<T> Vector3<T>::cwiseAbs() const
{
  return Vector3<T>(abs(x), abs(y), abs(z));
}

template<typename T>
RMAGINE_INLINE_FUNCTION
const Vector3<T> Vector3<T>::Constant(T val)
{
  return Vector3<T>(val, val, val);
}

template <typename T>
RMAGINE_INLINE_FUNCTION
bool operator==(const Vector3<T> one, const Vector3<T> &other)
{
  return other.x == one.x && other.y == one.y && other.z == one.z;
}

template <typename T, typename S>
RMAGINE_INLINE_FUNCTION
Vector3<T> operator*(S scalar, const Vector3<T> one)
{
  return one * (T)scalar;
}

using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;
using Vector3i = Vector3<int>;
using Pointf = Vector3<float>;
using Pointi = Vector3<int>;
using Pointl = Vector3<long>;
using Pointll = Vector3<long long>;

} // namespace rmagine