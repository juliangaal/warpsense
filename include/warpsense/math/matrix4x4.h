#pragma once

#include "common.h"
#include "vector3.h"
#include "matrix3x3.h"

namespace rmagine
{

/**
 * @brief Matrix4x4<S>  type.
 * 
 * Same order as Eigen-default -> can be reinterpret-casted or mapped.
 * 
 * Storage order ()-operator 
 * (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), ... 
 * 
 * Storage order []-operator
 * [0][0], [0][1], [0][2], [0][3], [1][0], [1][1], [1][2], ...
 * 
 */
template <typename S>
struct Matrix4x4 {
    S data[4][4];

    RMAGINE_INLINE_FUNCTION
    S& at(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    S at(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    S& operator()(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    S operator()(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    S* operator[](const unsigned int i);

    RMAGINE_INLINE_FUNCTION
    const S* operator[](const unsigned int i) const;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    RMAGINE_INLINE_FUNCTION
    void setOnes();

    // RMAGINE_INLINE_FUNCTION
    // void set(const Transform& T);

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> rotation() const;

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Matrix3x3<S> & R);

    // RMAGINE_INLINE_FUNCTION
    // void setRotation(const Quaternion<S> & q);

    // RMAGINE_INLINE_FUNCTION
    // void setRotation(const EulerAngles<S> & e);

    RMAGINE_INLINE_FUNCTION
    Vector3<S>  translation() const;

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Vector3<S> & t);

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  transpose() const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  T() const;

    RMAGINE_INLINE_FUNCTION
    S trace() const;

    RMAGINE_INLINE_FUNCTION
    S det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  inv() const ;

    /**
     * @brief Assuming Matrix4x4<S>  to be rigid transformation. Then: (R|t)^(-1) = (R^T| -R^S t)
     * 
     * @return Matrix4x4<S>  
     */
    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  invRigid();

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  mult(const S& s) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const S& s);

    RMAGINE_INLINE_FUNCTION
    Vector3<S>  mult(const Vector3<S> & v) const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  mult(const Matrix4x4<S> &   M) const;

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  div(const S& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const S& s);

    // OPERATORS
    // RMAGINE_INLINE_FUNCTION
    // void operator=(const Transform& T)
    // {
    //     set(T);
    // }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  operator*(const S& s) const
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S> & operator*=(const S& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  operator/(const S& s) const
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S> & operator/=(const S& s)
    {
        divInplace(s);
        return *this;
    }


    RMAGINE_INLINE_FUNCTION
    Vector3<S>  operator*(const Vector3<S> & v) const 
    {
        return mult(v);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  operator*(const Matrix4x4<S> & M) const 
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix4x4<S>  operator~() const 
    {
        return inv();
    }
};

using Matrix4x4f = Matrix4x4<float>;
using Matrix4x4i = Matrix4x4<int>;
using Matrix4x4d = Matrix4x4<double>;

template <typename S>
RMAGINE_INLINE_FUNCTION
S& Matrix4x4<S>::at(unsigned int i, unsigned int j)
{
    return data[j][i];
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix4x4<S>::at(unsigned int i, unsigned int j) const
{
    return data[j][i];
}   

template <typename S>
RMAGINE_INLINE_FUNCTION
S& Matrix4x4<S>::operator()(unsigned int i, unsigned int j)
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix4x4<S>::operator()(unsigned int i, unsigned int j) const
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S* Matrix4x4<S>::operator[](const unsigned int i) 
{
    return data[i];
};

template <typename S>
RMAGINE_INLINE_FUNCTION
const S* Matrix4x4<S>::operator[](const unsigned int i) const 
{
    return data[i];
};

// FUNCTIONS
template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::setIdentity()
{
    at(0,0) = 1.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 1.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 1.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 1.0;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::setZeros()
{
    at(0,0) = 0.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 0.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 0.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 0.0;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::setOnes()
{
    at(0,0) = 1.0;
    at(0,1) = 1.0;
    at(0,2) = 1.0;
    at(0,3) = 1.0;
    at(1,0) = 1.0;
    at(1,1) = 1.0;
    at(1,2) = 1.0;
    at(1,3) = 1.0;
    at(2,0) = 1.0;
    at(2,1) = 1.0;
    at(2,2) = 1.0;
    at(2,3) = 1.0;
    at(3,0) = 1.0;
    at(3,1) = 1.0;
    at(3,2) = 1.0;
    at(3,3) = 1.0;
}

// template <typename S>
// RMAGINE_INLINE_FUNCTION
// void Matrix4x4<S>::set(const Transform& T)
// {
//     setIdentity();
//     setRotation(T.R);
//     setTranslation(T.t);
// }

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix4x4<S>::rotation() const
{
    Matrix3x3<S> R;
    R(0,0) = at(0,0);
    R(0,1) = at(0,1);
    R(0,2) = at(0,2);
    R(1,0) = at(1,0);
    R(1,1) = at(1,1);
    R(1,2) = at(1,2);
    R(2,0) = at(2,0);
    R(2,1) = at(2,1);
    R(2,2) = at(2,2);
    return R;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::setRotation(const Matrix3x3<S>& R)
{
    at(0,0) = R(0,0);
    at(0,1) = R(0,1);
    at(0,2) = R(0,2);
    at(1,0) = R(1,0);
    at(1,1) = R(1,1);
    at(1,2) = R(1,2);
    at(2,0) = R(2,0);
    at(2,1) = R(2,1);
    at(2,2) = R(2,2);
}

// template <typename S>
// RMAGINE_INLINE_FUNCTION
// void Matrix4x4<S>::setRotation(const Quaternion& q)
// {
//     Matrix3x3<S> R;
//     R = q;
//     setRotation(R);
// }

// template <typename S>
// RMAGINE_INLINE_FUNCTION
// void Matrix4x4<S>::setRotation(const EulerAngles& e)
// {
//     Matrix3x3<S> R;
//     R = e;
//     setRotation(R);
// }

template <typename S>
RMAGINE_INLINE_FUNCTION
Vector3<S> Matrix4x4<S>::translation() const
{
    return {at(0,3), at(1,3), at(2,3)};
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::setTranslation(const Vector3<S>& t)
{
    at(0,3) = t.x;
    at(1,3) = t.y;
    at(2,3) = t.z;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::transpose() const 
{
    Matrix4x4<S> ret;

    ret(0,0) = at(0,0);
    ret(0,1) = at(1,0);
    ret(0,2) = at(2,0);
    ret(0,3) = at(3,0);
    
    ret(1,0) = at(0,1);
    ret(1,1) = at(1,1);
    ret(1,2) = at(2,1);
    ret(1,3) = at(3,1);

    ret(2,0) = at(0,2);
    ret(2,1) = at(1,2);
    ret(2,2) = at(2,2);
    ret(2,3) = at(3,2);

    ret(3,0) = at(0,3);
    ret(3,1) = at(1,3);
    ret(3,2) = at(2,3);
    ret(3,3) = at(3,3);

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::T() const 
{
    return transpose();
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix4x4<S>::trace() const
{
    return at(0,0) + at(1,1) + at(2,2) + at(3,3);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix4x4<S>::det() const 
{
    // TODO: check
    const S A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const S A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const S A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const S A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const S A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const S A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const S A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const S A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const S A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const S A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const S A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const S A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const S A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const S A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const S A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const S A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const S A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const S A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    return  at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
            - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
            + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
            - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::inv() const 
{
    // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    // answer of willnode at Jun 8 '17 at 23:09

    const S A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const S A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const S A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const S A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const S A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const S A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const S A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const S A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const S A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const S A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const S A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const S A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const S A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const S A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const S A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const S A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const S A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const S A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    S det_  = at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
                - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
                + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
                - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 ) ;

    // inv det
    det_ = 1.0f / det_;

    Matrix4x4<S> ret;
    ret(0,0) = det_ *   ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 );
    ret(0,1) = det_ * - ( at(0,1) * A2323 - at(0,2) * A1323 + at(0,3) * A1223 );
    ret(0,2) = det_ *   ( at(0,1) * A2313 - at(0,2) * A1313 + at(0,3) * A1213 );
    ret(0,3) = det_ * - ( at(0,1) * A2312 - at(0,2) * A1312 + at(0,3) * A1212 );
    ret(1,0) = det_ * - ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 );
    ret(1,1) = det_ *   ( at(0,0) * A2323 - at(0,2) * A0323 + at(0,3) * A0223 );
    ret(1,2) = det_ * - ( at(0,0) * A2313 - at(0,2) * A0313 + at(0,3) * A0213 );
    ret(1,3) = det_ *   ( at(0,0) * A2312 - at(0,2) * A0312 + at(0,3) * A0212 );
    ret(2,0) = det_ *   ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 );
    ret(2,1) = det_ * - ( at(0,0) * A1323 - at(0,1) * A0323 + at(0,3) * A0123 );
    ret(2,2) = det_ *   ( at(0,0) * A1313 - at(0,1) * A0313 + at(0,3) * A0113 );
    ret(2,3) = det_ * - ( at(0,0) * A1312 - at(0,1) * A0312 + at(0,3) * A0112 );
    ret(3,0) = det_ * - ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );
    ret(3,1) = det_ *   ( at(0,0) * A1223 - at(0,1) * A0223 + at(0,2) * A0123 );
    ret(3,2) = det_ * - ( at(0,0) * A1213 - at(0,1) * A0213 + at(0,2) * A0113 );
    ret(3,3) = det_ *   ( at(0,0) * A1212 - at(0,1) * A0212 + at(0,2) * A0112 );

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::invRigid()
{
    Matrix4x4<S> ret;
    ret.setIdentity();

    Matrix3x3<S> R = rotation();
    Vector3<S> t = translation();

    R.transposeInplace();
    ret.setRotation(R);
    ret.setTranslation(- (R * t) );

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::mult(const S& s) const
{
    Matrix4x4<S> ret;

    ret(0,0) = at(0,0) * s;
    ret(0,1) = at(0,1) * s;
    ret(0,2) = at(0,2) * s;
    ret(0,3) = at(0,3) * s;

    ret(1,0) = at(1,0) * s;
    ret(1,1) = at(1,1) * s;
    ret(1,2) = at(1,2) * s;
    ret(1,3) = at(1,3) * s;

    ret(2,0) = at(2,0) * s;
    ret(2,1) = at(2,1) * s;
    ret(2,2) = at(2,2) * s;
    ret(2,3) = at(2,3) * s;

    ret(3,0) = at(3,0) * s;
    ret(3,1) = at(3,1) * s;
    ret(3,2) = at(3,2) * s;
    ret(3,3) = at(3,3) * s;

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::multInplace(const S& s)
{
    at(0,0) *= s;
    at(0,1) *= s;
    at(0,2) *= s;
    at(0,3) *= s;

    at(1,0) *= s;
    at(1,1) *= s;
    at(1,2) *= s;
    at(1,3) *= s;

    at(2,0) *= s;
    at(2,1) *= s;
    at(2,2) *= s;
    at(2,3) *= s;

    at(3,0) *= s;
    at(3,1) *= s;
    at(3,2) *= s;
    at(3,3) *= s;
}


template <typename S>
RMAGINE_INLINE_FUNCTION
Vector3<S> Matrix4x4<S>::mult(const Vector3<S>& v) const
{
    return {
        at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z + at(0,3),
        at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z + at(1,3),
        at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z + at(2,3)
    };
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::mult(const Matrix4x4<S>& M) const 
{
    Matrix4x4<S> res;
    res.setZeros();

    for (unsigned int row = 0; row < 4; row++) {
        for (unsigned int col = 0; col < 4; col++) {
            for (unsigned int inner = 0; inner < 4; inner++) {
                res(row,col) += at(row,inner) * M(inner,col);
            }
        }
    }

    return res;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix4x4<S> Matrix4x4<S>::div(const S& s) const
{
    Matrix4x4<S> ret;

    ret(0,0) = at(0,0) / s;
    ret(0,1) = at(0,1) / s;
    ret(0,2) = at(0,2) / s;
    ret(0,3) = at(0,3) / s;

    ret(1,0) = at(1,0) / s;
    ret(1,1) = at(1,1) / s;
    ret(1,2) = at(1,2) / s;
    ret(1,3) = at(1,3) / s;

    ret(2,0) = at(2,0) / s;
    ret(2,1) = at(2,1) / s;
    ret(2,2) = at(2,2) / s;
    ret(2,3) = at(2,3) / s;

    ret(3,0) = at(3,0) / s;
    ret(3,1) = at(3,1) / s;
    ret(3,2) = at(3,2) / s;
    ret(3,3) = at(3,3) / s;

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix4x4<S>::divInplace(const S& s)
{
    at(0,0) /= s;
    at(0,1) /= s;
    at(0,2) /= s;
    at(0,3) /= s;

    at(1,0) /= s;
    at(1,1) /= s;
    at(1,2) /= s;
    at(1,3) /= s;

    at(2,0) /= s;
    at(2,1) /= s;
    at(2,2) /= s;
    at(2,3) /= s;

    at(3,0) /= s;
    at(3,1) /= s;
    at(3,2) /= s;
    at(3,3) /= s;
}

// Static Functions

// template <typename S>
// static
// RMAGINE_INLINE_FUNCTION
// void set_identity(Quaternion& q)
// {
//     q.x = 0.0;
//     q.y = 0.0;
//     q.z = 0.0;
//     q.w = 1.0;
// }

template <typename S>
static
RMAGINE_INLINE_FUNCTION
void set_identity(Matrix3x3<S>& M)
{
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;

    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;

    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
}

template <typename S>
static
RMAGINE_INLINE_FUNCTION
void set_identity(Matrix4x4<S>& M)
{
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;
    M[0][3] = 0.0;

    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;
    M[1][3] = 0.0;

    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
    M[2][3] = 0.0;

    M[3][0] = 0.0;
    M[3][1] = 0.0;
    M[3][2] = 0.0;
    M[3][3] = 1.0;
}


} // end namespace rmagine