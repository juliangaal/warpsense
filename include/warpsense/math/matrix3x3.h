#pragma once
#include "vector3.h"

namespace rmagine
{
    /**
 * @brief Matrix3x3<S> class
 * 
 * Same order than Eigen::Matrix3f default -> Can be reinterpret-casted or mapped.
 * 
 * Storage order ()-operator 
 * (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), ... 
 * 
 * Storage order []-operator
 * [0][0], [0][1], [0][2], [1][0], [1][1], [1][2], ...
 * 
 */
template <typename S>
struct Matrix3x3 {
    // DASA
    S data[3][3];
    
    // ACCESS
    RMAGINE_INLINE_FUNCTION
    S & at(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    volatile S & at(unsigned int i, unsigned int j) volatile;

    RMAGINE_INLINE_FUNCTION
    S at(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    S at(unsigned int i, unsigned int j) volatile const;

    RMAGINE_INLINE_FUNCTION
    S & operator()(unsigned int i, unsigned int j);

    RMAGINE_INLINE_FUNCTION
    volatile S& operator()(unsigned int i, unsigned int j) volatile;

    RMAGINE_INLINE_FUNCTION
    S operator()(unsigned int i, unsigned int j) const;

    RMAGINE_INLINE_FUNCTION
    S operator()(unsigned int i, unsigned int j) volatile const;

    RMAGINE_INLINE_FUNCTION
    S* operator[](const unsigned int i);

    RMAGINE_INLINE_FUNCTION
    const S* operator[](const unsigned int i) const;

    // FUNCSIONS
    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    RMAGINE_INLINE_FUNCTION
    void setOnes();

    // RMAGINE_INLINE_FUNCTION
    // void set(const Quaternion& q);

    // RMAGINE_INLINE_FUNCTION
    // void set(const EulerAngles& e);

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> transpose() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> T() const;

    RMAGINE_INLINE_FUNCTION
    void transposeInplace();

    RMAGINE_INLINE_FUNCTION
    S trace() const;

    RMAGINE_INLINE_FUNCTION
    S det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> inv() const;

    /**
     * @brief Assuming Matrix3x3<S> to be a rotation matrix. then M.inv = M.transpose
     * 
     * @return Matrix3x3<S> 
     */
    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> invRigid() const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> mult(const S & s) const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> div(const S & s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const S & s);

    RMAGINE_INLINE_FUNCTION
    void multInplace(const S & s);

    RMAGINE_INLINE_FUNCTION
    Vector3<S> mult(const Vector3<S>& p) const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> mult(const Matrix3x3<S>&  M) const;

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> add(const Matrix3x3<S>&  M) const;

    RMAGINE_INLINE_FUNCTION
    void addInplace(const Matrix3x3<S>&  M);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile Matrix3x3<S>&  M) volatile;

    // OPERASORS
    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> operator*(const S & s) const
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S>&  operator*=(const S & s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3<S> operator*(const Vector3<S>& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> operator*(const Matrix3x3<S>&  M) const 
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> operator/(const S& s) const
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S>&  operator/=(const S& s)
    {
        divInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> operator+(const Matrix3x3<S>& M) const
    {
        return add(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S>& operator+=(const Matrix3x3<S>& M)
    {
        addInplace(M);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    volatile Matrix3x3<S>& operator+=(volatile Matrix3x3<S>& M) volatile
    {
        addInplace(M);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix3x3<S> operator~() const
    {
        return inv();
    }

    // RMAGINE_INLINE_FUNCTION
    // void operator=(const Quaternion& q)
    // {
    //     set(q);
    // }

    // RMAGINE_INLINE_FUNCTION
    // void operator=(const EulerAngles& e)
    // {
    //     set(e);
    // }
};

using Matrix3x3i = Matrix3x3<int>;
using Matrix3x3f = Matrix3x3<float>;
using Matrix3x3d = Matrix3x3<double>;

template <typename S>
RMAGINE_INLINE_FUNCTION
S& Matrix3x3<S>::at(unsigned int i, unsigned int j)
{
    return data[j][i];
}

template <typename S>
RMAGINE_INLINE_FUNCTION
volatile S& Matrix3x3<S>::at(unsigned int i, unsigned int j) volatile
{
    return data[j][i];
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::at(unsigned int i, unsigned int j) const
{
    return data[j][i];
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::at(unsigned int i, unsigned int j) volatile const
{
    return data[j][i];
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S& Matrix3x3<S>::operator()(unsigned int i, unsigned int j)
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
volatile S& Matrix3x3<S>::operator()(unsigned int i, unsigned int j) volatile
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::operator()(unsigned int i, unsigned int j) const
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::operator()(unsigned int i, unsigned int j) volatile const
{
    return at(i,j);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S* Matrix3x3<S>::operator[](const unsigned int i) 
{
    return data[i];
};

template <typename S>
RMAGINE_INLINE_FUNCTION
const S* Matrix3x3<S>::operator[](const unsigned int i) const 
{
    return data[i];
};

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::setIdentity()
{
    at(0,0) = 1.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 1.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 1.0f;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::setZeros()
{
    at(0,0) = 0.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 0.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 0.0f;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::setOnes()
{
    at(0,0) = 1.0f;
    at(0,1) = 1.0f;
    at(0,2) = 1.0f;
    at(1,0) = 1.0f;
    at(1,1) = 1.0f;
    at(1,2) = 1.0f;
    at(2,0) = 1.0f;
    at(2,1) = 1.0f;
    at(2,2) = 1.0f;
}

// template <typename S>
//RMAGINE_INLINE_FUNCTION
// void Matrix3x3<S>::set(const Quaternion& q)
// {
//     // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    
//     // inhomogeneous expression
//     // only for unit quaternions

//     at(0,0) = 2.0f * (q.w * q.w + q.x * q.x) - 1.0f;
//     at(0,1) = 2.0f * (q.x * q.y - q.w * q.z);
//     at(0,2) = 2.0f * (q.x * q.z + q.w * q.y);
//     at(1,0) = 2.0f * (q.x * q.y + q.w * q.z);
//     at(1,1) = 2.0f * (q.w * q.w + q.y * q.y) - 1.0f;
//     at(1,2) = 2.0f * (q.y * q.z - q.w * q.x);
//     at(2,0) = 2.0f * (q.x * q.z - q.w * q.y);
//     at(2,1) = 2.0f * (q.y * q.z + q.w * q.x);
//     at(2,2) = 2.0f * (q.w * q.w + q.z * q.z) - 1.0f;

//     // SODO:
//     // homogeneous expession


//     // SESSED
// }

// template <typename S>
//RMAGINE_INLINE_FUNCTION
// void Matrix3x3<S>::set(const EulerAngles& e)
// {
//     // Wrong?
//     // SODO check
//     // 1. test: correct

//     const S cA = cosf(e.roll);
//     const S sA = sinf(e.roll);
//     const S cB = cosf(e.pitch);
//     const S sB = sinf(e.pitch);
//     const S cC = cosf(e.yaw);
//     const S sC = sinf(e.yaw);

//     at(0,0) =  cB * cC;
//     at(0,1) = -cB * sC;
//     at(0,2) =  sB;
   
//     at(1,0) =  sA * sB * cC + cA * sC;
//     at(1,1) = -sA * sB * sC + cA * cC;
//     at(1,2) = -sA * cB;
    
//     at(2,0) = -cA * sB * cC + sA * sC;
//     at(2,1) =  cA * sB * sC + sA * cC;
//     at(2,2) =  cA * cB;
// }

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::transpose() const 
{
    Matrix3x3<S> ret;

    ret(0,0) = at(0,0);
    ret(0,1) = at(1,0);
    ret(0,2) = at(2,0);
    
    ret(1,0) = at(0,1);
    ret(1,1) = at(1,1);
    ret(1,2) = at(2,1);

    ret(2,0) = at(0,2);
    ret(2,1) = at(1,2);
    ret(2,2) = at(2,2);

    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::T() const 
{
    return transpose();
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::transposeInplace()
{
    // use only one S as additional memory
    S swap_mem;
    // can we do this without additional memory?

    swap_mem = at(0,1);
    at(0,1) = at(1,0);
    at(1,0) = swap_mem;

    swap_mem = at(0,2);
    at(0,2) = at(2,0);
    at(2,0) = swap_mem;

    swap_mem = at(1,2);
    at(1,2) = at(2,1);
    at(2,1) = swap_mem;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::trace() const
{
    return at(0, 0) + at(1, 1) + at(2, 2);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
S Matrix3x3<S>::det() const
{
    return  at(0, 0) * (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) -
            at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
            at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::inv() const
{
    Matrix3x3<S> ret;

    const S invdet = 1 / det();

    ret(0, 0) = (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) * invdet;
    ret(0, 1) = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * invdet;
    ret(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * invdet;
    ret(1, 0) = (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * invdet;
    ret(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * invdet;
    ret(1, 2) = (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * invdet;
    ret(2, 0) = (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * invdet;
    ret(2, 1) = (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * invdet;
    ret(2, 2) = (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * invdet;

    return ret;
}

/**
    * @brief Assuming Matrix3x3<S> to be a rotation matrix. then M.inv = M.transpose
    * 
    * @return Matrix3x3<S> 
    */
template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::invRigid() const 
{
    return S();
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::mult(const S& s) const
{
    Matrix3x3<S> ret;
    ret(0,0) = at(0,0) * s;
    ret(0,1) = at(0,1) * s;
    ret(0,2) = at(0,2) * s;
    ret(1,0) = at(1,0) * s;
    ret(1,1) = at(1,1) * s;
    ret(1,2) = at(1,2) * s;
    ret(2,0) = at(2,0) * s;
    ret(2,1) = at(2,1) * s;
    ret(2,2) = at(2,2) * s;
    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::div(const S& s) const
{
    Matrix3x3<S> ret;
    ret(0,0) = at(0,0) / s;
    ret(0,1) = at(0,1) / s;
    ret(0,2) = at(0,2) / s;
    ret(1,0) = at(1,0) / s;
    ret(1,1) = at(1,1) / s;
    ret(1,2) = at(1,2) / s;
    ret(2,0) = at(2,0) / s;
    ret(2,1) = at(2,1) / s;
    ret(2,2) = at(2,2) / s;
    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::multInplace(const S& s)
{
    at(0,0) *= s;
    at(0,1) *= s;
    at(0,2) *= s;
    at(1,0) *= s;
    at(1,1) *= s;
    at(1,2) *= s;
    at(2,0) *= s;
    at(2,1) *= s;
    at(2,2) *= s;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::divInplace(const S& s)
{
    at(0,0) /= s;
    at(0,1) /= s;
    at(0,2) /= s;
    at(1,0) /= s;
    at(1,1) /= s;
    at(1,2) /= s;
    at(2,0) /= s;
    at(2,1) /= s;
    at(2,2) /= s;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Vector3<S> Matrix3x3<S>::mult(const Vector3<S>& p) const
{
    return {
        at(0,0) * p.x + at(0,1) * p.y + at(0,2) * p.z, 
        at(1,0) * p.x + at(1,1) * p.y + at(1,2) * p.z, 
        at(2,0) * p.x + at(2,1) * p.y + at(2,2) * p.z
    };
}

template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::mult(const Matrix3x3<S>& M) const
{
    Matrix3x3<S> res;
    res.setZeros();
    for (unsigned int row = 0; row < 3; row++) {
        for (unsigned int col = 0; col < 3; col++) {
            for (unsigned int inner = 0; inner < 3; inner++) {
                res(row,col) += at(row, inner) * M(inner, col);
            }
        }
    }
    return res;
}


template <typename S>
RMAGINE_INLINE_FUNCTION
Matrix3x3<S> Matrix3x3<S>::add(const Matrix3x3<S>& M) const
{
    Matrix3x3<S> ret;
    ret(0,0) = at(0,0) + M(0,0);
    ret(0,1) = at(0,1) + M(0,1);
    ret(0,2) = at(0,2) + M(0,2);
    ret(1,0) = at(1,0) + M(1,0);
    ret(1,1) = at(1,1) + M(1,1);
    ret(1,2) = at(1,2) + M(1,2);
    ret(2,0) = at(2,0) + M(2,0);
    ret(2,1) = at(2,1) + M(2,1);
    ret(2,2) = at(2,2) + M(2,2);
    return ret;
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::addInplace(const Matrix3x3<S>& M)
{
    at(0,0) += M(0,0);
    at(0,1) += M(0,1);
    at(0,2) += M(0,2);
    at(1,0) += M(1,0);
    at(1,1) += M(1,1);
    at(1,2) += M(1,2);
    at(2,0) += M(2,0);
    at(2,1) += M(2,1);
    at(2,2) += M(2,2);
}

template <typename S>
RMAGINE_INLINE_FUNCTION
void Matrix3x3<S>::addInplace(volatile Matrix3x3<S>& M) volatile
{
    at(0,0) += M(0,0);
    at(0,1) += M(0,1);
    at(0,2) += M(0,2);
    at(1,0) += M(1,0);
    at(1,1) += M(1,1);
    at(1,2) += M(1,2);
    at(2,0) += M(2,0);
    at(2,1) += M(2,1);
    at(2,2) += M(2,2);
}

} // end namespace rmagine