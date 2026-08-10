#pragma once

#include "lapack_connector.h"
#include "matrix_m.h"

#include <cstddef>
#include <stdexcept>

namespace librpa_int {

inline int shrink_q_owner(const std::size_t iq, const int nprocs)
{
    if (nprocs <= 0)
        throw std::invalid_argument("shrink q-point owner requires a positive process count");
    return static_cast<int>(iq % static_cast<std::size_t>(nprocs));
}

template <typename T>
void validate_shrink_local_blas_shapes(const matrix_m<T> &u,
                                       const matrix_m<T> &chi0,
                                       const matrix_m<T> &u_chi0,
                                       const matrix_m<T> &chi0_small)
{
    if (!u.is_col_major() || !chi0.is_col_major() ||
        !u_chi0.is_col_major() || !chi0_small.is_col_major())
        throw std::invalid_argument("local shrink BLAS matrices must be column major");
    if (chi0.nr() != chi0.nc() || u.nc() != chi0.nr() ||
        u_chi0.nr() != u.nr() || u_chi0.nc() != chi0.nc() ||
        chi0_small.nr() != u.nr() || chi0_small.nc() != u.nr())
        throw std::invalid_argument("incompatible local shrink BLAS matrix dimensions");
}

template <typename T>
void shrink_local_blas_left(const matrix_m<T> &u, const matrix_m<T> &chi0,
                            matrix_m<T> &u_chi0)
{
    LapackConnector::gemm_f(
        'N', 'N', u.nr(), chi0.nc(), u.nc(), T(1.0), u.ptr(), u.nr(),
        chi0.ptr(), chi0.nr(), T(0.0), u_chi0.ptr(), u_chi0.nr());
}

template <typename T>
void shrink_local_blas_right(const matrix_m<T> &u,
                             const matrix_m<T> &u_chi0,
                             matrix_m<T> &chi0_small)
{
    LapackConnector::gemm_f(
        'N', 'C', u.nr(), u.nr(), u.nc(), T(1.0), u_chi0.ptr(),
        u_chi0.nr(), u.ptr(), u.nr(), T(0.0), chi0_small.ptr(),
        chi0_small.nr());
}

template <typename T>
void shrink_local_blas(const matrix_m<T> &u, const matrix_m<T> &chi0,
                       matrix_m<T> &u_chi0, matrix_m<T> &chi0_small)
{
    validate_shrink_local_blas_shapes(u, chi0, u_chi0, chi0_small);
    shrink_local_blas_left(u, chi0, u_chi0);
    shrink_local_blas_right(u, u_chi0, chi0_small);
}

} // namespace librpa_int
