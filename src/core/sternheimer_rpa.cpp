#include "sternheimer_rpa.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../io/global_io.h"
#include "../math/lapack_connector.h"
#include "../utils/constants.h"
#include "librpa_enums.h"

namespace librpa_int
{
namespace
{

ComplexMatrix hermitize(const ComplexMatrix &mat)
{
    if (mat.nr != mat.nc)
    {
        throw std::logic_error("Cannot hermitize a non-square matrix");
    }

    ComplexMatrix result(mat);
    for (int i = 0; i != mat.nr; ++i)
    {
        result(i, i) = {result(i, i).real(), 0.0};
        for (int j = i + 1; j != mat.nc; ++j)
        {
            const auto value = 0.5 * (mat(i, j) + std::conj(mat(j, i)));
            result(i, j) = value;
            result(j, i) = std::conj(value);
        }
    }
    return result;
}

ComplexMatrix filtered_inverse_sqrt_hemat(ComplexMatrix mat, const double threshold)
{
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nc;
    const int nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    const int lwork = std::max(1, n * (nb + 1));
    int info = 0;
    std::vector<double> eigenvalues(n);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(std::max(1, 3 * n - 2));

    mat *= -1.0;
    LapackConnector::zheev(jobz, uplo, n, mat, n, eigenvalues.data(), work.data(), lwork,
                           rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("ST-RPA Coulomb eigensolver failed with info=" +
                                 std::to_string(info));
    }

    std::vector<double> inverse_sqrt(n, 0.0);
    for (int i = 0; i != n; ++i)
    {
        eigenvalues[i] = -eigenvalues[i];
        if (eigenvalues[i] > threshold)
        {
            inverse_sqrt[i] = 1.0 / std::sqrt(eigenvalues[i]);
        }
        else if (eigenvalues[i] < -1.0e-12)
        {
            global::lib_printf(LIBRPA_VERBOSE_WARN,
                               "Warning! ST-RPA filters negative Coulomb eigenvalue: # %d ev = "
                               "%.12e threshold = %.12e\n",
                               i, eigenvalues[i], threshold);
        }
    }

    ComplexMatrix evconj = transpose(mat, true);
    for (int i = 0; i != mat.nr; ++i)
    {
        for (int j = 0; j != mat.nc; ++j)
        {
            evconj.c[i * mat.nc + j] *= inverse_sqrt[i];
        }
    }
    return mat * evconj;
}

}  // namespace

ComplexMatrix compute_sternheimer_pi_from_m(const ComplexMatrix &coulomb,
                                            const ComplexMatrix &response_m,
                                            const double sqrt_coulomb_threshold)
{
    if (coulomb.nr != coulomb.nc || response_m.nr != response_m.nc || coulomb.nr != response_m.nr)
    {
        throw std::logic_error("ST-RPA Coulomb and response matrices must be square and same-size");
    }

    auto coulomb_herm = hermitize(coulomb);
    auto response_herm = hermitize(response_m);
    const auto inv_sqrt_coulomb = filtered_inverse_sqrt_hemat(coulomb_herm, sqrt_coulomb_threshold);
    auto pi = inv_sqrt_coulomb * response_herm * inv_sqrt_coulomb;
    return hermitize(pi);
}

std::complex<double> compute_rpa_trace_log_integrand(const ComplexMatrix &pi)
{
    if (pi.nr != pi.nc)
    {
        throw std::logic_error("RPA trace-log response matrix must be square");
    }

    ComplexMatrix identity_minus_pi(pi.nr, pi.nc);
    identity_minus_pi.set_as_identity_matrix();
    identity_minus_pi -= pi;

    std::complex<double> det_for_rpa(1.0, 0.0);
    std::vector<int> ipiv(pi.nr);
    int info_lu = 0;
    LapackConnector::zgetrf(pi.nr, pi.nc, identity_minus_pi, pi.nr, ipiv.data(), &info_lu);
    if (info_lu != 0)
    {
        throw std::runtime_error("RPA trace-log LU factorization failed with info=" +
                                 std::to_string(info_lu));
    }

    for (int ib = 0; ib != pi.nr; ++ib)
    {
        if (ipiv[ib] != ib + 1)
        {
            det_for_rpa = -det_for_rpa * identity_minus_pi(ib, ib);
        }
        else
        {
            det_for_rpa *= identity_minus_pi(ib, ib);
        }
    }

    return std::log(det_for_rpa) + trace(pi);
}

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency(const ComplexMatrix &coulomb,
                                                                const ComplexMatrix &response_m,
                                                                const int ifreq, const double omega,
                                                                const double weight,
                                                                const double qweight,
                                                                const double sqrt_coulomb_threshold)
{
    const auto pi = compute_sternheimer_pi_from_m(coulomb, response_m, sqrt_coulomb_threshold);
    const auto integrand = compute_rpa_trace_log_integrand(pi);

    SternheimerRpaFrequencyResult result;
    result.ifreq = ifreq;
    result.omega = omega;
    result.weight = weight;
    result.qweight = qweight;
    result.trace_pi = trace(pi);
    result.logdet = integrand - result.trace_pi;
    result.integrand = integrand;
    result.energy = integrand * weight * qweight / TWO_PI;
    return result;
}

}  // namespace librpa_int
