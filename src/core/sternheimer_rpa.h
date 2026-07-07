#pragma once

#include <complex>

#include "../math/complexmatrix.h"

namespace librpa_int
{

struct SternheimerRpaFrequencyResult
{
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    double qweight = 1.0;
    std::complex<double> trace_pi = {0.0, 0.0};
    std::complex<double> logdet = {0.0, 0.0};
    std::complex<double> integrand = {0.0, 0.0};
    std::complex<double> energy = {0.0, 0.0};
};

ComplexMatrix compute_sternheimer_pi_from_m(const ComplexMatrix &coulomb,
                                            const ComplexMatrix &response_m,
                                            double sqrt_coulomb_threshold);

std::complex<double> compute_rpa_trace_log_integrand(const ComplexMatrix &pi);

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency(const ComplexMatrix &coulomb,
                                                                const ComplexMatrix &response_m,
                                                                int ifreq, double omega,
                                                                double weight, double qweight,
                                                                double sqrt_coulomb_threshold);

}  // namespace librpa_int
