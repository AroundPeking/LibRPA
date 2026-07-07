#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>

#include "../core/sternheimer_rpa.h"
#include "../utils/constants.h"

namespace
{

void require_close(const std::complex<double> &actual, const std::complex<double> &expected,
                   const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

void test_sternheimer_pi_and_trace_log_match_diagonal_reference()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {4.0, 0.0};
    coulomb(1, 1) = {9.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-0.8, 0.0};
    response_m(1, 1) = {-0.9, 0.0};

    const auto pi = librpa_int::compute_sternheimer_pi_from_m(coulomb, response_m, 1e-12);
    require_close(pi(0, 0), {-0.2, 0.0}, 1e-12);
    require_close(pi(1, 1), {-0.1, 0.0}, 1e-12);
    require_close(pi(0, 1), {0.0, 0.0}, 1e-12);
    require_close(pi(1, 0), {0.0, 0.0}, 1e-12);

    const std::complex<double> expected_integrand = std::log(std::complex<double>(1.2, 0.0)) - 0.2 +
                                                    std::log(std::complex<double>(1.1, 0.0)) - 0.1;
    require_close(librpa_int::compute_rpa_trace_log_integrand(pi), expected_integrand, 1e-12);

    const auto result = librpa_int::compute_sternheimer_rpa_frequency(coulomb, response_m, 3, 0.5,
                                                                      0.25, 1.0, 1e-12);
    require_close(result.integrand, expected_integrand, 1e-12);
    require_close(result.energy, expected_integrand * 0.25 / librpa_int::TWO_PI, 1e-12);
}

}  // namespace

int main()
{
    test_sternheimer_pi_and_trace_log_match_diagonal_reference();
    return 0;
}
