#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

void test_sternheimer_headwing_frequency_uses_one_based_response_labels()
{
    assert(librpa_int::sternheimer_headwing_frequency_index(1, 12) == 0);
    assert(librpa_int::sternheimer_headwing_frequency_index(12, 12) == 11);
    for (const int invalid : {0, 13})
    {
        bool threw = false;
        try
        {
            (void)librpa_int::sternheimer_headwing_frequency_index(invalid, 12);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        assert(threw);
    }
}

void test_sternheimer_head_only_replaces_gamma_head()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-3.6, 0.0};
    response_m(1, 1) = {-0.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "head_only";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.1, 0.0};
    headwing.head(1, 1) = {-0.1, 0.0};
    headwing.head(2, 2) = {-0.1, 0.0};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);
    const auto expected = std::log(std::complex<double>(1.1, 0.0)) - 0.1 +
                          std::log(std::complex<double>(1.2, 0.0)) - 0.2;
    require_close(result.integrand, expected, 1e-12);
}

void test_sternheimer_qavg_uses_analytic_head_and_wing()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-3.6, 0.0};
    response_m(1, 1) = {-0.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "qavg";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.1, 0.0};
    headwing.head(1, 1) = {-0.1, 0.0};
    headwing.head(2, 2) = {-0.1, 0.0};
    headwing.wing_mu = librpa_int::ComplexMatrix(2, 3);
    headwing.wing_mu(1, 0) = {0.025, 0.0};
    headwing.directions = {{{1.0, 0.0, 0.0}, 1.0}};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);
    const double schur = 1.1 - 0.05 * 0.05 / 1.2;
    const auto expected = std::complex<double>(-0.3, 0.0) +
                          std::log(std::complex<double>(1.2, 0.0)) +
                          std::log(std::complex<double>(schur, 0.0));
    require_close(result.integrand, expected, 1e-12);
}

}  // namespace

int main()
{
    test_sternheimer_pi_and_trace_log_match_diagonal_reference();
    test_sternheimer_headwing_frequency_uses_one_based_response_labels();
    test_sternheimer_head_only_replaces_gamma_head();
    test_sternheimer_qavg_uses_analytic_head_and_wing();
    return 0;
}
