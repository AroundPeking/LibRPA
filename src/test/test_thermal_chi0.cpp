#include "../core/thermal_occupation.h"
#include "../core/timefreq.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace
{
using librpa_int::TFGrids;
using librpa_int::fermi_dirac_derivative;
using librpa_int::fermi_dirac_occupation;
using librpa_int::thermal_green_amplitude;

void require_near(const std::complex<double> actual,
                  const std::complex<double> expected,
                  const double tolerance,
                  const char *message)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

std::complex<double> transform_green_pair(const TFGrids &tfg,
                                          const std::size_t ifreq,
                                          const double energy_n,
                                          const double energy_m,
                                          const double kbt)
{
    std::complex<double> result = 0.0;
    for (std::size_t itime = 0; itime != tfg.get_n_time_grids(); ++itime)
    {
        const double tau = tfg.get_time_nodes()[itime];
        const double g_m_positive = thermal_green_amplitude(energy_m, tau, kbt);
        const double g_n_negative = -thermal_green_amplitude(energy_n, -tau, kbt);
        result += tfg.get_time_to_frequency_factor(ifreq, itime)
                  * g_m_positive * g_n_negative;
    }
    return result;
}

void test_two_level_green_product_matches_adler_wiser()
{
    constexpr std::size_t nfreq = 3;
    constexpr std::size_t ntau = 4096;
    constexpr double kbt = 0.5;
    constexpr double beta = 1.0 / kbt;
    constexpr double energy_n = -0.2;
    constexpr double energy_m = 0.2;

    TFGrids tfg(nfreq);
    tfg.generate_finite_beta_matsubara(ntau, beta);
    const double occupation_n = fermi_dirac_occupation(energy_n, kbt);
    const double occupation_m = fermi_dirac_occupation(energy_m, kbt);
    for (std::size_t ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        const std::complex<double> denominator(
            energy_n - energy_m, tfg.get_freq_nodes()[ifreq]);
        const auto expected = (occupation_n - occupation_m) / denominator;
        require_near(transform_green_pair(tfg, ifreq, energy_n, energy_m, kbt),
                     expected, 2.0e-8,
                     "finite-temperature GG transform differs from Adler-Wiser");
    }
}

void test_same_level_static_and_dynamic_limits()
{
    constexpr std::size_t nfreq = 4;
    constexpr std::size_t ntau = 64;
    constexpr double kbt = 0.25;
    constexpr double beta = 1.0 / kbt;
    constexpr double energy = 0.13;

    TFGrids tfg(nfreq);
    tfg.generate_finite_beta_matsubara(ntau, beta);
    const double occupation = fermi_dirac_occupation(energy, kbt);
    const double expected_static = -beta * occupation * (1.0 - occupation);
    require_near(transform_green_pair(tfg, 0, energy, energy, kbt), expected_static,
                 1.0e-13, "same-level static bubble is not -beta f(1-f)");
    require_near(expected_static, fermi_dirac_derivative(energy, kbt), 1.0e-13,
                 "same-level static bubble differs from the FD derivative");
    for (std::size_t ifreq = 1; ifreq != nfreq; ++ifreq)
    {
        require_near(transform_green_pair(tfg, ifreq, energy, energy, kbt), 0.0,
                     1.0e-13, "same-level bubble did not vanish at nonzero Matsubara frequency");
    }
}

void test_finite_q_same_band_response_needs_no_extra_drude_term()
{
    constexpr int nk = 64;
    constexpr int q_index = 3;
    constexpr std::size_t nfreq = 3;
    constexpr std::size_t ntau = 4096;
    constexpr double beta = 3.0;
    constexpr double kbt = 1.0 / beta;
    constexpr double hopping = 1.0;
    constexpr double tolerance = 2.0e-9;
    const double two_pi = 2.0 * std::acos(-1.0);

    TFGrids tfg(nfreq);
    tfg.generate_finite_beta_matsubara(ntau, beta);
    for (std::size_t ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        std::complex<double> gg_response = 0.0;
        std::complex<double> direct_response = 0.0;
        const double frequency = tfg.get_freq_nodes()[ifreq];
        for (int ik = 0; ik != nk; ++ik)
        {
            const double k = two_pi * ik / nk;
            const double kq = two_pi * ((ik + q_index) % nk) / nk;
            const double energy_k = -2.0 * hopping * std::cos(k);
            const double energy_kq = -2.0 * hopping * std::cos(kq);
            gg_response += transform_green_pair(tfg, ifreq, energy_k, energy_kq, kbt)
                           / static_cast<double>(nk);

            const double occupation_k = fermi_dirac_occupation(energy_k, kbt);
            const double occupation_kq = fermi_dirac_occupation(energy_kq, kbt);
            const std::complex<double> denominator(energy_k - energy_kq, frequency);
            if (std::abs(denominator) < 1.0e-14)
                direct_response += fermi_dirac_derivative(energy_k, kbt) / nk;
            else
                direct_response += (occupation_k - occupation_kq) / denominator
                                   / static_cast<double>(nk);
        }
        require_near(gg_response, direct_response, tolerance,
                     "finite-q same-band GG response differs from direct band summation");
    }
}
} // namespace

int main()
{
    test_two_level_green_product_matches_adler_wiser();
    test_same_level_static_and_dynamic_limits();
    test_finite_q_same_band_response_needs_no_extra_drude_term();
    return 0;
}
