#include <cmath>
#include <complex>
#include <stdexcept>

#include "../core/tetrahedron_quadrature.h"
#include "../core/thermal_occupation.h"
#include "../core/timefreq.h"

namespace
{
using librpa_int::build_periodic_tetrahedron_quadrature;
using librpa_int::fermi_dirac_derivative;
using librpa_int::fermi_dirac_occupation;
using librpa_int::finite_temperature_tetrahedron_kernel;
using librpa_int::make_fermi_dirac_reference;
using librpa_int::TFGrids;
using librpa_int::thermal_green_amplitude;
using librpa_int::Vector3_Order;

void require_near(const std::complex<double> actual, const std::complex<double> expected,
                  const double tolerance, const char *message)
{
    if (std::abs(actual - expected) > tolerance) throw std::runtime_error(message);
}

std::complex<double> transform_green_pair(const TFGrids &tfg, const std::size_t ifreq,
                                          const double energy_n, const double energy_m,
                                          const double kbt)
{
    std::complex<double> result = 0.0;
    for (std::size_t itime = 0; itime != tfg.get_n_time_grids(); ++itime)
    {
        const double tau = tfg.get_time_nodes()[itime];
        const double g_m_positive = thermal_green_amplitude(energy_m, tau, kbt);
        const double g_n_negative = -thermal_green_amplitude(energy_n, -tau, kbt);
        result += tfg.get_time_to_frequency_factor(ifreq, itime) * g_m_positive * g_n_negative;
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
        const std::complex<double> denominator(energy_n - energy_m, tfg.get_freq_nodes()[ifreq]);
        const auto expected = (occupation_n - occupation_m) / denominator;
        require_near(transform_green_pair(tfg, ifreq, energy_n, energy_m, kbt), expected, 2.0e-8,
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
    require_near(transform_green_pair(tfg, 0, energy, energy, kbt), expected_static, 1.0e-13,
                 "same-level static bubble is not -beta f(1-f)");
    require_near(expected_static, fermi_dirac_derivative(energy, kbt), 1.0e-13,
                 "same-level static bubble differs from the FD derivative");
    for (std::size_t ifreq = 1; ifreq != nfreq; ++ifreq)
    {
        require_near(transform_green_pair(tfg, ifreq, energy, energy, kbt), 0.0, 1.0e-13,
                     "same-level bubble did not vanish at nonzero Matsubara frequency");
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
            gg_response += transform_green_pair(tfg, ifreq, energy_k, energy_kq, kbt) /
                           static_cast<double>(nk);

            const double occupation_k = fermi_dirac_occupation(energy_k, kbt);
            const double occupation_kq = fermi_dirac_occupation(energy_kq, kbt);
            const std::complex<double> denominator(energy_k - energy_kq, frequency);
            if (std::abs(denominator) < 1.0e-14)
                direct_response += fermi_dirac_derivative(energy_k, kbt) / nk;
            else
                direct_response +=
                    (occupation_k - occupation_kq) / denominator / static_cast<double>(nk);
        }
        require_near(gg_response, direct_response, tolerance,
                     "finite-q same-band GG response differs from direct band summation");
    }
}

void test_tetrahedron_kernel_has_the_finite_temperature_equal_energy_limits()
{
    constexpr double beta = 20.0;
    constexpr double chemical_potential = 0.15;
    constexpr double energy = 0.20;
    const auto reference = make_fermi_dirac_reference(1.0 / beta, chemical_potential, 2.0, 1e-12);

    const auto static_value = finite_temperature_tetrahedron_kernel(energy, energy, 0.0, reference);
    const auto expected = fermi_dirac_derivative(energy - chemical_potential, reference.kbt_ha);
    require_near(static_value, {expected, 0.0}, 1.0e-14,
                 "tetrahedron kernel missed the static FD derivative limit");

    const auto dynamic_value =
        finite_temperature_tetrahedron_kernel(energy, energy, 0.37, reference);
    require_near(dynamic_value, {0.0, 0.0}, 1.0e-14,
                 "tetrahedron kernel has a spurious dynamic equal-energy contribution");
}

void test_periodic_tetrahedron_quadrature_preserves_constant_complete_integrand()
{
    const auto points = build_periodic_tetrahedron_quadrature(Vector3_Order<int>{2, 2, 2});
    std::complex<double> integral = 0.0;
    for (const auto &point : points) integral += point.weight * std::complex<double>{1.25, -0.75};
    require_near(integral, {1.25, -0.75}, 1.0e-14,
                 "periodic tetrahedron quadrature did not preserve a constant integrand");
}

void test_periodic_tetrahedron_quadrature_respects_explicit_grid_order()
{
    const Vector3_Order<int> period{2, 3, 2};
    const std::vector<Vector3_Order<double>> lexicographic{
        {0.0, 0.0, 0.0},       {0.0, 0.0, 0.5},       {0.0, 1.0 / 3.0, 0.0}, {0.0, 1.0 / 3.0, 0.5},
        {0.0, 2.0 / 3.0, 0.0}, {0.0, 2.0 / 3.0, 0.5}, {0.5, 0.0, 0.0},       {0.5, 0.0, 0.5},
        {0.5, 1.0 / 3.0, 0.0}, {0.5, 1.0 / 3.0, 0.5}, {0.5, 2.0 / 3.0, 0.0}, {0.5, 2.0 / 3.0, 0.5},
    };
    const std::vector<std::size_t> permutation{7, 2, 10, 0, 11, 4, 8, 1, 9, 3, 6, 5};
    std::vector<Vector3_Order<double>> reordered;
    reordered.reserve(permutation.size());
    for (const auto index : permutation) reordered.push_back(lexicographic[index]);

    const std::vector<std::complex<double>> lexicographic_values{
        {0.2, -0.1}, {0.4, 0.3}, {-0.7, 0.5}, {1.1, -0.6}, {0.8, 0.9},   {-0.2, -0.4},
        {0.5, -0.8}, {1.3, 0.2}, {-1.1, 0.7}, {0.6, 0.4},  {-0.3, -0.9}, {0.9, -0.2},
    };
    std::vector<std::complex<double>> reordered_values;
    reordered_values.reserve(permutation.size());
    for (const auto index : permutation) reordered_values.push_back(lexicographic_values[index]);

    const auto integrate = [](const auto &points, const auto &values)
    {
        std::complex<double> integral = 0.0;
        for (const auto &point : points)
        {
            std::complex<double> value = 0.0;
            for (int vertex = 0; vertex != 4; ++vertex)
                value += point.barycentric[vertex] * values[point.vertices[vertex]];
            integral += point.weight * value;
        }
        return integral;
    };
    const auto lexicographic_integral =
        integrate(build_periodic_tetrahedron_quadrature(period), lexicographic_values);
    const auto reordered_integral =
        integrate(build_periodic_tetrahedron_quadrature(period, reordered), reordered_values);
    require_near(reordered_integral, lexicographic_integral, 1.0e-14,
                 "tetrahedron quadrature ignored the supplied k-point order");
}

void test_periodic_tetrahedron_quadrature_rejects_degenerate_3d_mesh()
{
    bool rejected = false;
    try
    {
        (void)build_periodic_tetrahedron_quadrature(Vector3_Order<int>{2, 1, 2});
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error(
            "three-dimensional tetrahedron quadrature accepted a degenerate axis");
}
}  // namespace

int main()
{
    test_two_level_green_product_matches_adler_wiser();
    test_same_level_static_and_dynamic_limits();
    test_finite_q_same_band_response_needs_no_extra_drude_term();
    test_tetrahedron_kernel_has_the_finite_temperature_equal_energy_limits();
    test_periodic_tetrahedron_quadrature_preserves_constant_complete_integrand();
    test_periodic_tetrahedron_quadrature_respects_explicit_grid_order();
    test_periodic_tetrahedron_quadrature_rejects_degenerate_3d_mesh();
    return 0;
}
