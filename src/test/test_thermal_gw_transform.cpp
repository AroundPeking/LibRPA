#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/thermal_gw_transform.h"
#include "../core/thermal_occupation.h"

namespace
{
using librpa_int::ComplexMatrix;
using librpa_int::ThermalGWTransform;
using Complex = std::complex<double>;
const double PI = std::acos(-1.0);
const Complex I(0.0, 1.0);

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

double error(Complex actual, Complex expected)
{
    require(std::isfinite(actual.real()) && std::isfinite(actual.imag()), "nonfinite result");
    return std::abs(actual - expected);
}

void close(Complex actual, Complex expected, double tolerance, const std::string &message)
{
    if (error(actual, expected) > tolerance)
    {
        std::ostringstream detail;
        detail << std::setprecision(17) << message << ": actual=" << actual
               << " expected=" << expected << " error=" << std::abs(actual - expected);
        throw std::runtime_error(detail.str());
    }
}

void reject(const std::function<void()> &operation)
{
    try
    {
        operation();
    }
    catch (const std::invalid_argument &)
    {
        return;
    }
    throw std::runtime_error("invalid input was not rejected with invalid_argument");
}

std::vector<int> modes(int cutoff)
{
    std::vector<int> result;
    for (int m = -cutoff; m <= cutoff; ++m) result.push_back(m);
    return result;
}

std::vector<double> midpoints(double beta, int count)
{
    std::vector<double> result(count);
    for (int j = 0; j < count; ++j) result[j] = beta * (j + 0.5) / count;
    return result;
}

void check_signed_complex_modes()
{
    const double beta = 4.0;
    const std::vector<double> times{0.5, 1.5, 3.0};
    const std::vector<int> bosons{2, -1, 0}, fermions{1, -2, -1, 0};
    const auto transform =
        ThermalGWTransform::from_quadrature(beta, times, {1.0, 1.2, 1.8}, bosons, fermions);
    require(transform.get_beta_ha_inv() == beta && transform.get_times() == times &&
                transform.get_bosonic_indices() == bosons &&
                transform.get_fermionic_indices() == fermions,
            "metadata/order changed");
    const auto nu = transform.get_bosonic_frequencies_ha();
    const auto omega = transform.get_fermionic_frequencies_ha();
    close(nu[0], PI, 0.0, "bosonic frequency");
    close(nu[1], -PI / 2.0, 0.0, "negative bosonic frequency");
    close(nu[2], 0.0, 0.0, "zero mode");
    close(omega[2], -PI / beta, 0.0, "n=-1 is -pi/beta, not -3pi/beta");
    close(omega[3], PI / beta, 0.0, "n=0 is fermionic, not zero");

    ComplexMatrix samples(3, 2);
    samples(0, 0) = {1.0, 2.0};
    samples(1, 0) = {-0.5, 0.7};
    samples(2, 0) = {0.2, -0.4};
    samples(0, 1) = {-0.3, 0.9};
    samples(1, 1) = {2.0, -0.1};
    samples(2, 1) = {-0.7, 0.5};
    const auto result = transform.apply_bosonic_frequency_to_time(samples);
    require(result.nr == 3 && result.nc == 2, "bosonic output shape");
    // At tau=beta/8 the m=2 phase is -i and m=-1 phase is exp(+i*pi/4).
    for (int col = 0; col < 2; ++col)
        close(result(0, col),
              (-I * samples(0, col) + Complex(1.0, 1.0) / std::sqrt(2.0) * samples(1, col) +
               samples(2, col)) /
                  beta,
              2e-15, "complex non-even inverse/sign/normalization");

    const auto b = transform.copy_bosonic_frequency_to_time();
    const auto f = transform.copy_fermionic_time_to_frequency();
    require(b.nr == 3 && b.nc == 3 && f.nr == 4 && f.nc == 3, "operator shapes");
    close(f(2, 0), std::exp(-I * PI / 8.0), 1e-15, "fermionic positive Fourier sign");
    close(f(3, 2), 1.8 * std::exp(I * 3.0 * PI / 4.0), 1e-15, "integral weight applied once");

    const auto zero = ThermalGWTransform::from_quadrature(beta, times, {1.0, 1.2, 1.8}, {0}, {0});
    ComplexMatrix zero_sample(1, 2);
    zero_sample(0, 0) = {2.0, -3.0};
    zero_sample(0, 1) = {-0.1, 0.7};
    const auto zero_time = zero.apply_bosonic_frequency_to_time(zero_sample);
    for (int j = 0; j < 3; ++j)
        for (int col = 0; col < 2; ++col)
            close(zero_time(j, col), zero_sample(0, col) / beta, 0.0, "zero mode counted once");
}

void check_owned_external_operators()
{
    std::vector<double> times{0.2, 0.8};
    std::vector<int> bosons{-7, 0, 4}, fermions{-2, 3};
    ComplexMatrix b(2, 3), f(2, 2);
    for (int k = 0; k < b.size; ++k) b.c[k] = {0.3 * k - 0.4, 0.2 - 0.1 * k};
    for (int k = 0; k < f.size; ++k) f.c[k] = {-0.7 + 0.1 * k, 0.6 * k + 0.3};
    const ComplexMatrix b_original = b, f_original = f;
    const ThermalGWTransform transform(1.0, times, bosons, fermions, b, f);
    auto copy = transform;
    b.c[0] = 200.0;
    f.c[0] = 300.0;
    times[0] = 0.0;
    bosons[0] = 0;
    fermions[0] = 0;
    auto extracted = copy.copy_bosonic_frequency_to_time();
    extracted.c[0] = 400.0;
    auto extracted_f = copy.copy_fermionic_time_to_frequency();
    extracted_f.c[0] = 500.0;
    close(transform.copy_bosonic_frequency_to_time().c[0], b_original.c[0], 0.0, "input B copy");
    close(copy.copy_bosonic_frequency_to_time().c[0], b_original.c[0], 0.0, "output B copy");
    close(copy.copy_fermionic_time_to_frequency().c[0], f_original.c[0], 0.0,
          "input/output F copy");
    require(copy.get_times()[0] == 0.2 && copy.get_bosonic_indices()[0] == -7 &&
                copy.get_fermionic_indices()[0] == -2,
            "metadata is not owned");

    ComplexMatrix samples(3, 4);
    for (int k = 0; k < samples.size; ++k) samples.c[k] = {0.11 * k, -0.13 * k + 1.0};
    const ComplexMatrix original_samples = samples;
    const auto time = copy.apply_bosonic_frequency_to_time(samples);
    const auto frequency = copy.apply_fermionic_time_to_frequency(time);
    require(time.nr == 2 && time.nc == 4 && frequency.nr == 2 && frequency.nc == 4,
            "rectangular batch shape");
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 4; ++col)
        {
            Complex expected_time = 0.0, expected_frequency = 0.0;
            for (int m = 0; m < 3; ++m) expected_time += b_original(row, m) * samples(m, col);
            for (int j = 0; j < 2; ++j)
                for (int m = 0; m < 3; ++m)
                    expected_frequency += f_original(row, j) * b_original(j, m) * samples(m, col);
            close(time(row, col), expected_time, 2e-14, "external B must be used verbatim");
            close(frequency(row, col), expected_frequency, 2e-14,
                  "external F must be used verbatim");
        }
    for (int k = 0; k < samples.size; ++k)
        close(samples.c[k], original_samples.c[k], 0.0, "apply modified caller input");
}

void check_rectangular_batches()
{
    const std::vector<double> times{0.2, 0.7, 1.6};
    const std::vector<int> bosons{2, 0, -3, -1, 1}, fermions{3, -1, 0, -4};
    ComplexMatrix b(3, 5), f(4, 3);
    for (int row = 0; row < b.nr; ++row)
        for (int source = 0; source < b.nc; ++source)
            b(row, source) = {0.13 + 0.04 * row - 0.07 * source,
                              -0.05 + 0.03 * row * source + 0.02 * source};
    for (int row = 0; row < f.nr; ++row)
        for (int source = 0; source < f.nc; ++source)
            f(row, source) = {-0.21 + 0.09 * row + 0.06 * source,
                              0.17 - 0.02 * row * source - 0.04 * source};
    const ThermalGWTransform transform(2.7, times, bosons, fermions, b, f);
    for (int columns : {1, 2, 7, 32})
    {
        ComplexMatrix frequency_samples(b.nc, columns), time_samples(f.nc, columns);
        for (int row = 0; row < frequency_samples.nr; ++row)
            for (int col = 0; col < columns; ++col)
                frequency_samples(row, col) = {0.4 + 0.11 * row - 0.019 * col,
                                               -0.3 + 0.09 * row * row + 0.013 * row * col};
        for (int row = 0; row < time_samples.nr; ++row)
            for (int col = 0; col < columns; ++col)
                time_samples(row, col) = {-0.7 + 0.17 * row + 0.023 * row * col,
                                          0.2 - 0.07 * row + 0.011 * col * col};
        const ComplexMatrix original_frequency = frequency_samples, original_time = time_samples;
        const auto time = transform.apply_bosonic_frequency_to_time(frequency_samples);
        const auto frequency = transform.apply_fermionic_time_to_frequency(time_samples);
        require(time.nr == b.nr && time.nc == columns && frequency.nr == f.nr &&
                    frequency.nc == columns,
                "rectangular B/F batch shape changed");
        // Independent scalar sums: neither operator nor samples may be conjugated or projected.
        for (int row = 0; row < time.nr; ++row)
            for (int col = 0; col < columns; ++col)
            {
                Complex expected = 0.0;
                for (int source = 0; source < b.nc; ++source)
                    expected += b(row, source) * frequency_samples(source, col);
                close(time(row, col), expected, 2e-14, "rectangular B scalar reference");
            }
        for (int row = 0; row < frequency.nr; ++row)
            for (int col = 0; col < columns; ++col)
            {
                Complex expected = 0.0;
                for (int source = 0; source < f.nc; ++source)
                    expected += f(row, source) * time_samples(source, col);
                close(frequency(row, col), expected, 2e-14, "rectangular F scalar reference");
            }
        for (int k = 0; k < frequency_samples.size; ++k)
            close(frequency_samples.c[k], original_frequency.c[k], 0.0, "B changed samples");
        for (int k = 0; k < time_samples.size; ++k)
            close(time_samples.c[k], original_time.c[k], 0.0, "F changed samples");
    }
    require(transform.get_times() == times && transform.get_bosonic_indices() == bosons &&
                transform.get_fermionic_indices() == fermions,
            "rectangular B/F changed signed metadata order");
}

Complex pole_w_frequency(double nu, double positive_pole, double negative_pole, Complex residue)
{
    return residue * (1.0 / (I * nu - positive_pole) - 1.0 / (I * nu + negative_pole));
}

Complex pole_w_time(double tau, double beta, double positive_pole, double negative_pole,
                    Complex residue)
{
    return -residue *
           (std::exp(-positive_pole * tau) / (-std::expm1(-beta * positive_pole)) +
            std::exp(-negative_pole * (beta - tau)) / (-std::expm1(-beta * negative_pole)));
}

void check_single_pole_w()
{
    double max_error = 0.0;
    const auto indices = modes(4096);
    for (double beta : {0.7, 5.0, 16.0})
    {
        const std::vector<double> times{0.07 * beta, 0.23 * beta, 0.5 * beta, 0.81 * beta,
                                        0.97 * beta};
        const auto transform = ThermalGWTransform::from_quadrature(
            beta, times, std::vector<double>(times.size(), beta / times.size()), indices, {-1, 0});
        ComplexMatrix samples(indices.size(), 2);
        for (int m = 0; m < samples.nr; ++m)
        {
            const double nu = 2.0 * PI * indices[m] / beta;
            samples(m, 0) = pole_w_frequency(nu, 0.7, 0.7, 0.6);
            // Unequal poles and complex residue: neither real nor even in frequency/time.
            samples(m, 1) = pole_w_frequency(nu, 0.7, 1.1, {0.4, -0.3});
        }
        const auto time = transform.apply_bosonic_frequency_to_time(samples);
        for (int j = 0; j < time.nr; ++j)
        {
            max_error =
                std::max(max_error, error(time(j, 0), pole_w_time(times[j], beta, 0.7, 0.7, 0.6)));
            max_error = std::max(
                max_error, error(time(j, 1), pole_w_time(times[j], beta, 0.7, 1.1, {0.4, -0.3})));
        }
        close(samples(4096, 0), -2.0 * 0.6 / 0.7, 1e-15, "analytic static pole");
    }
    std::cout << "  single-pole W max error = " << max_error << '\n';
    require(max_error < 3e-7, "single-pole inverse Fourier model");
}

Complex analytic_sigma(double omega, double xi, double beta, double pole, Complex residue)
{
    const double f = 1.0 / (1.0 + std::exp(beta * xi));
    const double bose = 1.0 / std::expm1(beta * pole);
    return residue *
           ((1.0 + bose - f) / (I * omega - xi - pole) + (bose + f) / (I * omega - xi + pole));
}

void check_fermionic_sigma()
{
    const double beta = 4.3, pole = 0.9;
    const Complex residue(0.6, -0.2);
    const auto times = midpoints(beta, 8192);
    const std::vector<int> fermions{-5, -1, 0, 2, 6};
    const std::vector<double> energies{-1.2, -0.2, 0.0, 0.4, 1.3};
    const auto transform = ThermalGWTransform::from_quadrature(
        beta, times, std::vector<double>(times.size(), beta / times.size()), {0}, fermions);
    ComplexMatrix sigma_time(times.size(), energies.size());
    for (int j = 0; j < sigma_time.nr; ++j)
        for (int col = 0; col < sigma_time.nc; ++col)
        {
            const double xi = energies[col];
            const double g_lib = std::exp(-xi * times[j]) / (1.0 + std::exp(-beta * xi));
            close(librpa_int::thermal_green_amplitude(xi, times[j], 1.0 / beta), g_lib, 2e-15,
                  "existing thermal G positive branch");
            sigma_time(j, col) = g_lib * pole_w_time(times[j], beta, pole, pole, residue);
        }
    const auto sigma = transform.apply_fermionic_time_to_frequency(sigma_time);
    double max_error = 0.0;
    for (int n = 0; n < sigma.nr; ++n)
        for (int col = 0; col < sigma.nc; ++col)
            max_error =
                std::max(max_error,
                         error(sigma(n, col), analytic_sigma((2.0 * fermions[n] + 1.0) * PI / beta,
                                                             energies[col], beta, pole, residue)));
    std::cout << "  analytic-time G*W -> Sigma max error = " << max_error << '\n';
    require(max_error < 2e-7, "fermionic GW sign/normalization/pole model");
}

void check_composed_sigma()
{
    const double beta = 4.3, pole = 0.9, residue = 0.6;
    const auto times = midpoints(beta, 4096);
    const auto bosons = modes(512);
    const std::vector<int> fermions{-3, -1, 0, 2};
    const std::vector<double> energies{-0.6, 0.0, 0.4};
    const auto transform = ThermalGWTransform::from_quadrature(
        beta, times, std::vector<double>(times.size(), beta / times.size()), bosons, fermions);
    ComplexMatrix w_frequency(bosons.size(), 1);
    for (int m = 0; m < w_frequency.nr; ++m)
        w_frequency(m, 0) =
            -2.0 * residue * pole / (std::pow(2.0 * PI * bosons[m] / beta, 2) + pole * pole);
    const auto w_time = transform.apply_bosonic_frequency_to_time(w_frequency);
    ComplexMatrix sigma_time(times.size(), energies.size());
    for (int j = 0; j < sigma_time.nr; ++j)
        for (int col = 0; col < sigma_time.nc; ++col)
            sigma_time(j, col) = w_time(j, 0) * std::exp(-energies[col] * times[j]) /
                                 (1.0 + std::exp(-beta * energies[col]));
    const auto sigma = transform.apply_fermionic_time_to_frequency(sigma_time);
    double max_analytic = 0.0, max_convolution = 0.0;
    for (int n = 0; n < sigma.nr; ++n)
        for (int col = 0; col < sigma.nc; ++col)
        {
            const double omega = (2.0 * fermions[n] + 1.0) * PI / beta;
            Complex convolution = 0.0;
            for (int m = -16384; m <= 16384; ++m)
            {
                const double nu = 2.0 * PI * m / beta;
                const double w = -2.0 * residue * pole / (nu * nu + pole * pole);
                convolution -= w / (I * (omega - nu) - energies[col]) / beta;
            }
            max_analytic = std::max(
                max_analytic,
                error(sigma(n, col), analytic_sigma(omega, energies[col], beta, pole, residue)));
            max_convolution = std::max(max_convolution, error(sigma(n, col), convolution));
            close(convolution, analytic_sigma(omega, energies[col], beta, pole, residue), 1e-11,
                  "independent Matsubara convolution vs closed form");
        }
    std::cout << "  composed Sigma errors: analytic=" << max_analytic
              << " direct convolution=" << max_convolution << '\n';
    require(max_analytic < 3e-7 && max_convolution < 3e-7, "composed W -> G*W -> Sigma model");
}

void check_branches_and_endpoints()
{
    const double beta = 3.7, xi = -0.4, pole = 0.8, tau = 0.31 * beta;
    const double occupation = 1.0 / (1.0 + std::exp(beta * xi));
    const double positive_g = (1.0 - occupation) * std::exp(-xi * tau);
    const double negative_g = -occupation * std::exp(-xi * (tau - beta));
    close(negative_g, -positive_g, 2e-15, "G antiperiodicity");
    close(-librpa_int::thermal_green_amplitude(xi, tau - beta, 1.0 / beta), negative_g, 2e-15,
          "existing thermal G negative branch needs explicit minus");
    const auto w_positive = pole_w_time(tau, beta, pole, pole, 0.6);
    // The negative-time value is obtained by periodic extension, not by assuming evenness.
    const auto sigma_positive = positive_g * w_positive;
    const auto sigma_negative = negative_g * w_positive;
    close(sigma_negative, -sigma_positive, 2e-15, "Sigma antiperiodicity");
    for (int n : {-3, -1, 0, 2})
    {
        const double omega = (2.0 * n + 1.0) * PI / beta;
        close(std::exp(I * omega * (tau - beta)) * sigma_negative,
              std::exp(I * omega * tau) * sigma_positive, 3e-15,
              "negative and positive interval Fourier integrands agree");
    }
    close(pole_w_time(0.0, beta, pole, pole, 0.6), pole_w_time(beta, beta, pole, pole, 0.6), 1e-15,
          "regular W periodic endpoints");
    const double epsilon = 1e-10;
    close((1.0 - occupation) * std::exp(-xi * epsilon) + occupation * std::exp(xi * epsilon), 1.0,
          1e-10, "G_lib endpoint jump is +1");
    const auto b = ThermalGWTransform::from_quadrature(
        beta, {epsilon, beta - epsilon}, {beta / 2.0, beta / 2.0}, {-2, 0, 1}, {-1, 0});
    const auto inverse = b.copy_bosonic_frequency_to_time();
    for (int m = 0; m < inverse.nc; ++m)
        close(inverse(0, m), inverse(1, m), 2e-10, "bosonic endpoint kernel limit");
    const auto forward = b.copy_fermionic_time_to_frequency();
    for (int n = 0; n < forward.nr; ++n)
        close(forward(n, 0), -forward(n, 1), 4e-10, "fermionic endpoint kernel limit");
}

void check_rejections()
{
    const double beta = 2.0, nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<double> times{0.25, 1.0, 1.75};
    const std::vector<int> bosons{-1, 0}, fermions{-1, 0, 2};
    ComplexMatrix b(3, 2), f(3, 3);
    const auto construct = [&](double beta_in, const std::vector<double> &tau,
                               const std::vector<int> &bm, const std::vector<int> &fn,
                               const ComplexMatrix &bin, const ComplexMatrix &fin)
    { return ThermalGWTransform(beta_in, tau, bm, fn, bin, fin); };
    for (double invalid_beta : {0.0, -1.0, nan, inf})
        reject([&] { construct(invalid_beta, times, bosons, fermions, b, f); });
    for (const auto &invalid_times : std::vector<std::vector<double>>{{},
                                                                      {0.0, 1.0, 1.75},
                                                                      {-0.1, 1.0, 1.75},
                                                                      {0.25, 1.0, beta},
                                                                      {0.25, 1.0, beta + 0.1},
                                                                      {0.25, 0.25, 1.75},
                                                                      {1.0, 0.25, 1.75},
                                                                      {nan, 1.0, 1.75},
                                                                      {0.25, inf, 1.75}})
        reject([&] { construct(beta, invalid_times, bosons, fermions, b, f); });
    for (const auto &bad_indices : std::vector<std::vector<int>>{{}, {0, 0}})
    {
        reject([&] { construct(beta, times, bad_indices, fermions, b, f); });
        reject([&] { construct(beta, times, bosons, bad_indices, b, f); });
    }
    reject([&] { construct(beta, times, bosons, fermions, ComplexMatrix(2, 3), f); });
    reject([&] { construct(beta, times, bosons, fermions, b, ComplexMatrix(3, 2)); });
    for (Complex bad_value :
         {Complex(nan, 0.0), Complex(0.0, nan), Complex(inf, 0.0), Complex(0.0, -inf)})
    {
        auto bad_b = b, bad_f = f;
        bad_b.c[0] = bad_value;
        bad_f.c[0] = bad_value;
        reject([&] { construct(beta, times, bosons, fermions, bad_b, f); });
        reject([&] { construct(beta, times, bosons, fermions, b, bad_f); });
    }
    b.size = 5;
    reject([&] { construct(beta, times, bosons, fermions, b, f); });
    b.size = 6;
    ComplexMatrix null_storage;
    null_storage.nr = 3;
    null_storage.nc = 2;
    null_storage.size = 6;
    reject([&] { construct(beta, times, bosons, fermions, null_storage, f); });
    ComplexMatrix impossible_shape;
    impossible_shape.nr = std::numeric_limits<int>::max();
    impossible_shape.nc = 2;
    reject([&] { construct(beta, times, bosons, fermions, impossible_shape, f); });
    for (const auto &weights :
         std::vector<std::vector<double>>{{}, {1.0}, {0.5, nan, 1.0}, {0.5, 0.5, inf}})
        reject([&]
               { ThermalGWTransform::from_quadrature(beta, times, weights, bosons, fermions); });
    reject([&] { ThermalGWTransform::from_quadrature(1e-308, {1e-309}, {1e-308}, {1}, {0}); });

    const auto transform =
        ThermalGWTransform::from_quadrature(beta, times, {0.5, 1.0, 0.5}, bosons, fermions);
    reject([&] { transform.apply_bosonic_frequency_to_time(ComplexMatrix(3, 1)); });
    reject([&] { transform.apply_fermionic_time_to_frequency(ComplexMatrix(2, 1)); });
    reject([&] { transform.apply_bosonic_frequency_to_time(ComplexMatrix(2, 0)); });
    reject([&] { transform.apply_fermionic_time_to_frequency(ComplexMatrix()); });
    ComplexMatrix bad_sample(2, 1), bad_time(3, 1);
    for (Complex bad_value : {Complex(nan, 0.0), Complex(0.0, inf)})
    {
        bad_sample.c[0] = bad_value;
        bad_time.c[0] = bad_value;
        reject([&] { transform.apply_bosonic_frequency_to_time(bad_sample); });
        reject([&] { transform.apply_fermionic_time_to_frequency(bad_time); });
    }
    bad_sample.zero_out();
    bad_sample.size = 1;
    reject([&] { transform.apply_bosonic_frequency_to_time(bad_sample); });
    bad_sample.size = 2;
}

void check_batch_rejections()
{
    const ThermalGWTransform transform(2.7, {0.2, 0.7, 1.6}, {2, 0, -3, -1, 1}, {3, -1, 0, -4},
                                       ComplexMatrix(3, 5), ComplexMatrix(4, 3));
    for (bool bosonic : {true, false})
    {
        const int rows = bosonic ? 5 : 3;
        const auto apply_samples = [&](const ComplexMatrix &samples)
        {
            if (bosonic)
                transform.apply_bosonic_frequency_to_time(samples);
            else
                transform.apply_fermionic_time_to_frequency(samples);
        };
        reject([&] { apply_samples(ComplexMatrix(rows - 1, 7)); });
        reject([&] { apply_samples(ComplexMatrix(rows, 0)); });
        reject([&] { apply_samples(ComplexMatrix()); });
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        for (Complex value :
             {Complex(nan, 0.0), Complex(0.0, nan), Complex(inf, 0.0), Complex(0.0, -inf)})
        {
            ComplexMatrix samples(rows, 7);
            samples(rows - 1, 6) = value;
            reject([&] { apply_samples(samples); });
        }
        ComplexMatrix inconsistent(rows, 7);
        --inconsistent.size;
        reject([&] { apply_samples(inconsistent); });
        ComplexMatrix null_storage;
        null_storage.nr = rows;
        null_storage.nc = 7;
        null_storage.size = rows * 7;
        reject([&] { apply_samples(null_storage); });
        null_storage.nr = -rows;
        reject([&] { apply_samples(null_storage); });
        null_storage.nr = rows;
        null_storage.nc = -7;
        reject([&] { apply_samples(null_storage); });
        null_storage.nc = std::numeric_limits<int>::max();
        reject([&] { apply_samples(null_storage); });
    }

    // Each input fits int storage, but the 65536 x 32768 output does not.
    const auto many_times = midpoints(1.0, 65536);
    std::vector<int> many_fermions(65536);
    for (int n = 0; n < 65536; ++n) many_fermions[n] = n;
    const ThermalGWTransform wide_b(1.0, many_times, {0}, {0}, ComplexMatrix(65536, 1),
                                    ComplexMatrix(1, 65536));
    const ThermalGWTransform wide_f(1.0, {0.5}, {0}, many_fermions, ComplexMatrix(1, 1),
                                    ComplexMatrix(65536, 1));
    const ComplexMatrix wide_samples(1, 32768);
    reject([&] { wide_b.apply_bosonic_frequency_to_time(wide_samples); });
    reject([&] { wide_f.apply_fermionic_time_to_frequency(wide_samples); });
}

void check_extreme_indices_and_overflow()
{
    const int low = std::numeric_limits<int>::min(), high = std::numeric_limits<int>::max();
    const auto transform = ThermalGWTransform::from_quadrature(4.0, {0.25, 3.75}, {2.0, 2.0},
                                                               {low, 0, high}, {low, high});
    const auto nu = transform.get_bosonic_frequencies_ha(),
               omega = transform.get_fermionic_frequencies_ha();
    close(nu[0], 2.0 * static_cast<double>(low) * PI / 4.0, 0.0, "signed bosonic index overflow");
    close(omega[1], (2.0 * static_cast<double>(high) + 1.0) * PI / 4.0, 0.0,
          "signed fermionic index overflow");
    const double largest = std::numeric_limits<double>::max();
    ComplexMatrix b(1, 1), f(1, 1), sample(1, 1);
    b.c[0] = largest;
    f.c[0] = largest;
    sample.c[0] = 2.0;
    const ThermalGWTransform overflowing(1.0, {0.5}, {0}, {0}, b, f);
    int rejected = 0;
    for (bool bosonic : {true, false}) try
        {
            if (bosonic)
                overflowing.apply_bosonic_frequency_to_time(sample);
            else
                overflowing.apply_fermionic_time_to_frequency(sample);
        }
        catch (const std::overflow_error &)
        {
            ++rejected;
        }
    require(rejected == 2, "nonfinite output from finite input must throw overflow_error");
}

void check_rectangular_output_overflow()
{
    for (bool bosonic : {true, false})
        for (bool imaginary : {true, false})
        {
            ComplexMatrix b(3, 5), f(4, 3), samples(bosonic ? 5 : 3, 7);
            auto &coefficients = bosonic ? b : f;
            const double large = 0.75 * std::numeric_limits<double>::max();
            const Complex value = imaginary ? Complex(0.0, large) : Complex(large, 0.0);
            // Individual products are finite; their sum overflows only in the last output entry.
            for (int source : {coefficients.nc - 2, coefficients.nc - 1})
            {
                coefficients(coefficients.nr - 1, source) = value;
                samples(source, samples.nc - 1) = 1.0;
            }
            const ThermalGWTransform transform(2.7, {0.2, 0.7, 1.6}, {2, 0, -3, -1, 1},
                                               {3, -1, 0, -4}, b, f);
            bool rejected = false;
            try
            {
                if (bosonic)
                    transform.apply_bosonic_frequency_to_time(samples);
                else
                    transform.apply_fermionic_time_to_frequency(samples);
            }
            catch (const std::overflow_error &exception)
            {
                require(std::string(exception.what()) ==
                            "thermal GW transform produced a nonfinite result",
                        "rectangular output overflow diagnostic changed");
                rejected = true;
            }
            require(rejected, "rectangular output overflow must throw overflow_error");
        }
}
}  // namespace

int main()
{
    std::cout << std::setprecision(12);
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"signed complex modes and zero mode", check_signed_complex_modes},
        {"owned external rectangular operators", check_owned_external_operators},
        {"rectangular B/F batches vs scalar reference", check_rectangular_batches},
        {"independent thermal single-pole W", check_single_pole_w},
        {"independent fermionic Sigma poles", check_fermionic_sigma},
        {"composed W and G*W vs analytic/direct Sigma", check_composed_sigma},
        {"negative branches and endpoint limits", check_branches_and_endpoints},
        {"metadata/shape/nonfinite rejections", check_rejections},
        {"batch storage/nonfinite/output-dimension rejections", check_batch_rejections},
        {"extreme signed indices and output overflow", check_extreme_indices_and_overflow},
        {"rectangular real/imaginary output overflow", check_rectangular_output_overflow}};
    int failures = 0;
    for (const auto &[name, test] : tests) try
        {
            test();
            std::cout << "PASS " << name << '\n';
        }
        catch (const std::exception &exception)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": " << exception.what() << '\n';
        }
    std::cout << tests.size() - failures << '/' << tests.size() << " groups passed\n";
    return failures == 0 ? 0 : 1;
}
