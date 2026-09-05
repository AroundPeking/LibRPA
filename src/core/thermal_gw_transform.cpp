#include "thermal_gw_transform.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace librpa_int
{
namespace
{
const double PI = std::acos(-1.0);

bool finite(const std::complex<double> value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void validate_dimensions(const std::size_t rows, const std::size_t columns)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (rows == 0 || columns == 0 || rows > maximum || columns > maximum ||
        rows > maximum / columns)
        throw std::invalid_argument("thermal GW matrix dimensions must fit positive int storage");
}

void validate_matrix(const ComplexMatrix &matrix, const char *name)
{
    if (matrix.nr <= 0 || matrix.nc <= 0)
        throw std::invalid_argument(std::string(name) + " must have positive dimensions");
    validate_dimensions(matrix.nr, matrix.nc);
    if (matrix.size != matrix.nr * matrix.nc || matrix.c == nullptr)
        throw std::invalid_argument(std::string(name) + " has inconsistent storage");
    for (int k = 0; k < matrix.size; ++k)
        if (!finite(matrix.c[k]))
            throw std::invalid_argument(std::string(name) + " contains a nonfinite value");
}

std::vector<double> frequencies(double beta, const std::vector<int> &indices, bool fermionic)
{
    std::vector<double> result;
    result.reserve(indices.size());
    for (int index : indices)
    {
        // Promote before doubling, including INT_MIN/INT_MAX and fermionic n=-1.
        const double frequency =
            (2.0 * static_cast<double>(index) + (fermionic ? 1.0 : 0.0)) * PI / beta;
        if (!std::isfinite(frequency) || (frequency == 0.0 && (fermionic || index != 0)))
            throw std::invalid_argument("thermal GW Matsubara frequency is not representable");
        result.push_back(frequency);
    }
    return result;
}

void validate_grid(double beta, const std::vector<double> &times,
                   const std::vector<int> &bosonic_indices,
                   const std::vector<int> &fermionic_indices)
{
    if (!std::isfinite(beta) || beta <= 0.0 || !std::isfinite(1.0 / beta))
        throw std::invalid_argument("thermal GW requires positive finite beta and finite 1/beta");
    validate_dimensions(times.size(), bosonic_indices.size());
    validate_dimensions(fermionic_indices.size(), times.size());
    double previous = 0.0;
    for (double time : times)
    {
        if (!std::isfinite(time) || time <= previous || time >= beta)
            throw std::invalid_argument("thermal GW times must increase strictly inside (0,beta)");
        previous = time;
    }
    for (const auto *indices : {&bosonic_indices, &fermionic_indices})
    {
        std::unordered_set<int> unique(indices->begin(), indices->end());
        if (unique.size() != indices->size())
            throw std::invalid_argument("thermal GW Matsubara indices must be unique");
    }
    frequencies(beta, bosonic_indices, false);
    frequencies(beta, fermionic_indices, true);
}

ComplexMatrix apply(const ComplexMatrix &coefficients, const ComplexMatrix &samples)
{
    validate_matrix(samples, "thermal GW samples");
    if (samples.nr != coefficients.nc)
        throw std::invalid_argument("thermal GW sample rows do not match the source grid");
    validate_dimensions(coefficients.nr, samples.nc);
    ComplexMatrix result = coefficients * samples;
    for (int k = 0; k < result.size; ++k)
        if (!finite(result.c[k]))
            throw std::overflow_error("thermal GW transform produced a nonfinite result");
    return result;
}
}  // namespace

ThermalGWTransform::ThermalGWTransform(double beta_ha_inv, const std::vector<double> &times,
                                       const std::vector<int> &bosonic_indices,
                                       const std::vector<int> &fermionic_indices,
                                       const ComplexMatrix &bosonic_frequency_to_time,
                                       const ComplexMatrix &fermionic_time_to_frequency)
    : beta_ha_inv_(beta_ha_inv)
{
    validate_grid(beta_ha_inv, times, bosonic_indices, fermionic_indices);
    validate_matrix(bosonic_frequency_to_time, "thermal GW bosonic operator");
    validate_matrix(fermionic_time_to_frequency, "thermal GW fermionic operator");
    if (static_cast<std::size_t>(bosonic_frequency_to_time.nr) != times.size() ||
        static_cast<std::size_t>(bosonic_frequency_to_time.nc) != bosonic_indices.size() ||
        static_cast<std::size_t>(fermionic_time_to_frequency.nr) != fermionic_indices.size() ||
        static_cast<std::size_t>(fermionic_time_to_frequency.nc) != times.size())
        throw std::invalid_argument("thermal GW operator shapes do not match grid metadata");

    // Validate before invoking ComplexMatrix's unchecked deep-copy operations.
    times_ = times;
    bosonic_indices_ = bosonic_indices;
    fermionic_indices_ = fermionic_indices;
    bosonic_frequency_to_time_ = bosonic_frequency_to_time;
    fermionic_time_to_frequency_ = fermionic_time_to_frequency;
}

ThermalGWTransform ThermalGWTransform::from_quadrature(double beta_ha_inv,
                                                       const std::vector<double> &times,
                                                       const std::vector<double> &time_weights,
                                                       const std::vector<int> &bosonic_indices,
                                                       const std::vector<int> &fermionic_indices)
{
    validate_grid(beta_ha_inv, times, bosonic_indices, fermionic_indices);
    if (time_weights.size() != times.size())
        throw std::invalid_argument("thermal GW quadrature weights do not match the time grid");
    for (double weight : time_weights)
        if (!std::isfinite(weight))
            throw std::invalid_argument("thermal GW quadrature weight must be finite");

    ComplexMatrix bosonic(static_cast<int>(times.size()), static_cast<int>(bosonic_indices.size()));
    ComplexMatrix fermionic(static_cast<int>(fermionic_indices.size()),
                            static_cast<int>(times.size()));
    for (int j = 0; j < bosonic.nr; ++j)
    {
        const double fraction = times[j] / beta_ha_inv;
        // Form dimensionless phases to avoid overflowing frequency*tau at extreme beta.
        for (int m = 0; m < bosonic.nc; ++m)
        {
            const double phase =
                -2.0 * PI * std::remainder(static_cast<double>(bosonic_indices[m]) * fraction, 1.0);
            bosonic(j, m) = std::complex<double>(std::cos(phase), std::sin(phase)) / beta_ha_inv;
        }
        for (int n = 0; n < fermionic.nr; ++n)
        {
            const double phase =
                2.0 * PI *
                std::remainder((static_cast<double>(fermionic_indices[n]) + 0.5) * fraction, 1.0);
            fermionic(n, j) =
                time_weights[j] * std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }
    return ThermalGWTransform(beta_ha_inv, times, bosonic_indices, fermionic_indices, bosonic,
                              fermionic);
}

std::vector<double> ThermalGWTransform::get_bosonic_frequencies_ha() const
{
    return frequencies(beta_ha_inv_, bosonic_indices_, false);
}

std::vector<double> ThermalGWTransform::get_fermionic_frequencies_ha() const
{
    return frequencies(beta_ha_inv_, fermionic_indices_, true);
}

ComplexMatrix ThermalGWTransform::copy_bosonic_frequency_to_time() const
{
    return bosonic_frequency_to_time_;
}

ComplexMatrix ThermalGWTransform::copy_fermionic_time_to_frequency() const
{
    return fermionic_time_to_frequency_;
}

ComplexMatrix ThermalGWTransform::apply_bosonic_frequency_to_time(
    const ComplexMatrix &samples) const
{
    return apply(bosonic_frequency_to_time_, samples);
}

ComplexMatrix ThermalGWTransform::apply_fermionic_time_to_frequency(
    const ComplexMatrix &samples) const
{
    return apply(fermionic_time_to_frequency_, samples);
}
}  // namespace librpa_int
