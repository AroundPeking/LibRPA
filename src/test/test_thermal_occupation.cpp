#include "../core/thermal_occupation.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace librpa_int;

namespace
{
void require_near(const double actual, const double expected, const double tolerance)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("values differ beyond tolerance");
}

void test_stable_fermi_dirac_functions()
{
    const double kbt = 0.0025;
    require_near(fermi_dirac_occupation(0.0, kbt), 0.5, 1.0e-15);
    require_near(fermi_dirac_derivative(0.0, kbt), -1.0 / (4.0 * kbt), 1.0e-12);
    assert(fermi_dirac_occupation(-1000.0 * kbt, kbt) == 1.0);
    assert(fermi_dirac_occupation(1000.0 * kbt, kbt) == 0.0);
    assert(std::isfinite(stable_log1pexp(1000.0)));
    require_near(stable_log1pexp(1000.0), 1000.0, 1.0e-13);
    require_near(stable_log1pexp(-1000.0), 0.0, 1.0e-300);
}

void test_stable_thermal_green_amplitudes()
{
    const double kbt = 0.0025;
    const double beta = 1.0 / kbt;
    const double expected = std::exp(-500.0);

    const double hole_amplitude =
        thermal_green_amplitude(-1000.0 * kbt, 0.5 * beta, kbt);
    const double particle_amplitude =
        thermal_green_amplitude(1000.0 * kbt, -0.5 * beta, kbt);
    require_near(hole_amplitude / expected, 1.0, 1.0e-13);
    require_near(particle_amplitude / expected, 1.0, 1.0e-13);

    bool rejected = false;
    try
    {
        (void)thermal_green_amplitude(0.0, 0.0, kbt);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

std::string valid_metadata_text()
{
    return "format thermal_occupation_v1\n"
           "occupation_model fermi_dirac\n"
           "chemical_potential_ha 1.25000000000000000e-01\n"
           "kbt_ha 2.50000000000000005e-03\n"
           "smearing_sigma_ry 5.00000000000000010e-03\n"
           "occupation_storage band_out_times_nk\n"
           "max_occupation_per_band 2.00000000000000000e+00\n"
           "spin_channels 1\n"
           "kpoints_per_spin 2\n"
           "bands 1\n";
}

void test_metadata_parser()
{
    std::istringstream input(valid_metadata_text());
    const auto metadata = parse_thermal_occupation_metadata(input);
    assert(metadata.occupation_model == "fermi_dirac");
    assert(metadata.occupation_storage == "band_out_times_nk");
    require_near(metadata.chemical_potential_ha, 0.125, 0.0);
    require_near(metadata.kbt_ha, 0.0025, 0.0);
    require_near(metadata.smearing_sigma_ry, 0.005, 0.0);
    require_near(metadata.max_occupation_per_band, 2.0, 0.0);
    assert(metadata.spin_channels == 1);
    assert(metadata.kpoints_per_spin == 2);
    assert(metadata.bands == 1);

    bool rejected = false;
    std::istringstream invalid(valid_metadata_text() + "unexpected value\n");
    try
    {
        (void)parse_thermal_occupation_metadata(invalid);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_occupation_validation_uses_kpoint_and_spin_normalization()
{
    std::istringstream input(valid_metadata_text());
    const auto metadata = parse_thermal_occupation_metadata(input);
    const double log_three = std::log(3.0);
    const std::vector<ThermalOccupationSample> samples{
        {0.125 - metadata.kbt_ha * log_three, 0.2 * 2.0 * 0.75, 0.2},
        {0.125 + metadata.kbt_ha * log_three, 0.8 * 2.0 * 0.25, 0.8},
    };
    const double max_error =
        validate_thermal_occupations(metadata, 0.125, samples, 1.0e-13);
    assert(max_error < 1.0e-14);

    auto inconsistent = samples;
    inconsistent[1].stored_weight += 1.0e-3;
    bool rejected = false;
    try
    {
        (void)validate_thermal_occupations(metadata, 0.125, inconsistent, 1.0e-6);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_normalized_fermi_dirac_band_weight()
{
    const double kbt = 0.0025;
    const auto reference = make_fermi_dirac_reference(kbt, 0.125, 2.0, 1.0e-12);
    require_near(normalized_fermi_dirac_band_weight(0.125, reference, 8), 0.125,
                 1.0e-15);
    require_near(normalized_fermi_dirac_band_weight(0.125 - 1000.0 * kbt, reference, 8), 0.25,
                 1.0e-15);
    require_near(normalized_fermi_dirac_band_weight(0.125 + 1000.0 * kbt, reference, 8), 0.0,
                 1.0e-15);

    bool rejected = false;
    try
    {
        (void)normalized_fermi_dirac_band_weight(0.125, reference, 0);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_invalid_temperature_is_rejected()
{
    for (const double kbt : {0.0,
                             -1.0,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    {
        bool rejected = false;
        try
        {
            (void)fermi_dirac_occupation(0.0, kbt);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        assert(rejected);
    }
}
} // namespace

int main()
{
    test_stable_fermi_dirac_functions();
    test_stable_thermal_green_amplitudes();
    test_metadata_parser();
    test_occupation_validation_uses_kpoint_and_spin_normalization();
    test_normalized_fermi_dirac_band_weight();
    test_invalid_temperature_is_rejected();
    return 0;
}
