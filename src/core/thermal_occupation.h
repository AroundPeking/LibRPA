#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace librpa_int
{

double stable_log1pexp(double value);
double fermi_dirac_occupation(double energy_minus_mu, double kbt_ha);
double fermi_dirac_derivative(double energy_minus_mu, double kbt_ha);

// Returns the positive spectral amplitude. Existing Green-function builders
// retain responsibility for their own branch sign and k/spin normalization.
double thermal_green_amplitude(double energy_minus_mu, double tau, double kbt_ha);

struct ThermalOccupationMetadata
{
    std::string occupation_model;
    std::string occupation_storage;
    double chemical_potential_ha = 0.0;
    double kbt_ha = 0.0;
    double smearing_sigma_ry = 0.0;
    double max_occupation_per_band = 0.0;
    int spin_channels = 0;
    int kpoints_per_spin = 0;
    int bands = 0;
};

struct ThermalOccupationSample
{
    double energy_ha = 0.0;
    double stored_weight = 0.0;
    double kpoint_weight = 0.0;
};

ThermalOccupationMetadata parse_thermal_occupation_metadata(std::istream &input);
ThermalOccupationMetadata read_thermal_occupation_metadata(const std::string &path);

// Returns the maximum absolute error after removing k-point and spin
// normalization. Throws when the reference or an occupation is inconsistent.
double validate_thermal_occupations(const ThermalOccupationMetadata &metadata,
                                    double meanfield_chemical_potential_ha,
                                    const std::vector<ThermalOccupationSample> &samples,
                                    double occupation_tolerance);

} // namespace librpa_int

