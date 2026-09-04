#include "thermal_occupation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace librpa_int
{
namespace
{
void require_positive_temperature(const double kbt_ha)
{
    if (!std::isfinite(kbt_ha) || kbt_ha <= 0.0)
        throw std::invalid_argument("kBT must be positive and finite");
}

double parse_double(const std::string &key, const std::string &value)
{
    std::size_t parsed = 0;
    double result = 0.0;
    try
    {
        result = std::stod(value, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("invalid floating-point value for " + key);
    }
    if (parsed != value.size() || !std::isfinite(result))
        throw std::invalid_argument("invalid floating-point value for " + key);
    return result;
}

int parse_integer(const std::string &key, const std::string &value)
{
    std::size_t parsed = 0;
    int result = 0;
    try
    {
        result = std::stoi(value, &parsed);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("invalid integer value for " + key);
    }
    if (parsed != value.size()) throw std::invalid_argument("invalid integer value for " + key);
    return result;
}

void validate_metadata(const ThermalOccupationMetadata &metadata)
{
    if (metadata.occupation_model != "fermi_dirac")
        throw std::invalid_argument("thermal occupation model must be fermi_dirac");
    if (metadata.occupation_storage != "band_out_times_nk")
        throw std::invalid_argument("unsupported thermal occupation storage convention");
    if (!std::isfinite(metadata.chemical_potential_ha))
        throw std::invalid_argument("thermal chemical potential must be finite");
    require_positive_temperature(metadata.kbt_ha);
    if (!std::isfinite(metadata.smearing_sigma_ry) || metadata.smearing_sigma_ry <= 0.0)
        throw std::invalid_argument("thermal smearing sigma must be positive and finite");
    const double energy_scale =
        std::max({1.0, metadata.smearing_sigma_ry, 2.0 * metadata.kbt_ha});
    const double energy_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * energy_scale;
    if (std::abs(metadata.smearing_sigma_ry - 2.0 * metadata.kbt_ha) > energy_tolerance)
        throw std::invalid_argument("thermal Ha and Ry energies are inconsistent");
    if (metadata.max_occupation_per_band != 1.0
        && metadata.max_occupation_per_band != 2.0)
        throw std::invalid_argument("maximum occupation per band must be one or two");
    if (metadata.spin_channels <= 0 || metadata.kpoints_per_spin <= 0 || metadata.bands <= 0)
        throw std::invalid_argument("thermal metadata dimensions must be positive");
}
} // namespace

double stable_log1pexp(const double value)
{
    if (value > 0.0) return value + std::log1p(std::exp(-value));
    return std::log1p(std::exp(value));
}

double fermi_dirac_occupation(const double energy_minus_mu, const double kbt_ha)
{
    require_positive_temperature(kbt_ha);
    if (!std::isfinite(energy_minus_mu))
        throw std::invalid_argument("energy relative to the chemical potential must be finite");
    return std::exp(-stable_log1pexp(energy_minus_mu / kbt_ha));
}

double fermi_dirac_derivative(const double energy_minus_mu, const double kbt_ha)
{
    require_positive_temperature(kbt_ha);
    if (!std::isfinite(energy_minus_mu))
        throw std::invalid_argument("energy relative to the chemical potential must be finite");
    const double scaled_energy = energy_minus_mu / kbt_ha;
    const double log_product =
        -stable_log1pexp(scaled_energy) - stable_log1pexp(-scaled_energy);
    return -std::exp(log_product) / kbt_ha;
}

double thermal_green_amplitude(const double energy_minus_mu,
                               const double tau,
                               const double kbt_ha)
{
    require_positive_temperature(kbt_ha);
    if (!std::isfinite(energy_minus_mu) || !std::isfinite(tau))
        throw std::invalid_argument("thermal Green-function arguments must be finite");
    if (tau == 0.0)
        throw std::invalid_argument("thermal Green function requires an explicit time branch");

    const double scaled_energy = energy_minus_mu / kbt_ha;
    const double log_occupation = tau > 0.0 ? -stable_log1pexp(-scaled_energy)
                                             : -stable_log1pexp(scaled_energy);
    return std::exp(-energy_minus_mu * tau + log_occupation);
}

FermiDiracReference make_fermi_dirac_reference(const double kbt_ha,
                                                const double chemical_potential_ha,
                                                const double max_occupation_per_band,
                                                const double occupation_tolerance)
{
    require_positive_temperature(kbt_ha);
    if (!std::isfinite(chemical_potential_ha))
        throw std::invalid_argument("Fermi-Dirac chemical potential must be finite");
    if (max_occupation_per_band != 1.0 && max_occupation_per_band != 2.0)
        throw std::invalid_argument("maximum occupation per band must be one or two");
    if (!std::isfinite(occupation_tolerance) || occupation_tolerance < 0.0)
        throw std::invalid_argument("occupation tolerance must be nonnegative and finite");

    FermiDiracReference reference;
    reference.enabled = true;
    reference.chemical_potential_ha = chemical_potential_ha;
    reference.kbt_ha = kbt_ha;
    reference.max_occupation_per_band = max_occupation_per_band;
    reference.occupation_tolerance = occupation_tolerance;
    return reference;
}

void validate_fermi_dirac_chemical_potential(const FermiDiracReference &reference,
                                              const double meanfield_chemical_potential_ha)
{
    if (!reference.enabled) return;
    if (!std::isfinite(meanfield_chemical_potential_ha))
        throw std::invalid_argument("mean-field chemical potential must be finite");
    const double scale =
        std::max({1.0, std::abs(reference.chemical_potential_ha),
                  std::abs(meanfield_chemical_potential_ha)});
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(reference.chemical_potential_ha - meanfield_chemical_potential_ha)
        > tolerance)
        throw std::invalid_argument("Fermi-Dirac and mean-field chemical potentials differ");
}

ThermalOccupationMetadata parse_thermal_occupation_metadata(std::istream &input)
{
    std::map<std::string, std::string> fields;
    std::string key;
    std::string value;
    while (input >> key)
    {
        if (!(input >> value)) throw std::invalid_argument("thermal metadata field has no value");
        if (!fields.emplace(key, value).second)
            throw std::invalid_argument("duplicate thermal metadata field: " + key);
    }
    if (!input.eof()) throw std::invalid_argument("failed while reading thermal metadata");

    const std::vector<std::string> required{
        "format",
        "occupation_model",
        "chemical_potential_ha",
        "kbt_ha",
        "smearing_sigma_ry",
        "occupation_storage",
        "max_occupation_per_band",
        "spin_channels",
        "kpoints_per_spin",
        "bands",
    };
    if (fields.size() != required.size())
        throw std::invalid_argument("thermal metadata contains missing or unknown fields");
    for (const auto &name : required)
        if (fields.count(name) == 0)
            throw std::invalid_argument("missing thermal metadata field: " + name);
    if (fields.at("format") != "thermal_occupation_v1")
        throw std::invalid_argument("unsupported thermal metadata format");

    ThermalOccupationMetadata metadata;
    metadata.occupation_model = fields.at("occupation_model");
    metadata.occupation_storage = fields.at("occupation_storage");
    metadata.chemical_potential_ha =
        parse_double("chemical_potential_ha", fields.at("chemical_potential_ha"));
    metadata.kbt_ha = parse_double("kbt_ha", fields.at("kbt_ha"));
    metadata.smearing_sigma_ry =
        parse_double("smearing_sigma_ry", fields.at("smearing_sigma_ry"));
    metadata.max_occupation_per_band =
        parse_double("max_occupation_per_band", fields.at("max_occupation_per_band"));
    metadata.spin_channels = parse_integer("spin_channels", fields.at("spin_channels"));
    metadata.kpoints_per_spin =
        parse_integer("kpoints_per_spin", fields.at("kpoints_per_spin"));
    metadata.bands = parse_integer("bands", fields.at("bands"));
    validate_metadata(metadata);
    return metadata;
}

ThermalOccupationMetadata read_thermal_occupation_metadata(const std::string &path)
{
    std::ifstream input(path);
    if (!input.good()) throw std::invalid_argument("failed to open thermal metadata: " + path);
    return parse_thermal_occupation_metadata(input);
}

double validate_thermal_occupations(const ThermalOccupationMetadata &metadata,
                                    const double meanfield_chemical_potential_ha,
                                    const std::vector<ThermalOccupationSample> &samples,
                                    const double occupation_tolerance)
{
    validate_metadata(metadata);
    if (!std::isfinite(meanfield_chemical_potential_ha))
        throw std::invalid_argument("mean-field chemical potential must be finite");
    if (!std::isfinite(occupation_tolerance) || occupation_tolerance < 0.0)
        throw std::invalid_argument("occupation tolerance must be nonnegative and finite");

    const double mu_scale =
        std::max({1.0, std::abs(metadata.chemical_potential_ha),
                  std::abs(meanfield_chemical_potential_ha)});
    const double mu_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * mu_scale;
    if (std::abs(metadata.chemical_potential_ha - meanfield_chemical_potential_ha)
        > mu_tolerance)
        throw std::invalid_argument("thermal and mean-field chemical potentials differ");

    const std::size_t expected_count = static_cast<std::size_t>(metadata.spin_channels)
                                       * static_cast<std::size_t>(metadata.kpoints_per_spin)
                                       * static_cast<std::size_t>(metadata.bands);
    if (samples.size() != expected_count)
        throw std::invalid_argument("thermal occupation sample count does not match metadata");

    double maximum_error = 0.0;
    for (const auto &sample : samples)
    {
        if (!std::isfinite(sample.energy_ha) || !std::isfinite(sample.stored_weight)
            || sample.stored_weight < 0.0 || !std::isfinite(sample.kpoint_weight)
            || sample.kpoint_weight <= 0.0)
            throw std::invalid_argument("invalid thermal occupation sample");
        const double occupation = sample.stored_weight
                                  / (sample.kpoint_weight
                                     * metadata.max_occupation_per_band);
        const double expected = fermi_dirac_occupation(
            sample.energy_ha - metadata.chemical_potential_ha, metadata.kbt_ha);
        const double error = std::abs(occupation - expected);
        maximum_error = std::max(maximum_error, error);
        if (occupation < -occupation_tolerance || occupation > 1.0 + occupation_tolerance
            || error > occupation_tolerance)
            throw std::invalid_argument("band occupation is inconsistent with thermal metadata");
    }
    return maximum_error;
}

} // namespace librpa_int
