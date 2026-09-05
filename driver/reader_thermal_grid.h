#pragma once

#include <string>
#include <vector>

#include "librpa.hpp"

namespace driver
{
struct ExternalThermalGrid
{
    std::string package, package_version;
    double beta_ha_inv = 0.0, wmax_ha = 0.0, tolerance = 0.0;
    int nfreq = 0;
    std::vector<double> times, transform_real, transform_imag;
};

ExternalThermalGrid read_external_thermal_grid(const std::string &path);
void load_external_thermal_grid(const std::string &path, librpa::Handler &handler,
                                const librpa::Options &options);
}  // namespace driver
