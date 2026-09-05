#pragma once

#include <string>
#include <vector>

#include "librpa.hpp"

namespace driver
{
struct ExternalThermalGWGrid
{
    std::string package, package_version;
    double beta_ha_inv = 0.0, g_wmax_ha = 0.0, w_wmax_ha = 0.0;
    double sigma_wmax_ha = 0.0, tolerance = 0.0;
    int ntau = 0, nboson = 0, nfermion = 0;
    std::vector<double> times, b_real, b_imag, f_real, f_imag;
    std::vector<int> bosonic_indices, fermionic_indices;
};

ExternalThermalGWGrid read_external_thermal_gw_grid(const std::string &path);
void load_external_thermal_gw_grid(const std::string &path, librpa::Handler &handler,
                                   const librpa::Options &options);
}  // namespace driver
