#include "reader_thermal_gw_grid.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace driver
{
ExternalThermalGWGrid read_external_thermal_gw_grid(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    input.imbue(std::locale::classic());
    if (!input) throw std::runtime_error("cannot open external thermal GW grid: " + path);
    auto fail = [&](const std::string &reason)
    { throw std::runtime_error("external thermal GW grid " + path + ": " + reason); };
    const auto file_end = input.tellg();
    if (file_end < std::streampos(0)) fail("cannot determine file size");
    input.seekg(0);
    // Validate bytes without allocating a copy of the file, including provenance/trailing data.
    char buffer[4096];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0)
        for (std::streamsize i = 0; i < input.gcount(); ++i)
        {
            const auto byte = static_cast<unsigned char>(buffer[i]);
            if (!(byte >= 32 && byte <= 126) && !(byte >= 9 && byte <= 13))
                fail("requires printable ASCII and ASCII whitespace");
        }
    if (!input.eof() || input.bad()) fail("read failure during ASCII validation");
    input.clear();
    input.seekg(0);

    auto read_token = [&](const char *name)
    {
        std::string token;
        if (!(input >> token)) fail(std::string("invalid or truncated ") + name);
        return token;
    };
    auto read_number = [&](auto &value, const char *name)
    {
        std::istringstream token(read_token(name));
        token.imbue(std::locale::classic());
        if (!(token >> std::noskipws >> value) || token.peek() != std::char_traits<char>::eof())
            fail(std::string("invalid ") + name);
    };
    ExternalThermalGWGrid grid;
    if (read_token("version") != "LIBRPA_THERMAL_GW_V1") fail("unsupported version");
    grid.package = read_token("package provenance");
    grid.package_version = read_token("package version");
    if (read_token("units") != "Ha" || read_token("units") != "Ha^-1")
        fail("unsupported energy/time units");
    for (const auto *expected : {"B", "exp_minus", "inverse", "F", "exp_plus", "integral"})
        if (read_token("Fourier convention") != expected)
            fail("unsupported statistics/Fourier convention");
    for (auto *value : {&grid.beta_ha_inv, &grid.g_wmax_ha, &grid.w_wmax_ha, &grid.sigma_wmax_ha,
                        &grid.tolerance})
    {
        read_number(*value, "thermal metadata");
        if (!std::isfinite(*value) || *value <= 0.0) fail("invalid thermal metadata");
    }
    const double summed_wmax = grid.g_wmax_ha + grid.w_wmax_ha;
    const double support_limit = summed_wmax * (1.0 - 64 * std::numeric_limits<double>::epsilon());
    if (grid.tolerance >= 1.0 || !std::isfinite(1.0 / grid.beta_ha_inv) ||
        !std::isfinite(summed_wmax) || grid.sigma_wmax_ha < support_limit ||
        !std::isfinite(grid.beta_ha_inv * grid.sigma_wmax_ha))
        fail("invalid thermal metadata: require finite beta and sigma_wmax >= g_wmax + w_wmax");
    for (auto *dimension : {&grid.ntau, &grid.nboson, &grid.nfermion})
    {
        read_number(*dimension, "dimensions");
        if (*dimension <= 0) fail("dimensions must be positive");
    }
    const auto max_dimension = std::numeric_limits<int>::max();
    if (grid.nboson > max_dimension / grid.ntau || grid.nfermion > max_dimension / grid.ntau)
        fail("overflowing matrix dimensions");
    const auto b_entries = static_cast<std::uintmax_t>(grid.ntau) * grid.nboson;
    const auto f_entries = static_cast<std::uintmax_t>(grid.nfermion) * grid.ntau;
    const auto nodes = static_cast<std::uintmax_t>(grid.ntau) + grid.nboson + grid.nfermion;
    const auto minimum_bytes = 2 * nodes + 4 * (b_entries + f_entries) - 1;
    const auto payload_start = input.tellg();
    // Count only remaining payload bytes: a padded header must not license huge allocations.
    if (payload_start < std::streampos(0) || payload_start > file_end ||
        static_cast<std::uintmax_t>(file_end - payload_start) < minimum_bytes)
        fail("file size is too small for declared dimensions");
    if (b_entries > grid.b_real.max_size() || f_entries > grid.f_real.max_size() ||
        static_cast<std::uintmax_t>(grid.ntau) > grid.times.max_size() ||
        static_cast<std::uintmax_t>(grid.nboson) > grid.bosonic_indices.max_size() ||
        static_cast<std::uintmax_t>(grid.nfermion) > grid.fermionic_indices.max_size())
        fail("dimensions exceed container storage");

    grid.times.resize(grid.ntau);
    double previous = 0.0;
    for (auto &time : grid.times)
    {
        read_number(time, "time node");
        if (!std::isfinite(time) || time <= previous || time >= grid.beta_ha_inv)
            fail("time nodes must increase strictly inside (0,beta)");
        previous = time;
    }
    auto read_indices = [&](std::vector<int> &indices, int count, bool fermionic)
    {
        const char *name = fermionic ? "fermionic index" : "bosonic index";
        indices.resize(count);
        std::unordered_set<int> unique;
        for (auto &index : indices)
        {
            read_number(index, name);
            if (!unique.insert(index).second) fail(std::string("duplicate ") + name);
            const double frequency = (2.0 * static_cast<double>(index) + (fermionic ? 1.0 : 0.0)) *
                                     std::acos(-1.0) / grid.beta_ha_inv;
            if (!std::isfinite(frequency) || (frequency == 0.0 && (fermionic || index != 0)))
                fail(std::string("unrepresentable frequency for ") + name);
        }
        if (!fermionic && unique.count(0) == 0) fail("bosonic indices must include zero");
        for (int index : indices)
        {
            // INT_MIN cannot have a bosonic partner but pairs with INT_MAX for fermions.
            const auto partner = -static_cast<std::int64_t>(index) - (fermionic ? 1 : 0);
            if (partner < std::numeric_limits<int>::min() || partner > max_dimension ||
                unique.count(static_cast<int>(partner)) == 0)
                fail(std::string("missing opposite-frequency partner for ") + name);
        }
    };
    read_indices(grid.bosonic_indices, grid.nboson, false);
    read_indices(grid.fermionic_indices, grid.nfermion, true);
    auto read_operator = [&](std::vector<double> &real, std::vector<double> &imag,
                             std::size_t entries, const char *name)
    {
        real.resize(entries);
        imag.resize(entries);
        for (std::size_t k = 0; k < entries; ++k)
        {
            read_number(real[k], name);
            read_number(imag[k], name);
            if (!std::isfinite(real[k]) || !std::isfinite(imag[k]))
                fail(std::string("nonfinite ") + name);
        }
    };
    read_operator(grid.b_real, grid.b_imag, b_entries, "bosonic operator");
    read_operator(grid.f_real, grid.f_imag, f_entries, "fermionic operator");
    for (int column = 0; column < grid.nboson; ++column)
        if (grid.bosonic_indices[column] == 0)
            for (int row = 0; row < grid.ntau; ++row)
            {
                const auto k = row * grid.nboson + column;
                const double real = grid.beta_ha_inv * grid.b_real[k];
                const double imag = grid.beta_ha_inv * grid.b_imag[k];
                if (!std::isfinite(real) || !std::isfinite(imag) ||
                    std::hypot(real - 1.0, imag) > 1e-10)
                    fail("bosonic zero-mode normalization requires beta*B_zero = 1");
            }
    std::string extra;
    if (input >> extra) fail("unexpected trailing data");
    if (!input.eof() || input.bad()) fail("read failure after operators");
    return grid;
}

void load_external_thermal_gw_grid(const std::string &path, librpa::Handler &handler,
                                   const librpa::Options &options)
{
    if (options.tfgrids_type != LIBRPA_TFGRID_FD_MATSUBARA)
        throw std::runtime_error("external thermal GW grid requires tfgrids_type=fd_matsubara");
    const auto grid = read_external_thermal_gw_grid(path);
    handler.set_external_thermal_gw_grid(
        grid.beta_ha_inv, grid.g_wmax_ha, grid.w_wmax_ha, grid.sigma_wmax_ha, grid.tolerance,
        grid.ntau, grid.nboson, grid.nfermion, grid.times.data(), grid.bosonic_indices.data(),
        grid.fermionic_indices.data(), grid.b_real.data(), grid.b_imag.data(), grid.f_real.data(),
        grid.f_imag.data());
}
}  // namespace driver
