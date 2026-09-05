#include "reader_thermal_grid.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace driver
{
ExternalThermalGrid read_external_thermal_grid(const std::string &path)
{
    std::ifstream input(path);
    input.imbue(std::locale::classic());
    if (!input) throw std::runtime_error("cannot open external thermal grid: " + path);
    auto fail = [&](const std::string &reason)
    { throw std::runtime_error("external thermal grid " + path + ": " + reason); };
    std::string magic, statistics, sign, normalization;
    ExternalThermalGrid grid;
    if (!(input >> magic) || magic != "LIBRPA_THERMAL_TAU_V1") fail("unsupported version");
    if (!(input >> grid.package >> grid.package_version >> statistics >> sign >> normalization))
        fail("truncated provenance/convention header");
    if (statistics != "B" || sign != "exp_plus" || normalization != "integral")
        fail("unsupported statistics/Fourier convention");
    if (!(input >> grid.beta_ha_inv >> grid.wmax_ha >> grid.tolerance) ||
        !std::isfinite(grid.beta_ha_inv) || grid.beta_ha_inv <= 0.0 ||
        !std::isfinite(grid.wmax_ha) || grid.wmax_ha <= 0.0 || !std::isfinite(grid.tolerance) ||
        grid.tolerance <= 0.0 || grid.tolerance >= 1.0)
        fail("invalid thermal metadata");
    int ntau = 0;
    const int max_dimension = std::numeric_limits<int>::max();
    if (!(input >> ntau >> grid.nfreq) || ntau <= 0 || grid.nfreq <= 0 ||
        grid.nfreq > max_dimension / ntau || grid.nfreq > max_dimension / grid.nfreq)
        fail("invalid or overflowing dimensions");
    const auto entries = static_cast<std::uintmax_t>(ntau) * grid.nfreq;
    // Bound allocations using the minimum text size before trusting file dimensions.
    if (std::filesystem::file_size(path) < 4 * entries + 2 * static_cast<std::uintmax_t>(ntau))
        fail("file size is too small for declared dimensions");
    grid.times.resize(ntau);
    double previous_time = 0.0;
    for (auto &time : grid.times)
    {
        if (!(input >> time) || !std::isfinite(time) || time <= previous_time ||
            time >= grid.beta_ha_inv)
            fail("invalid, unordered or truncated time nodes");
        previous_time = time;
    }
    grid.transform_real.resize(entries);
    grid.transform_imag.resize(entries);
    for (int row = 0; row < grid.nfreq; ++row)
    {
        int index = -1;
        if (!(input >> index) || index != row) fail("nonconsecutive bosonic frequency index");
        for (int col = 0; col < ntau; ++col)
        {
            const int offset = row * ntau + col;
            auto &real = grid.transform_real[offset];
            auto &imag = grid.transform_imag[offset];
            if (!(input >> real >> imag) || !std::isfinite(real) || !std::isfinite(imag))
                fail("invalid or truncated complex transform");
        }
    }
    std::string extra;
    if (input >> extra) fail("unexpected trailing data");
    if (!input.eof()) fail("read failure after transform");
    return grid;
}

void load_external_thermal_grid(const std::string &path, librpa::Handler &handler,
                                const librpa::Options &options)
{
    if (options.tfgrids_type != LIBRPA_TFGRID_FD_MATSUBARA)
        throw std::runtime_error("external thermal grid requires tfgrids_type=fd_matsubara");
    const auto grid = read_external_thermal_grid(path);
    if (options.nfreq != grid.nfreq || options.ntau != static_cast<int>(grid.times.size()))
        throw std::runtime_error("external thermal grid dimensions differ from nfreq/ntau");
    handler.set_external_thermal_time_grid(grid.beta_ha_inv, grid.wmax_ha, grid.tolerance,
                                           grid.nfreq, options.ntau, grid.times.data(),
                                           grid.transform_real.data(), grid.transform_imag.data());
}
}  // namespace driver
