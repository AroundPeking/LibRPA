#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace driver
{

struct FermiSurfacePayloadHeader
{
    std::int64_t record_count = 0;
    std::array<std::int32_t, 3> mesh{};
    std::int32_t nspin = 0;
    std::int32_t nspinor = 0;
    std::int32_t basis_size = 0;
    std::int64_t record_bytes = 0;
    double kbt_ha = 0.0;
    double chemical_potential_ha = 0.0;
    double cutoff_kbt = 0.0;
    double spin_degeneracy = 0.0;
    double cell_volume_bohr3 = 0.0;
    double retained_kappa_fraction = 0.0;
    double retained_plasma_trace_fraction = 0.0;
};

struct FermiSurfacePayloadRecord
{
    std::int64_t k_index = 0;
    std::int32_t spin = 0;
    std::int32_t band = 0;
    std::array<double, 3> kfrac{};
    double energy_ha = 0.0;
    double minus_fprime_ha_inv = 0.0;
    std::array<std::complex<double>, 3> velocity{};
    std::vector<std::complex<double>> wfc;
};

using FermiSurfacePayloadConsumer = std::function<void(const FermiSurfacePayloadRecord &record)>;

FermiSurfacePayloadHeader read_fermi_surface_payload(const std::string &path,
                                                     const FermiSurfacePayloadConsumer &consumer);

void validate_fermi_surface_payload_reference(
    const FermiSurfacePayloadHeader &header, double expected_kbt_ha,
    double expected_chemical_potential_ha, std::int32_t expected_nspin,
    std::int32_t expected_nspinor, std::int32_t expected_basis_size,
    double expected_spin_degeneracy, double expected_cell_volume_bohr3, double tolerance);

}  // namespace driver
