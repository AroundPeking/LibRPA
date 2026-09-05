#include "reader_fermi_surface_payload.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "../src/core/thermal_occupation.h"

namespace driver
{
namespace
{

template <typename Value>
Value read_binary(std::ifstream &input, const std::string &path)
{
    Value value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(Value));
    if (!input)
    {
        throw std::runtime_error(path + ": truncated Fermi-surface payload");
    }
    return value;
}

std::complex<double> read_complex(std::ifstream &input, const std::string &path)
{
    const double real = read_binary<double>(input, path);
    const double imag = read_binary<double>(input, path);
    return {real, imag};
}

bool host_is_little_endian()
{
    const std::uint16_t one = 1;
    return *reinterpret_cast<const unsigned char *>(&one) == 1;
}

}  // namespace

FermiSurfacePayloadHeader read_fermi_surface_payload(const std::string &path,
                                                     const FermiSurfacePayloadConsumer &consumer)
{
    constexpr std::array<char, 8> expected_magic{'F', 'S', 'P', 'A', 'Y', 'L', '1', '\0'};
    constexpr std::int32_t expected_version = 1;
    constexpr std::int32_t expected_header_bytes = 116;

    if (!host_is_little_endian())
    {
        throw std::runtime_error("Fermi-surface payload v1 requires a little-endian host");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open Fermi-surface payload: " + path);
    }

    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    const auto version = read_binary<std::int32_t>(input, path);
    const auto header_bytes = read_binary<std::int32_t>(input, path);
    if (magic != expected_magic || version != expected_version ||
        header_bytes != expected_header_bytes)
    {
        throw std::runtime_error(path + ": unsupported Fermi-surface payload header");
    }

    FermiSurfacePayloadHeader header;
    header.record_count = read_binary<std::int64_t>(input, path);
    for (auto &value : header.mesh) value = read_binary<std::int32_t>(input, path);
    header.nspin = read_binary<std::int32_t>(input, path);
    header.nspinor = read_binary<std::int32_t>(input, path);
    header.basis_size = read_binary<std::int32_t>(input, path);
    const auto reserved = read_binary<std::int32_t>(input, path);
    header.record_bytes = read_binary<std::int64_t>(input, path);
    header.kbt_ha = read_binary<double>(input, path);
    header.chemical_potential_ha = read_binary<double>(input, path);
    header.cutoff_kbt = read_binary<double>(input, path);
    header.spin_degeneracy = read_binary<double>(input, path);
    header.cell_volume_bohr3 = read_binary<double>(input, path);
    header.retained_kappa_fraction = read_binary<double>(input, path);
    header.retained_plasma_trace_fraction = read_binary<double>(input, path);
    if (reserved != 0)
    {
        throw std::runtime_error(path + ": unsupported Fermi-surface payload flags");
    }
    if (header.record_count <= 0 || header.mesh[0] <= 0 || header.mesh[1] <= 0 ||
        header.mesh[2] <= 0 || header.nspin <= 0 || header.nspinor <= 0 || header.basis_size <= 0)
    {
        throw std::runtime_error(path + ": invalid Fermi-surface payload dimensions");
    }
    if (!std::isfinite(header.kbt_ha) || header.kbt_ha <= 0.0 ||
        !std::isfinite(header.chemical_potential_ha) || !std::isfinite(header.cutoff_kbt) ||
        header.cutoff_kbt <= 0.0 ||
        (header.spin_degeneracy != 1.0 && header.spin_degeneracy != 2.0) ||
        !std::isfinite(header.cell_volume_bohr3) || header.cell_volume_bohr3 <= 0.0 ||
        !std::isfinite(header.retained_kappa_fraction) || header.retained_kappa_fraction <= 0.0 ||
        header.retained_kappa_fraction > 1.0 ||
        !std::isfinite(header.retained_plasma_trace_fraction) ||
        header.retained_plasma_trace_fraction <= 0.0 || header.retained_plasma_trace_fraction > 1.0)
    {
        throw std::runtime_error(path + ": invalid Fermi-surface payload metadata");
    }
    const auto maximum_size = std::numeric_limits<std::int64_t>::max();
    const auto spinor_stride = 16 * static_cast<std::int64_t>(header.nspinor);
    if (header.basis_size > (maximum_size - 104) / spinor_stride)
    {
        throw std::runtime_error(path + ": invalid Fermi-surface payload dimensions");
    }
    const std::int64_t expected_record_bytes =
        104 + spinor_stride * static_cast<std::int64_t>(header.basis_size);
    if (header.record_bytes != expected_record_bytes ||
        header.record_count > (maximum_size - expected_header_bytes) / header.record_bytes)
    {
        throw std::runtime_error(path + ": invalid Fermi-surface payload record size");
    }
    const auto expected_file_bytes = static_cast<std::uintmax_t>(
        expected_header_bytes + header.record_count * header.record_bytes);
    if (std::filesystem::file_size(path) != expected_file_bytes)
    {
        throw std::runtime_error(path + ": Fermi-surface payload file size is inconsistent");
    }

    if (static_cast<std::int64_t>(header.mesh[0]) > maximum_size / header.mesh[1] ||
        static_cast<std::int64_t>(header.mesh[0]) * header.mesh[1] > maximum_size / header.mesh[2])
    {
        throw std::runtime_error(path + ": invalid Fermi-surface payload mesh size");
    }
    const std::int64_t total_kpoints =
        static_cast<std::int64_t>(header.mesh[0]) * header.mesh[1] * header.mesh[2];
    std::tuple<std::int64_t, std::int32_t, std::int32_t> previous_key{-1, -1, -1};
    for (std::int64_t index = 0; index != header.record_count; ++index)
    {
        FermiSurfacePayloadRecord record;
        record.k_index = read_binary<std::int64_t>(input, path);
        record.spin = read_binary<std::int32_t>(input, path);
        record.band = read_binary<std::int32_t>(input, path);
        for (auto &value : record.kfrac) value = read_binary<double>(input, path);
        if (record.k_index < 0 || record.k_index >= total_kpoints || record.spin < 0 ||
            record.spin >= header.nspin || record.band < 0 || record.band >= header.basis_size)
        {
            throw std::runtime_error(path + ": Fermi-surface state index is out of range");
        }
        const auto key = std::make_tuple(record.k_index, record.spin, record.band);
        if (key <= previous_key)
        {
            throw std::runtime_error(path + ": records do not have strict state order");
        }
        previous_key = key;
        const std::int64_t yz = static_cast<std::int64_t>(header.mesh[1]) * header.mesh[2];
        const std::array<double, 3> expected_kfrac{
            static_cast<double>(record.k_index / yz) / header.mesh[0],
            static_cast<double>((record.k_index / header.mesh[2]) % header.mesh[1]) /
                header.mesh[1],
            static_cast<double>(record.k_index % header.mesh[2]) / header.mesh[2]};
        for (int direction = 0; direction != 3; ++direction)
        {
            if (!std::isfinite(record.kfrac[direction]) ||
                std::abs(record.kfrac[direction] - expected_kfrac[direction]) > 1.0e-12)
            {
                throw std::runtime_error(path + ": Fermi-surface k point is inconsistent");
            }
        }
        record.energy_ha = read_binary<double>(input, path);
        record.minus_fprime_ha_inv = read_binary<double>(input, path);
        if (!std::isfinite(record.energy_ha) || !std::isfinite(record.minus_fprime_ha_inv) ||
            record.minus_fprime_ha_inv < 0.0)
        {
            throw std::runtime_error(path + ": invalid Fermi-surface energy or -df/de");
        }
        const double expected_minus_fprime = -librpa_int::fermi_dirac_derivative(
            record.energy_ha - header.chemical_potential_ha, header.kbt_ha);
        const double derivative_tolerance = 2.0e-12 * std::max(1.0, expected_minus_fprime);
        if (std::abs(record.minus_fprime_ha_inv - expected_minus_fprime) > derivative_tolerance)
        {
            throw std::runtime_error(path + ": Fermi-surface -df/de is inconsistent");
        }
        const double scaled_energy =
            (record.energy_ha - header.chemical_potential_ha) / header.kbt_ha;
        if (std::abs(scaled_energy) > header.cutoff_kbt + 1.0e-12)
        {
            throw std::runtime_error(path + ": Fermi-surface state is outside its energy window");
        }
        for (auto &value : record.velocity)
        {
            value = read_complex(input, path);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            {
                throw std::runtime_error(path + ": nonfinite Fermi-surface velocity");
            }
        }
        record.wfc.resize(static_cast<std::size_t>(header.nspinor) *
                          static_cast<std::size_t>(header.basis_size));
        for (auto &value : record.wfc)
        {
            value = read_complex(input, path);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            {
                throw std::runtime_error(path + ": nonfinite Fermi-surface wavefunction");
            }
        }
        consumer(record);
    }
    return header;
}

void validate_fermi_surface_payload_reference(
    const FermiSurfacePayloadHeader &header, const double expected_kbt_ha,
    const double expected_chemical_potential_ha, const std::int32_t expected_nspin,
    const std::int32_t expected_nspinor, const std::int32_t expected_basis_size,
    const double expected_spin_degeneracy, const double expected_cell_volume_bohr3,
    const double tolerance)
{
    if (!std::isfinite(expected_kbt_ha) || expected_kbt_ha <= 0.0 ||
        !std::isfinite(expected_chemical_potential_ha) || expected_nspin <= 0 ||
        expected_nspinor <= 0 || expected_basis_size <= 0 ||
        (expected_spin_degeneracy != 1.0 && expected_spin_degeneracy != 2.0) ||
        !std::isfinite(expected_cell_volume_bohr3) || expected_cell_volume_bohr3 <= 0.0 ||
        !std::isfinite(tolerance) || tolerance < 0.0)
    {
        throw std::invalid_argument(
            "Fermi-surface payload reference tolerance must be nonnegative and finite");
    }
    if (std::abs(header.kbt_ha - expected_kbt_ha) > tolerance)
    {
        throw std::runtime_error("Fermi-surface payload temperature is inconsistent");
    }
    if (std::abs(header.chemical_potential_ha - expected_chemical_potential_ha) > tolerance)
    {
        throw std::runtime_error("Fermi-surface payload chemical potential is inconsistent");
    }
    if (header.nspin != expected_nspin || header.nspinor != expected_nspinor ||
        header.basis_size != expected_basis_size ||
        header.spin_degeneracy != expected_spin_degeneracy)
    {
        throw std::runtime_error("Fermi-surface payload basis or spin convention is inconsistent");
    }
    const double volume_scale =
        std::max({1.0, std::abs(header.cell_volume_bohr3), std::abs(expected_cell_volume_bohr3)});
    if (std::abs(header.cell_volume_bohr3 - expected_cell_volume_bohr3) > tolerance * volume_scale)
    {
        throw std::runtime_error("Fermi-surface payload cell volume is inconsistent");
    }
}

}  // namespace driver
