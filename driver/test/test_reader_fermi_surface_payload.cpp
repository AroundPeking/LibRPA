#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../reader_fermi_surface_payload.h"

namespace
{

class TempDirectory
{
public:
    TempDirectory()
    {
        path = std::filesystem::temp_directory_path() /
               ("librpa_fermi_payload_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

template <typename Value>
void write_binary(std::ofstream &output, const Value &value)
{
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
}

template <typename Value>
void overwrite_binary(const std::filesystem::path &path, const std::streamoff offset,
                      const Value &value)
{
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(offset);
    file.write(reinterpret_cast<const char *>(&value), sizeof(Value));
}

void write_complex(std::ofstream &output, const std::complex<double> value)
{
    write_binary(output, value.real());
    write_binary(output, value.imag());
}

void write_record(std::ofstream &output, const std::int64_t k_index, const std::int32_t band,
                  const std::array<double, 3> &kfrac, const double energy,
                  const double minus_fprime, const std::complex<double> marker)
{
    write_binary(output, k_index);
    write_binary(output, std::int32_t{0});
    write_binary(output, band);
    for (const double value : kfrac) write_binary(output, value);
    write_binary(output, energy);
    write_binary(output, minus_fprime);
    for (int direction = 0; direction != 3; ++direction)
        write_complex(output, marker + std::complex<double>{static_cast<double>(direction),
                                                            -static_cast<double>(direction)});
    write_complex(output, marker);
    write_complex(output, marker + std::complex<double>{1.0, -1.0});
}

void write_valid_payload(const std::filesystem::path &path)
{
    constexpr char magic[8] = {'F', 'S', 'P', 'A', 'Y', 'L', '1', '\0'};
    constexpr std::int32_t version = 1;
    constexpr std::int32_t header_bytes = 116;
    constexpr std::int64_t record_count = 2;
    constexpr std::int64_t record_bytes = 136;
    constexpr double kbt = 0.25;
    constexpr double chemical_potential = 0.0;
    constexpr double cutoff_kbt = 10.0;
    const double minus_fprime = 1.0 / (4.0 * kbt);

    std::ofstream output(path, std::ios::binary);
    output.write(magic, sizeof(magic));
    write_binary(output, version);
    write_binary(output, header_bytes);
    write_binary(output, record_count);
    write_binary(output, std::int32_t{2});
    write_binary(output, std::int32_t{1});
    write_binary(output, std::int32_t{1});
    write_binary(output, std::int32_t{1});
    write_binary(output, std::int32_t{1});
    write_binary(output, std::int32_t{2});
    write_binary(output, std::int32_t{0});
    write_binary(output, record_bytes);
    write_binary(output, kbt);
    write_binary(output, chemical_potential);
    write_binary(output, cutoff_kbt);
    write_binary(output, 2.0);
    write_binary(output, 100.0);
    write_binary(output, 0.99);
    write_binary(output, 0.98);
    write_record(output, 0, 0, {0.0, 0.0, 0.0}, 0.0, minus_fprime, {1.0, 2.0});
    write_record(output, 1, 1, {0.5, 0.0, 0.0}, 0.0, minus_fprime, {3.0, 4.0});
}

void require_throws(const std::function<void()> &operation, const std::string &fragment)
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
    {
        if (std::string(error.what()).find(fragment) != std::string::npos) return;
        std::cerr << "unexpected error: " << error.what() << std::endl;
        std::abort();
    }
    std::cerr << "expected an exception containing: " << fragment << std::endl;
    std::abort();
}

void test_streams_version_one_records()
{
    TempDirectory temp;
    const auto path = temp.path / "fermi_surface_payload.v1";
    write_valid_payload(path);

    std::vector<driver::FermiSurfacePayloadRecord> records;
    const auto header = driver::read_fermi_surface_payload(
        path.string(), [&](const auto &record) { records.push_back(record); });

    const std::array<std::int32_t, 3> expected_mesh{2, 1, 1};
    const std::array<double, 3> expected_kfrac{0.5, 0.0, 0.0};
    assert(header.record_count == 2);
    assert(header.mesh == expected_mesh);
    assert(header.nspinor == 1);
    assert(header.basis_size == 2);
    assert(records.size() == 2);
    assert(records[0].k_index == 0);
    assert(records[1].k_index == 1);
    assert(records[1].band == 1);
    assert(records[1].kfrac == expected_kfrac);
    assert(records[1].velocity[0] == std::complex<double>(3.0, 4.0));
    assert(records[1].wfc.size() == 2);
    assert(records[1].wfc[1] == std::complex<double>(4.0, 3.0));
}

void test_rejects_trailing_bytes()
{
    TempDirectory temp;
    const auto path = temp.path / "trailing.v1";
    write_valid_payload(path);
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output.put('\0');
    output.close();

    require_throws([&]()
                   { driver::read_fermi_surface_payload(path.string(), [](const auto &) {}); },
                   "file size");
}

void test_rejects_inconsistent_fermi_derivative()
{
    TempDirectory temp;
    const auto path = temp.path / "bad_fprime.v1";
    write_valid_payload(path);
    constexpr std::streamoff header_bytes = 116;
    constexpr std::streamoff fprime_offset_in_record = 48;
    overwrite_binary(path, header_bytes + fprime_offset_in_record, 0.5);

    require_throws([&]()
                   { driver::read_fermi_surface_payload(path.string(), [](const auto &) {}); },
                   "-df/de");
}

void test_rejects_duplicate_state_key()
{
    TempDirectory temp;
    const auto path = temp.path / "duplicate.v1";
    write_valid_payload(path);
    constexpr std::streamoff header_bytes = 116;
    constexpr std::streamoff record_bytes = 136;
    overwrite_binary(path, header_bytes + record_bytes, std::int64_t{0});
    overwrite_binary(path, header_bytes + record_bytes + 12, std::int32_t{0});

    require_throws([&]()
                   { driver::read_fermi_surface_payload(path.string(), [](const auto &) {}); },
                   "strict state order");
}

void test_rejects_payload_from_a_different_thermal_reference()
{
    TempDirectory temp;
    const auto path = temp.path / "reference.v1";
    write_valid_payload(path);
    const auto header = driver::read_fermi_surface_payload(path.string(), [](const auto &) {});

    driver::validate_fermi_surface_payload_reference(header, 0.25, 0.0, 1, 1, 2, 2.0, 100.0,
                                                     1.0e-12);
    require_throws(
        [&]()
        {
            driver::validate_fermi_surface_payload_reference(header, 0.25, 0.01, 1, 1, 2, 2.0,
                                                             100.0, 1.0e-12);
        },
        "chemical potential");
}

}  // namespace

int main(const int argc, char **argv)
{
    if (argc == 2)
    {
        std::int64_t records = 0;
        const auto header =
            driver::read_fermi_surface_payload(argv[1], [&](const auto &) { ++records; });
        std::cout << std::setprecision(17) << "records " << records << "\n"
                  << "mesh " << header.mesh[0] << " " << header.mesh[1] << " " << header.mesh[2]
                  << "\n"
                  << "basis_size " << header.basis_size << "\n"
                  << "kbt_ha " << header.kbt_ha << "\n"
                  << "chemical_potential_ha " << header.chemical_potential_ha << "\n";
        return 0;
    }
    if (argc != 1) return 2;
    test_streams_version_one_records();
    test_rejects_trailing_bytes();
    test_rejects_inconsistent_fermi_derivative();
    test_rejects_duplicate_state_key();
    test_rejects_payload_from_a_different_thermal_reference();
    return 0;
}
