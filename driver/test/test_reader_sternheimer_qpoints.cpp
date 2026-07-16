#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../reader_sternheimer_qpoints.h"

namespace
{

class TempDirectory
{
public:
    TempDirectory()
    {
        const auto suffix = std::to_string(std::rand());
        path = std::filesystem::temp_directory_path() / ("librpa_st_qpoints_" + suffix);
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

void write_text(const std::filesystem::path &path, const std::string &text)
{
    std::ofstream output(path);
    output << text;
}

template <typename Value>
void write_binary(std::ofstream &output, const Value &value)
{
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
}

void write_minimal_coulomb_v1(const std::filesystem::path &path, const int iq)
{
    constexpr std::int32_t marker = -20129433;
    constexpr std::int32_t naux = 1;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 1;
    constexpr std::int32_t nblocks = 1;
    constexpr std::int32_t pair_index = 0;
    constexpr std::int64_t payload_offset = 40;
    const std::complex<double> payload = {1.0, 0.0};

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, nblocks);
    write_binary(output, naux);
    write_binary(output, pair_index);
    write_binary(output, payload_offset);
    write_binary(output, payload);
}

void require_throws(const std::function<void()> &operation, const std::string &message_fragment)
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
    {
        if (std::string(error.what()).find(message_fragment) != std::string::npos)
        {
            return;
        }
        std::cerr << "unexpected error: " << error.what() << std::endl;
        std::abort();
    }
    std::cerr << "expected an exception containing: " << message_fragment << std::endl;
    std::abort();
}

void test_reads_normalized_manifest()
{
    TempDirectory temp;
    const auto manifest = temp.path / "qpoints.dat";
    write_text(manifest, "# iq qx qy qz qweight\n2 0.5 0 0 0.25\n3 0 0.5 0 0.75 ! row\n");

    const auto qpoints = driver::read_sternheimer_qpoint_manifest(manifest.string());
    assert(qpoints.size() == 2);
    assert(qpoints[0].iq == 2);
    assert((qpoints[0].q == std::array<double, 3>{0.5, 0.0, 0.0}));
    assert(std::abs(qpoints[0].weight - 0.25) < 1.0e-15);
    assert(qpoints[1].iq == 3);
    assert((qpoints[1].q == std::array<double, 3>{0.0, 0.5, 0.0}));
    assert(std::abs(qpoints[1].weight - 0.75) < 1.0e-15);
}

void test_rejects_duplicate_iq()
{
    TempDirectory temp;
    const auto manifest = temp.path / "qpoints.dat";
    write_text(manifest, "2 0.5 0 0 0.5\n2 -0.5 0 0 0.5\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(manifest.string()); },
                   "duplicate iq=2");
}

void test_rejects_unnormalized_or_nonpositive_weights()
{
    TempDirectory temp;
    const auto unnormalized = temp.path / "unnormalized.dat";
    write_text(unnormalized, "2 0.5 0 0 0.25\n3 0 0.5 0 0.5\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(unnormalized.string()); },
                   "weights must sum to 1");

    const auto nonpositive = temp.path / "nonpositive.dat";
    write_text(nonpositive, "2 0.5 0 0 1.0\n3 0 0.5 0 0.0\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(nonpositive.string()); },
                   "positive q weight");
}

void test_rejects_missing_coulomb_iq()
{
    TempDirectory temp;
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws(
        [&]()
        {
            driver::validate_sternheimer_qpoint_input_files(
                qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1);
        },
        "No Coulomb v1 files found");
}

void test_rejects_missing_frequency_files()
{
    TempDirectory temp;
    write_minimal_coulomb_v1(temp.path / "v1_coulomb_full_iq_2", 2);
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws(
        [&]()
        {
            driver::validate_sternheimer_qpoint_input_files(
                qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1);
        },
        "No Sternheimer chi0 v1 files found");
}

}  // namespace

int main()
{
    test_reads_normalized_manifest();
    test_rejects_duplicate_iq();
    test_rejects_unnormalized_or_nonpositive_weights();
    test_rejects_missing_coulomb_iq();
    test_rejects_missing_frequency_files();
    return 0;
}
