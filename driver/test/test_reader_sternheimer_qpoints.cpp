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
#include <utility>
#include <vector>

#include "../reader_sternheimer.h"
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

void write_minimal_sternheimer_v1(const std::filesystem::path &path, const int iq, const int ifreq,
                                  const double omega, const double weight,
                                  const std::complex<double> payload)
{
    constexpr std::int32_t marker = -41073291;
    constexpr std::int32_t naux = 1;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 1;
    constexpr std::int32_t nblocks = 1;
    constexpr std::int32_t pair_index = 0;
    constexpr std::int64_t payload_offset = 60;

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, static_cast<std::int32_t>(ifreq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, omega);
    write_binary(output, weight);
    write_binary(output, nblocks);
    write_binary(output, naux);
    write_binary(output, pair_index);
    write_binary(output, payload_offset);
    write_binary(output, payload);
}

struct CoulombBlock
{
    std::int32_t pair_index;
    std::complex<double> value;
};

void write_two_atom_coulomb_v1_shard(const std::filesystem::path &path, const int iq,
                                     const std::vector<CoulombBlock> &blocks)
{
    constexpr std::int32_t marker = -20129433;
    constexpr std::int32_t naux = 2;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 2;
    constexpr std::int32_t atom_naux = 1;
    const auto nblocks = static_cast<std::int32_t>(blocks.size());
    const std::int64_t payload_start = 6 * sizeof(std::int32_t) + natoms * sizeof(std::int32_t) +
                                       nblocks * (sizeof(std::int32_t) + sizeof(std::int64_t));

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, nblocks);
    write_binary(output, atom_naux);
    write_binary(output, atom_naux);
    for (std::size_t iblock = 0; iblock != blocks.size(); ++iblock)
    {
        write_binary(output, blocks[iblock].pair_index);
        write_binary(output, payload_start +
                                 static_cast<std::int64_t>(iblock * sizeof(std::complex<double>)));
    }
    for (const auto &block : blocks)
    {
        write_binary(output, block.value);
    }
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

void test_skips_missing_gamma_files_when_gamma_is_excluded()
{
    TempDirectory temp;
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 1.0}};

    driver::validate_sternheimer_qpoint_input_files(
        qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1, false);
}

void test_partial_mode_requires_coulomb_but_not_aggregate_response_files()
{
    TempDirectory temp;
    write_minimal_coulomb_v1(temp.path / "v1_coulomb_full_iq_2", 2);
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};

    driver::validate_sternheimer_partial_qpoint_input_files(qpoints, temp.path.string(),
                                                            "v1_coulomb_full_iq_");
}

void test_requires_gamma_manifest_row_when_gamma_is_excluded()
{
    const std::vector<driver::SternheimerQPoint> without_gamma{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws([&]() { driver::validate_sternheimer_gamma_contract(without_gamma, false); },
                   "exactly one Gamma row");

    const std::vector<driver::SternheimerQPoint> with_gamma{{1, {0.0, 0.0, 0.0}, 0.25},
                                                            {2, {0.5, 0.0, 0.0}, 0.75}};
    driver::validate_sternheimer_gamma_contract(with_gamma, false);
}

void test_reads_one_sternheimer_response_from_explicit_path()
{
    TempDirectory temp;
    const auto response_path = temp.path / "representative_k17_ifreq2.bin";
    write_minimal_sternheimer_v1(response_path, 3, 2, 0.75, 0.125, {-4.0, 0.0});

    const auto response = driver::read_sternheimer_chi0_v1_matrix_file(response_path.string());
    assert(response.path == response_path.string());
    assert(response.iq == 3);
    assert(response.ifreq == 2);
    assert(std::abs(response.omega - 0.75) < 1.0e-15);
    assert(std::abs(response.weight - 0.125) < 1.0e-15);
    assert((response.atom_naux == std::vector<int>{1}));
    assert(response.matrix.nr == 1);
    assert(response.matrix.nc == 1);
    assert(std::abs(response.matrix(0, 0) - std::complex<double>(-4.0, 0.0)) < 1.0e-15);
}

void test_merges_coulomb_atom_pair_blocks_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank3.dat", 2,
                                    {{1, {0.5, 0.25}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank9.dat", 2,
                                    {{2, {3.0, 0.0}}});

    const auto matrix =
        driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2);
    assert(matrix.nr == 2);
    assert(matrix.nc == 2);
    assert(std::abs(matrix(0, 0) - std::complex<double>(2.0, 0.0)) < 1.0e-15);
    assert(std::abs(matrix(0, 1) - std::complex<double>(0.5, 0.25)) < 1.0e-15);
    assert(std::abs(matrix(1, 0) - std::complex<double>(0.5, -0.25)) < 1.0e-15);
    assert(std::abs(matrix(1, 1) - std::complex<double>(3.0, 0.0)) < 1.0e-15);
}

void test_rejects_duplicate_coulomb_block_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}, {1, {0.5, 0.25}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank1.dat", 2,
                                    {{1, {0.5, 0.25}}, {2, {3.0, 0.0}}});

    require_throws(
        [&]()
        {
            static_cast<void>(
                driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2));
        },
        "duplicate atom-pair block across Coulomb v1 shards");
}

void test_rejects_missing_coulomb_block_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank1.dat", 2,
                                    {{1, {0.5, 0.25}}});

    require_throws(
        [&]() {
            driver::validate_coulomb_v1_full_matrix_file(temp.path.string(), "v1_coulomb_full_iq_",
                                                         2);
        },
        "missing atom-pair block across Coulomb v1 shards");
}

}  // namespace

int main()
{
    test_reads_normalized_manifest();
    test_rejects_duplicate_iq();
    test_rejects_unnormalized_or_nonpositive_weights();
    test_rejects_missing_coulomb_iq();
    test_rejects_missing_frequency_files();
    test_skips_missing_gamma_files_when_gamma_is_excluded();
    test_partial_mode_requires_coulomb_but_not_aggregate_response_files();
    test_requires_gamma_manifest_row_when_gamma_is_excluded();
    test_reads_one_sternheimer_response_from_explicit_path();
    test_merges_coulomb_atom_pair_blocks_across_rank_shards();
    test_rejects_duplicate_coulomb_block_across_rank_shards();
    test_rejects_missing_coulomb_block_across_rank_shards();
    return 0;
}
