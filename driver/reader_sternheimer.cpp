#include "reader_sternheimer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../src/io/fs.h"

namespace driver
{
namespace
{

constexpr std::int32_t kCoulombV1Marker = -20129433;
constexpr std::int32_t kSternheimerChi0V1Marker = -41073291;
constexpr std::int32_t kRealFlag = 0;
constexpr std::int32_t kComplexFlag = 1;
constexpr std::int64_t kMissingBlock = -1;

struct BlockRecord
{
    int pair_index = -1;
    std::int64_t offset = kMissingBlock;
};

struct BlockedMatrixFile
{
    std::string path;
    int marker = 0;
    int iq = 0;
    int ifreq = 0;
    int naux = 0;
    int value_flag = kComplexFlag;
    int natoms = 0;
    int nblocks = 0;
    double omega = 0.0;
    double weight = 1.0;
    std::vector<int> atom_naux;
    std::vector<BlockRecord> blocks;
};

template <typename Value>
Value read_scalar(std::ifstream &input, const std::string &path)
{
    Value value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(Value));
    if (!input.good())
    {
        throw std::runtime_error("Failed to read binary scalar from " + path);
    }
    return value;
}

std::size_t checked_matrix_size(const int nrow, const int ncol, const std::string &context)
{
    if (nrow < 0 || ncol < 0)
    {
        throw std::runtime_error(context + ": negative matrix dimension");
    }
    const auto rows = static_cast<std::size_t>(nrow);
    const auto cols = static_cast<std::size_t>(ncol);
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows)
    {
        throw std::runtime_error(context + ": matrix size overflows size_t");
    }
    return rows * cols;
}

std::size_t upper_pair_index(const std::size_t iatom, const std::size_t jatom,
                             const std::size_t natoms)
{
    if (iatom > jatom)
    {
        throw std::runtime_error("upper_pair_index expects I <= J");
    }
    return iatom * natoms - iatom * (iatom - 1) / 2 + (jatom - iatom);
}

std::vector<std::pair<std::size_t, std::size_t>> make_atom_pairs(const int natoms)
{
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const auto n = static_cast<std::size_t>(natoms);
    pairs.reserve(n * (n + 1) / 2);
    for (std::size_t iatom = 0; iatom != n; ++iatom)
    {
        for (std::size_t jatom = iatom; jatom != n; ++jatom)
        {
            pairs.emplace_back(iatom, jatom);
        }
    }
    return pairs;
}

std::vector<int> make_atom_offsets(const std::vector<int> &atom_naux)
{
    std::vector<int> offsets(atom_naux.size() + 1, 0);
    for (std::size_t iatom = 0; iatom != atom_naux.size(); ++iatom)
    {
        if (atom_naux[iatom] <= 0)
        {
            throw std::runtime_error("Invalid non-positive per-atom auxiliary size");
        }
        offsets[iatom + 1] = offsets[iatom] + atom_naux[iatom];
    }
    return offsets;
}

BlockedMatrixFile read_coulomb_header(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    BlockedMatrixFile file;
    file.path = path;
    file.marker = read_scalar<std::int32_t>(input, path);
    file.iq = read_scalar<std::int32_t>(input, path);
    file.naux = read_scalar<std::int32_t>(input, path);
    file.value_flag = read_scalar<std::int32_t>(input, path);
    file.natoms = read_scalar<std::int32_t>(input, path);
    file.nblocks = read_scalar<std::int32_t>(input, path);
    if (file.marker != kCoulombV1Marker)
    {
        throw std::runtime_error(path + ": invalid Coulomb v1 marker");
    }
    return file;
}

BlockedMatrixFile read_sternheimer_header(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    BlockedMatrixFile file;
    file.path = path;
    file.marker = read_scalar<std::int32_t>(input, path);
    file.iq = read_scalar<std::int32_t>(input, path);
    file.ifreq = read_scalar<std::int32_t>(input, path);
    file.naux = read_scalar<std::int32_t>(input, path);
    file.value_flag = read_scalar<std::int32_t>(input, path);
    file.natoms = read_scalar<std::int32_t>(input, path);
    file.omega = read_scalar<double>(input, path);
    file.weight = read_scalar<double>(input, path);
    file.nblocks = read_scalar<std::int32_t>(input, path);
    if (file.marker != kSternheimerChi0V1Marker)
    {
        throw std::runtime_error(path + ": invalid Sternheimer chi0 v1 marker");
    }
    return file;
}

void read_atom_sizes_and_blocks(BlockedMatrixFile &file, std::ifstream &input)
{
    if (file.iq <= 0 || file.naux <= 0 || file.natoms <= 0 || file.nblocks < 0)
    {
        throw std::runtime_error(file.path + ": invalid v1 matrix dimensions");
    }
    if (file.value_flag != kRealFlag && file.value_flag != kComplexFlag)
    {
        throw std::runtime_error(file.path + ": invalid v1 value flag");
    }

    file.atom_naux.resize(static_cast<std::size_t>(file.natoms));
    int naux_sum = 0;
    for (int &atom_aux : file.atom_naux)
    {
        atom_aux = read_scalar<std::int32_t>(input, file.path);
        if (atom_aux <= 0)
        {
            throw std::runtime_error(file.path + ": non-positive atom_naux entry");
        }
        naux_sum += atom_aux;
    }
    if (naux_sum != file.naux)
    {
        throw std::runtime_error(file.path + ": atom_naux does not sum to naux");
    }

    const auto npairs =
        static_cast<std::size_t>(file.natoms) * (static_cast<std::size_t>(file.natoms) + 1) / 2;
    if (static_cast<std::size_t>(file.nblocks) > npairs)
    {
        throw std::runtime_error(file.path + ": block count exceeds atom-pair count");
    }
    file.blocks.resize(static_cast<std::size_t>(file.nblocks));
    std::vector<bool> seen(npairs, false);
    for (BlockRecord &block : file.blocks)
    {
        block.pair_index = read_scalar<std::int32_t>(input, file.path);
        block.offset = read_scalar<std::int64_t>(input, file.path);
        if (block.pair_index < 0 || static_cast<std::size_t>(block.pair_index) >= npairs)
        {
            throw std::runtime_error(file.path + ": invalid atom-pair index");
        }
        if (seen[static_cast<std::size_t>(block.pair_index)])
        {
            throw std::runtime_error(file.path + ": duplicate atom-pair block");
        }
        seen[static_cast<std::size_t>(block.pair_index)] = true;
    }
}

BlockedMatrixFile read_coulomb_file_metadata(const std::string &path)
{
    auto file = read_coulomb_header(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    input.seekg(6 * static_cast<std::streamoff>(sizeof(std::int32_t)));
    read_atom_sizes_and_blocks(file, input);
    return file;
}

BlockedMatrixFile read_sternheimer_file_metadata(const std::string &path)
{
    auto file = read_sternheimer_header(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    input.seekg(6 * static_cast<std::streamoff>(sizeof(std::int32_t)) +
                2 * static_cast<std::streamoff>(sizeof(double)) +
                static_cast<std::streamoff>(sizeof(std::int32_t)));
    read_atom_sizes_and_blocks(file, input);
    return file;
}

std::vector<std::complex<double>> read_block_payload(std::ifstream &input,
                                                     const BlockedMatrixFile &file,
                                                     const BlockRecord &block,
                                                     const std::size_t nvalues)
{
    input.seekg(static_cast<std::streamoff>(block.offset));
    if (!input.good())
    {
        throw std::runtime_error(file.path + ": failed to seek atom-pair block");
    }

    std::vector<std::complex<double>> values(nvalues);
    if (file.value_flag == kComplexFlag)
    {
        input.read(reinterpret_cast<char *>(values.data()),
                   static_cast<std::streamsize>(nvalues * sizeof(std::complex<double>)));
    }
    else
    {
        std::vector<double> buffer(nvalues);
        input.read(reinterpret_cast<char *>(buffer.data()),
                   static_cast<std::streamsize>(nvalues * sizeof(double)));
        for (std::size_t i = 0; i != nvalues; ++i)
        {
            values[i] = {buffer[i], 0.0};
        }
    }
    if (!input.good())
    {
        throw std::runtime_error(file.path + ": failed to read atom-pair block payload");
    }
    return values;
}

librpa_int::ComplexMatrix read_dense_blocked_matrix(const BlockedMatrixFile &file)
{
    const auto atom_pairs = make_atom_pairs(file.natoms);
    const auto atom_offsets = make_atom_offsets(file.atom_naux);
    librpa_int::ComplexMatrix matrix(file.naux, file.naux);
    std::ifstream input(file.path.c_str(), std::ios::binary);

    for (const BlockRecord &block : file.blocks)
    {
        const auto [iatom, jatom] = atom_pairs[static_cast<std::size_t>(block.pair_index)];
        const int ioffset = atom_offsets[iatom];
        const int joffset = atom_offsets[jatom];
        const int inaux = file.atom_naux[iatom];
        const int jnaux = file.atom_naux[jatom];
        const auto nvalues = checked_matrix_size(inaux, jnaux, file.path);
        const auto values = read_block_payload(input, file, block, nvalues);

        for (int imu = 0; imu != inaux; ++imu)
        {
            for (int jmu = 0; jmu != jnaux; ++jmu)
            {
                const auto value =
                    values[static_cast<std::size_t>(imu) * static_cast<std::size_t>(jnaux) +
                           static_cast<std::size_t>(jmu)];
                matrix(ioffset + imu, joffset + jmu) = value;
                if (iatom != jatom)
                {
                    matrix(joffset + jmu, ioffset + imu) = std::conj(value);
                }
            }
        }
    }
    return matrix;
}

BlockedMatrixFile find_single_coulomb_file(const std::string &dir_path, const std::string &prefix,
                                           const int iq)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, prefix);
    if (files.empty())
    {
        throw std::runtime_error("No Coulomb v1 files found with prefix " + prefix);
    }

    std::vector<BlockedMatrixFile> matches;
    for (const auto &path : files)
    {
        const auto file = read_coulomb_file_metadata(path);
        if (file.iq == iq)
        {
            matches.push_back(file);
        }
    }
    if (matches.empty())
    {
        throw std::runtime_error("No Coulomb v1 file found for iq=" + std::to_string(iq));
    }
    if (matches.size() > 1)
    {
        throw std::runtime_error("Multiple Coulomb v1 files found for iq=" + std::to_string(iq));
    }
    return matches.front();
}

std::vector<BlockedMatrixFile> find_sternheimer_files(const std::string &dir_path,
                                                      const std::string &prefix, const int iq)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, prefix);
    if (files.empty())
    {
        throw std::runtime_error("No Sternheimer chi0 v1 files found with prefix " + prefix);
    }

    std::vector<BlockedMatrixFile> matches;
    for (const auto &path : files)
    {
        auto metadata = read_sternheimer_file_metadata(path);
        if (metadata.iq != iq)
        {
            continue;
        }
        if (metadata.value_flag != kComplexFlag)
        {
            throw std::runtime_error(path + ": Sternheimer chi0 v1 must be complex-valued");
        }
        matches.push_back(std::move(metadata));
    }
    if (matches.empty())
    {
        throw std::runtime_error("No Sternheimer chi0 v1 file found for iq=" + std::to_string(iq));
    }

    std::sort(matches.begin(), matches.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.ifreq < rhs.ifreq; });
    for (std::size_t i = 0; i != matches.size(); ++i)
    {
        if (matches[i].ifreq != static_cast<int>(i + 1))
        {
            throw std::runtime_error("Sternheimer chi0 v1 files are not contiguous in ifrequency");
        }
    }
    return matches;
}

}  // namespace

librpa_int::ComplexMatrix read_coulomb_v1_full_matrix(const std::string &dir_path,
                                                      const std::string &prefix, const int iq)
{
    const auto file = find_single_coulomb_file(dir_path, prefix, iq);
    return read_dense_blocked_matrix(file);
}

void validate_coulomb_v1_full_matrix_file(const std::string &dir_path, const std::string &prefix,
                                          const int iq)
{
    static_cast<void>(find_single_coulomb_file(dir_path, prefix, iq));
}

std::vector<SternheimerChi0V1Matrix> read_sternheimer_chi0_v1_matrices(const std::string &dir_path,
                                                                       const std::string &prefix,
                                                                       const int iq)
{
    const auto files = find_sternheimer_files(dir_path, prefix, iq);
    std::vector<SternheimerChi0V1Matrix> responses;
    responses.reserve(files.size());
    for (const auto &metadata : files)
    {
        SternheimerChi0V1Matrix response;
        response.path = metadata.path;
        response.iq = metadata.iq;
        response.ifreq = metadata.ifreq;
        response.omega = metadata.omega;
        response.weight = metadata.weight;
        response.atom_naux = metadata.atom_naux;
        response.matrix = read_dense_blocked_matrix(metadata);
        responses.push_back(std::move(response));
    }
    return responses;
}

void validate_sternheimer_chi0_v1_files(const std::string &dir_path, const std::string &prefix,
                                        const int iq, const int expected_nfreq)
{
    const auto files = find_sternheimer_files(dir_path, prefix, iq);
    if (static_cast<int>(files.size()) != expected_nfreq)
    {
        throw std::runtime_error("Sternheimer chi0 frequency count (" +
                                 std::to_string(files.size()) + ") for iq=" + std::to_string(iq) +
                                 " does not match nfreq (" + std::to_string(expected_nfreq) + ")");
    }
}

}  // namespace driver
