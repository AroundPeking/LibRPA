#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "../../src/core/ri.h"

#include "reader_context.h"
namespace librpa::reader
{

struct HartreeLriBlock
{
    int atom_i = 0, atom_j = 0;
    std::array<int, 3> translation{};
    int n_aos_i = 0, n_aos_j = 0, n_aux_i = 0;
    std::vector<double> values;
};

// Root-local, lossless binary-v1 input with physical R retained. Does not set GW Cs.
std::vector<HartreeLriBlock> read_hartree_lri_blocks(ReaderContext &ctx,
                                                     const std::string &dir_path,
                                                     const std::string &prefix,
                                                     bool use_shrink_basis);

int detect_Cs_reader_version(ReaderContext &ctx, const std::string &dir_path,
                             const std::string keyword = "Cs_data");

size_t read_Cs(ReaderContext &ctx, const std::string &dir_path, double threshold,
               const std::vector<librpa_int::atpair_t> &local_atpair,
               const std::string keyword = "Cs_data",
               int reader_version = 0);

size_t read_Cs_evenly_distribute(ReaderContext &ctx, const std::string &dir_path, double threshold, int myid,
                                 int nprocs, const std::string keyword = "Cs_data",
                                 int reader_version = 0);

void get_natom_ncell_from_first_Cs_file(ReaderContext &ctx, int &n_atom, int &n_cell,
                                        const std::string &dir_path);

std::vector<size_t> read_aux_basis_from_Cs(ReaderContext &ctx, const std::string &dir_path,
                                           const std::string &keyword);

// Fallback method to read basis dimensions from Cs files, for early dataset versions
void read_basis_from_Cs(ReaderContext &ctx, const std::string &dir_path);

}
