#pragma once

#include <cstddef>
#include <complex>
#include <string>
#include <vector>

#include "../../src/core/ri.h"

#include "reader_context.h"
namespace librpa::reader
{

struct HartreeCoulombQ0
{
    int dimension = 0;
    std::vector<std::complex<double>> values;  // full row-major matrix, Hartree
};

// Root-local input operation; does not insert or replace the handler's GW V(q).
HartreeCoulombQ0 read_hartree_coulomb_q0(ReaderContext &ctx, const std::string &dir_path,
                                         const std::string &prefix, int reader_version,
                                         bool use_shrink_basis);

bool check_coulomb_file_binary(const std::string &file_path);

int detect_coulomb_reader_version(const std::string &dir_path,
                                  const std::string &vq_fprefix);

size_t read_Vq_full(ReaderContext &ctx, const std::string &dir_path, const std::string &vq_fprefix,
                    bool is_cut_coulomb, int reader_version = 0,
                    bool use_shrink_basis = false);

size_t read_Vq_row(ReaderContext &ctx, const std::string &dir_path, const std::string &vq_fprefix,
                   double threshold, const std::vector<librpa_int::atpair_t> &local_atpair,
                   bool is_cut_coulomb, int reader_version = 0,
                   bool use_shrink_basis = false);

}
