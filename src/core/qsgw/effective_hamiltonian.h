#pragma once

#include "../meanfield.h"
#include "matrix_map.h"

namespace librpa_int
{
namespace qsgw
{

SpinKMatrixMap build_reference_hamiltonian(
    const MeanField& reference);

SpinKMatrixMap assemble_effective_hamiltonian(const SpinKMatrixMap& reference_hamiltonian,
                                              const SpinKMatrixMap& dft_vxc,
                                              const SpinKMatrixMap& exchange,
                                              const SpinKMatrixMap& correlation,
                                              const SpinKMatrixMap* delta_hartree = nullptr);

// The QP equation subtracts the exchange/correlation static operator already
// contained in the entering Hamiltonian. Its explicit Hartree feedback remains.
SpinKMatrixMap build_qp_static_subtraction(const SpinKMatrixMap& entering_hamiltonian,
                                           const SpinKMatrixMap& reference_hamiltonian,
                                           const SpinKMatrixMap& dft_vxc,
                                           const SpinKMatrixMap* delta_hartree = nullptr);

} // namespace qsgw
} // namespace librpa_int
