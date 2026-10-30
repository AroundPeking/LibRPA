#pragma once

#include <map>
#include <vector>

#include "../math/matrix_m.h"
#include "../mpi/base_mpi.h"
#include "atomic_basis.h"
#include "pbc.h"
#include "ri.h"

namespace librpa_int
{

// Uses the official LibRI Hartree kernel on a complete regular BvK mesh.
// Cs blocks have unique MPI ownership and shape {naux_I,nao_I,nao_J}.
// The supplied q=0 bare Coulomb metric must match that auxiliary basis and
// the caller's electrostatic convention; no EXX head correction is removed here.
// weighted_density_k and coulomb_q0 are replicated.  Density can be a total
// physical density or a signed density difference.  Results are replicated,
// column-major spatial AO operators in the original k-point order, in Hartree
// when coulomb_q0 is in Hartree.  Both LRI endpoints are included by LibRI.
// Optional target points use inverse mesh Fourier followed by the same
// atom-pair BvK remapping and forward Fourier convention as EXX band projection.
// For a shifted mesh, phases are measured from its first point.
std::map<int, Matz> compute_hartree_ao_k(
    const AtomicBasis& basis_wfc, const AtomicBasis& basis_aux, const Cs_LRI& cs,
    const Matz& coulomb_q0, const PeriodicBoundaryData& pbc,
    const std::vector<Matz>& weighted_density_k, const MpiCommHandler& comm_h,
    const std::vector<Vector3_Order<double>>* evaluation_kpoints = nullptr,
    const AtomPairBvKRemap<atom_t>& bvk_remap = {});

}  // namespace librpa_int
