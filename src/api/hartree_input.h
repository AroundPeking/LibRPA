#pragma once

#include <vector>

#include "dataset.h"

namespace librpa_int
{
// Physical spin-summed D_ij(k) = sum_{sigma,tau,n} wg_sigma(k,n)
// C_{i,tau,n,sigma}(k) conj(C_{j,tau,n,sigma}(k)). Stored wg already
// contains the integration weight; no EXX spin factor or occupation cutoff.
// Returns replicated owning matrices and leaves the mean field unchanged.
std::vector<Matz> build_hartree_density_k(Dataset& dataset);

// Collect the explicitly supplied, bare periodic q=0 metric. The caller owns
// its G=0 convention; an EXX singularity-corrected metric is not interchangeable.
Matz collect_hartree_coulomb_q0(Dataset& dataset, bool use_shrink);
}  // namespace librpa_int
