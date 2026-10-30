#pragma once

#include <string>
#include <vector>

#include "../../math/vector3_order.h"
#include "../meanfield.h"

namespace librpa_int
{
namespace qsgw
{

void validate_projection_target(const MeanField& reference,
                                const std::vector<Vector3_Order<double>>& kpoints,
                                int expected_n_spins, int expected_n_spinors, int expected_n_aos,
                                const std::string& label);

// Validate every live E/C/f frame before symmetry use. The historical
// unpolarized scalar full-grid route retains its existing spatial-symmetry
// algorithm; explicit spin operations, spinful storage and IBZ-expanded sources
// additionally require the strict magnetic sewing covariance contract.
void validate_live_symmetry_state(const MeanField& meanfield, const AtomicBasis& basis,
                                  const PeriodicBoundaryData& pbc,
                                  const SymmetryContext& symmetry, bool expanded_from_ibz);

} // namespace qsgw
} // namespace librpa_int
