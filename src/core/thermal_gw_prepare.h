#pragma once

#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#include "atomic_basis.h"
#include "pbc.h"
#include "qpoint_view.h"
#include "symmetry_context.h"
#include "thermal_gw_transform.h"

namespace librpa_int
{
using ThermalWcQMap = std::map<double, std::map<Vector3_Order<double>, Matz>>;

// Unfold W_full = U^dagger W_compressed U when U is supplied, restore q stars
// in the original ABF basis, then complete signed samples by adjoints. W stays
// distributed; no rank gathers a dense global screened matrix.
ThermalWcQMap prepare_thermal_wc(
    const MpiCommHandler &comm_h, const ThermalWcQMap &positive_wc, const PeriodicBoundaryData &pbc,
    const ThermalGWTransform &transform, const ArrayDesc &input_desc, const ArrayDesc &output_desc,
    const std::map<Vector3_Order<double>, ComplexMatrix> *sinvS = nullptr,
    const SymmetryQPointView *qpoints = nullptr, const SymmetryContext *symmetry = nullptr,
    const AtomicBasis *full_basis = nullptr);
}  // namespace librpa_int
