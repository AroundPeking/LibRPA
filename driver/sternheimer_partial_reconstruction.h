#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include "reader_sternheimer_partial.h"
#include "reader_sternheimer_qpoints.h"
#include "../src/core/atomic_basis.h"
#include "../src/core/sternheimer_rpa.h"
#include "../src/core/sternheimer_symmetry.h"
#include "../src/math/vector3_order.h"

namespace librpa_int
{
struct SymmetryContext;
}

namespace driver
{

struct SternheimerReconstructedResponse
{
    int iq = 0;
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    double q_weight = 0.0;
    int full_k_count = 0;
    int representative_k_count = 0;
    int little_group_order = 0;
    librpa_int::ComplexMatrix matrix;
    std::vector<librpa_int::SternheimerQStarResponse> qstar_responses;
};

struct SternheimerQStarRpaAudit
{
    librpa_int::SternheimerRpaFrequencyResult result;
    int qstar_size = 0;
    double max_integrand_difference = 0.0;
};

std::vector<SternheimerReconstructedResponse> reconstruct_sternheimer_partial_responses(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const std::vector<SternheimerQPoint> &qpoints,
    const SternheimerPartialResponseGroups &groups,
    int expected_nfreq,
    bool use_rpa_gamma,
    int lmax);

SternheimerQStarRpaAudit compute_sternheimer_qstar_rpa_frequency(
    const SternheimerReconstructedResponse &response,
    const std::vector<librpa_int::SternheimerQStarResponse> &coulomb_qstar,
    double sqrt_coulomb_threshold,
    double invariance_tolerance = 1.0e-10);

}  // namespace driver
