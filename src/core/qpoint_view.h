/*!
 * @file qpoint_view.h
 * @brief Derived q-point views for response and screened-Coulomb calculations.
 */
#pragma once

#include <map>
#include <vector>

#include "../math/vector3_order.h"

namespace librpa_int
{

class PeriodicBoundaryData;
struct SymmetryContext;

enum class SymmetryQPointRestoreMode
{
    NONE,
    TIME_REVERSAL,
    FULL_CRYSTAL,
};

//! Radial partitions used by the strict-2D Gamma quadrature diagnostics.
enum class Strict2dQshellRegion { gamma, first, rest };
enum class Strict2dQradialRegion { gamma_or_first, near, middle, far };

Vector3_Order<double> strict_2d_minimum_image_q(const PeriodicBoundaryData &pbc,
                                                const Vector3_Order<double> &qfrac);
Strict2dQshellRegion classify_strict_2d_qshell(double q_norm, double first_q_norm);
Strict2dQradialRegion classify_strict_2d_qradial(double q_norm, double first_q_norm);
bool strict_2d_qradial_is_corner(double q_norm, double first_q_norm);

struct SymmetryQPointView
{
    SymmetryQPointRestoreMode restore_mode = SymmetryQPointRestoreMode::NONE;
    std::vector<Vector3_Order<double>> representatives;
    std::map<Vector3_Order<double>, std::vector<Vector3_Order<double>>> members;
    std::map<Vector3_Order<double>, double> weights;
};

SymmetryQPointView build_symmetry_qpoint_view(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc,
    bool use_symmetry);

} // namespace librpa_int
