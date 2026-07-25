#include "../core/symmetry_context.h"
#include "../core/pbc.h"
#include "../core/qpoint_view.h"
#include "../math/rsh.h"
#include "../utils/constants.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "testutils.h"
#include "../io/stl_io_helper.h"

namespace {

using namespace std;
using namespace librpa_int;

void assert_matrix_close(const ComplexMatrix& actual,
                         const ComplexMatrix& expected,
                         const double thres = 1e-12)
{
    assert(actual.nr == expected.nr);
    assert(actual.nc == expected.nc);
    for (int row = 0; row < actual.nr; ++row)
    {
        for (int col = 0; col < actual.nc; ++col)
        {
            assert(fequal(actual(row, col),
                          expected(row, col),
                          std::complex<double>(thres, 0.0)));
        }
    }
}

void test_kspace_shell_rotations_use_direct_rotation()
{
    const BasisConvention basis_convention{-1,
                                           0,
                                           LIBRPA_ANGULAR_ORDER_NATURAL,
                                           LIBRPA_RSH_COEFF_1_M,
                                           LIBRPA_RSH_COEFF_1_M};
    const Matrix3 cartesian_rotation(0.0, -1.0, 0.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0);
    const Matrix3 skew_lattice(2.0, 0.0, 0.0,
                               0.5, 1.5, 0.0,
                               0.2, 0.3, 2.0);
    const Matrix3 fractional_rotation =
        skew_lattice * cartesian_rotation.Transpose() * skew_lattice.Inverse();
    const SpaceGroupSymOp op{fractional_rotation, {0.0, 0.0, 0.0}};

    const auto shell_rotations =
        build_symmetry_shell_rotations_from_direct_rotation(
            op, skew_lattice, 1, basis_convention);
    assert(shell_rotations.size() == 2);

    const auto expected_p_rotation =
        real_spherical_harmonic_rotation_matrix(cartesian_rotation,
                                                1,
                                                basis_convention.order,
                                                basis_convention.coeff_m_negative,
                                                basis_convention.coeff_m_positive);
    assert_matrix_close(shell_rotations.at(1), expected_p_rotation);

    const Vector3_Order<double> k_source{0.25, 0.0, 0.0};
    const Vector3_Order<double> k_target{0.25, 0.0, 0.0};
    const Vector3_Order<double> atom_from{0.0, 0.0, 0.0};
    const Vector3_Order<double> atom_to{0.0, 0.0, 0.0};
    const Vector3_Order<int> return_lattice{1, 0, 0};
    const auto phase = build_symmetry_kspace_phase(
        k_source, k_target, atom_from, atom_to, return_lattice, basis_convention);
    assert(fequal(phase, std::complex<double>(0.0, 1.0), std::complex<double>(1e-12, 0.0)));

    const auto phased_shell_rotations =
        build_symmetry_kspace_shell_rotations(op,
                                                    skew_lattice,
                                                    0,
                                                    basis_convention,
                                                    k_source,
                                                    k_target,
                                                    atom_from,
                                                    atom_to,
                                                    return_lattice);
    assert(phased_shell_rotations.at(0).nr == 1);
    assert(phased_shell_rotations.at(0).nc == 1);
    assert(fequal(phased_shell_rotations.at(0)(0, 0),
                  phase,
                  std::complex<double>(1e-12, 0.0)));
}

void test_kstar_member_return_lattice_preserves_input_fractional_representative()
{
    SymmetryContext ctx;
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}, {1, 1}},
                              {{0, {0.0, 0.0, 0.0}}, {1, {-0.25, -0.25, -0.25}}});
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};

    SymmetryOperation op;
    op.rotation = {0, 0, -1,
                   1, 0, -1,
                   0, 1, -1};
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    ctx.rspace_operations.push_back(op);

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    ctx.kstars.push_back(star);

    ctx.build_kstar_member_rotations(0);
    assert(ctx.kspace_return_lattice.at({1, 0}) == Vector3_Order<int>(0, 0, 1));
}

void test_kstar_member_atom_mapping_tolerates_text_coordinate_noise()
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(lattice,
                              lattice,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};

    SymmetryOperation op;
    op.rotation = Matrix3(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    op.translation = {1.0 + 1.1e-5, 0.0, 0.0};
    op.use_row_convention = true;
    ctx.rspace_operations.push_back(op);

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    ctx.kstars.push_back(star);

    ctx.build_kstar_member_rotations(0);
    assert(ctx.kspace_return_lattice.at({0, 0}) == Vector3_Order<int>(1, 0, 0));
    assert(ctx.kstars.at(0).members.at(0).atom_rotations.at(0).atom_to == 0);
}

void test_kstar_member_atom_mapping_tolerates_si_fractional_roundoff()
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice,
        lattice,
        {{0, 0}, {1, 0}},
        {{0, {0.12500258782275, 0.12500258782275, 0.12500258782275}},
         {1, {0.87501811475925, 0.87501811475925, 0.87501811475925}}});
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};

    SymmetryOperation op;
    op.rotation = Matrix3(0.0, -1.0, 0.0,
                          1.0, -1.0, 0.0,
                          0.0, -1.0, 1.0);
    op.translation = {0.0, 0.5, -1.0};
    op.use_row_convention = true;
    ctx.rspace_operations.push_back(op);

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    ctx.kstars.push_back(star);

    ctx.build_kstar_member_rotations(0);
    assert(ctx.kspace_return_lattice.at({0, 0}) == Vector3_Order<int>(0, 0, -1));
    assert(ctx.kspace_return_lattice.at({1, 0}) == Vector3_Order<int>(0, -3, -1));
    assert(ctx.kstars.at(0).members.at(0).atom_rotations.at(0).atom_to == 0);
    assert(ctx.kstars.at(0).members.at(0).atom_rotations.at(1).atom_to == 1);
}

void test_kspace_rotation_derivative_matches_bloch_phase_difference()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    pbc.set_irreducible_kgrids_kvec(
        3, 1, 1,
        {0.0, 0.0, 0.0,
         TWO_PI / 3.0, 0.0, 0.0},
        {{{0.0, 0.0, 0.0}},
         {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}});

    SymmetryContext ctx;
    ctx.set_crystal_structure(
        pbc.latvec, pbc.G,
        {{0, 0}, {1, 0}},
        {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});
    SymmetryOperation identity;
    identity.rotation.Identity();
    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0);
    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             1,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);

    const auto &star = find_symmetry_kstar_for_ibz_kpoint(ctx, {1.0 / 3.0, 0.0, 0.0});
    const auto member_iter = std::find_if(
        star.members.begin(), star.members.end(), [](const SymmetryKStarMember &member) {
            return !member.time_reversal && std::abs(member.k_bz.x + 1.0 / 3.0) < 1e-12;
        });
    assert(member_iter != star.members.end());
    const auto &member = *member_iter;

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const auto derivative = build_symmetry_kspace_rotation_matrix_derivatives(
        ctx, {layout}, member, atom_nw, star.k_ibz, false, nullptr);

    const double delta_kfrac = 1e-6;
    auto shifted_member = [&](const double shift) {
        auto shifted = member;
        shifted.k_bz.x += shift;
        const Vector3_Order<double> shifted_kibz{star.k_ibz.x - shift, 0.0, 0.0};
        for (auto &atom_rotation : shifted.atom_rotations)
        {
            const auto return_lattice = ctx.kspace_return_lattice.at(
                {atom_rotation.atom_from, shifted.spatial_isym});
            atom_rotation.bloch_rsh_rotations = build_symmetry_kspace_shell_rotations(
                ctx.rspace_operations.at(static_cast<size_t>(shifted.spatial_isym)),
                ctx.lattice_vectors,
                atom_rotation.lmax,
                ctx.basis_convention,
                shifted.k_bz,
                shifted_kibz,
                Vector3_Order<double>{ctx.input_coord_frac.at(atom_rotation.atom_from)},
                Vector3_Order<double>{ctx.input_coord_frac.at(atom_rotation.atom_to)},
                return_lattice);
        }
        return std::pair{std::move(shifted), shifted_kibz};
    };
    const auto [member_plus, kibz_plus] = shifted_member(delta_kfrac);
    const auto [member_minus, kibz_minus] = shifted_member(-delta_kfrac);
    const auto rotation_plus = build_symmetry_kspace_rotation_matrix(
        ctx, {layout}, member_plus, atom_nw, kibz_plus, false, nullptr);
    const auto rotation_minus = build_symmetry_kspace_rotation_matrix(
        ctx, {layout}, member_minus, atom_nw, kibz_minus, false, nullptr);
    for (int row = 0; row != 2; ++row)
    {
        for (int col = 0; col != 2; ++col)
        {
            const auto finite_difference =
                (rotation_plus(row, col) - rotation_minus(row, col)) /
                (2.0 * TWO_PI * delta_kfrac);
            assert(fequal(derivative[0](row, col), finite_difference,
                          std::complex<double>(1e-7, 0.0)));
        }
    }
}

void test_species_basis_layout_keeps_shell_order()
{
    SpeciesBasisLayout layout;
    assert(!layout.is_shell_available());
    layout.label = "X";
    layout.set({1, 0});
    assert(layout.is_shell_available());
    assert(layout.n_ao == 4);
    assert(layout.shell_counts.at(1) == 1);
    assert(layout.shell_indices.at(1).front() == 0);
    assert(layout.shell_indices.at(0).front() == 1);

    ComplexMatrix p_rotation(3, 3);
    p_rotation.zero_out();
    p_rotation(0, 0) = {2.0, 0.0};
    p_rotation(1, 1) = {3.0, 0.0};
    p_rotation(2, 2) = {4.0, 0.0};
    ComplexMatrix s_rotation(1, 1);
    s_rotation.zero_out();
    s_rotation(0, 0) = {7.0, 0.0};

    const auto rotation =
        build_symmetry_rotation_matrix(layout, {{1, p_rotation}, {0, s_rotation}});

    ComplexMatrix expected(4, 4);
    expected.zero_out();
    expected(0, 0) = {2.0, 0.0};
    expected(1, 1) = {3.0, 0.0};
    expected(2, 2) = {4.0, 0.0};
    expected(3, 3) = {7.0, 0.0};
    assert_matrix_close(rotation, expected);
}

void test_species_layouts_match_basis_dimensions()
{
    const std::map<atom_t, int> atom_to_type{{0, 0}, {1, 0}};

    SpeciesBasisLayout full_layout;
    full_layout.label = "full";
    full_layout.set({1, 0});

    SpeciesBasisLayout shrink_layout;
    shrink_layout.label = "shrink";
    shrink_layout.set({0});

    assert(symmetry_species_layouts_match_atom_counts(
        {full_layout}, atom_to_type, {{0, 4}, {1, 4}}));
    assert(!symmetry_species_layouts_match_atom_counts(
        {full_layout}, atom_to_type, {{0, 1}, {1, 1}}));
    assert(symmetry_species_layouts_match_atom_counts(
        {shrink_layout}, atom_to_type, {{0, 1}, {1, 1}}));
}

void test_atomic_basis_builds_symmetry_species_layouts()
{
    AtomicBasis basis(std::vector<std::size_t>{1, 3});
    basis.label = "WFC";
    basis.set_l_shells({{0}, {1}});

    const auto layouts = basis.build_species_basis_layouts({{0, 0}, {1, 1}});

    assert(get_symmetry_species_layout(layouts, 0).n_ao == 1);
    assert(get_symmetry_species_layout(layouts, 1).n_ao == 3);
}

void test_periodic_mappings_store_full_kpoint_members()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    assert(ctx.kstars.size() == 2);
    assert(ctx.kstar_grid_mapping.size() == 2);
    assert(ctx.full_kpoint_members.size() == 3);
    assert(ctx.full_kpoint_members[0].ik_ibz == 0);
    assert(ctx.full_kpoint_members[1].ik_ibz == 1);
    assert(ctx.full_kpoint_members[2].ik_ibz == 1);
    std::size_t restored_rspace_members = 0;
    for (const auto& pair_stars : ctx.rspace_sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_rspace_members += R_star.second.size();
        }
    }
    assert(restored_rspace_members == pbc.Rlist.size());
}

void test_periodic_mappings_accept_kq_reduced_coulomb_grid()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    assert(ctx.kstars.size() == 2);
    assert(ctx.kstar_grid_mapping.size() == 3);
    assert(ctx.kstar_grid_mapping[0].iq_ibz == 0);
    assert(ctx.kstar_grid_mapping[1].iq_ibz == 1);
    assert(ctx.kstar_grid_mapping[2].iq_ibz == 2);
    assert(ctx.kstar_grid_mapping[0].star_list_index == 0);
    assert(ctx.kstar_grid_mapping[1].star_list_index == 1);
    assert(ctx.kstar_grid_mapping[2].star_list_index == 1);
    assert(ctx.kstar_grid_mapping[0].member_q_bz_keys.size() == 1);
    assert(ctx.kstar_grid_mapping[1].member_q_bz_keys.size() == 2);
    assert(ctx.kstar_grid_mapping[2].member_q_bz_keys.size() == 2);
}

void test_qpoint_view_uses_pbc_without_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);

    SymmetryContext ctx;
    const auto view = build_symmetry_qpoint_view(ctx, pbc, false);
    assert(view.restore_mode == SymmetryQPointRestoreMode::NONE);
    assert(view.representatives.size() == 3);
    assert(std::abs(view.weights.at(pbc.klist_coul[0]) - 1.0 / 3.0) < 1e-12);
}

void test_qpoint_view_preserves_time_reversal_mapping_without_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    const auto view = build_symmetry_qpoint_view(ctx, pbc, false);
    assert(view.restore_mode == SymmetryQPointRestoreMode::TIME_REVERSAL);
    assert(view.representatives.size() == 2);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_preserves_time_reversal_q_reduction_with_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    pbc.set_kq_mapping({0, 1, 1});

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::TIME_REVERSAL);
    assert(view.representatives == pbc.klist_coul);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_rejects_reduced_input_without_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    bool threw = false;
    try
    {
        SymmetryContext ctx;
        (void)build_symmetry_qpoint_view(ctx, pbc, false);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    assert(threw);
}

void test_qpoint_view_accepts_reduced_input_with_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives == pbc.klist_coul);
    assert(view.members.at(pbc.klist_coul[1]).size() == 2);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 2.0 / 3.0) < 1e-12);
}

void test_qpoint_view_keeps_full_input_grid_without_parsed_crystal_symmetry()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        2.0 * librpa_int::TWO_PI / 3.0, 0.0, 0.0,
    };
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::NONE);
    assert(view.representatives == pbc.klist_coul);
    assert(view.representatives.size() == 3);
    assert(std::abs(view.weights.at(pbc.klist_coul[1]) - 1.0 / 3.0) < 1e-12);
}

void add_irreducible_sector_entry(symmetry_irreducible_sector_t& sector,
                                  const atom_t atom_i,
                                  const atom_t atom_j,
                                  const symmetry_R_t& R)
{
    sector[{atom_i, atom_j}].insert(R);
}

SymmetryOperation make_row_symmetry_operation(const std::array<int, 9>& rotation)
{
    SymmetryOperation op;
    op.rotation = Matrix3(rotation[0], rotation[1], rotation[2],
                                      rotation[3], rotation[4], rotation[5],
                                      rotation[6], rotation[7], rotation[8]);
    op.translation = {0.0, 0.0, 0.0};
    op.use_row_convention = true;
    return op;
}

void test_symmetry_context_saves_fractional_row_operations()
{
    SymmetryContext ctx;
    const Matrix3 col_rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const Vector3_Order<double> translation{0.25, -0.5, 1.0 / 3.0};
    SymmetryOperation op;
    op.rotation = col_rotation;
    op.translation = translation;
    op.use_row_convention = false;

    const SpaceGroupSymOp original_op{col_rotation, translation, false};
    const Vector3_Order<double> coord{0.2, 0.3, 0.4};
    const auto expected = apply_space_group_symmetry_operation(original_op, coord);

    ctx.add_rspace_operation(op);
    assert(!ctx.available);
    ctx.set_available();
    assert(ctx.available);
    assert(ctx.rspace_operations.size() == 1);

    const auto& saved = ctx.rspace_operations[0];
    assert(saved.use_row_convention);
    assert(saved.rotation == col_rotation.Transpose());
    assert(fequal(saved.translation.x, translation.x));
    assert(fequal(saved.translation.y, translation.y));
    assert(fequal(saved.translation.z, translation.z));

    const auto actual = apply_space_group_symmetry_operation(saved, coord);
    assert(fequal(actual.x, expected.x));
    assert(fequal(actual.y, expected.y));
    assert(fequal(actual.z, expected.z));
}

SymmetryContext make_bn_shrink_symmetry_context()
{
    SymmetryContext ctx;
    ctx.atom_to_type = {{0, 0}, {1, 1}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.25, 0.25, 0.25}}};

    const std::vector<std::array<int, 9>> rotations = {
        { 1,  0,  0,  0,  1,  0,  0,  0,  1},
        { 0, -1,  0,  1, -1,  0,  0, -1,  1},
        {-1,  1,  0, -1,  0,  0, -1,  0,  1},
        {-1,  0,  1, -1,  0,  0, -1,  1,  0},
        { 1,  0,  0,  0,  0,  1,  0,  1,  0},
        { 0,  0, -1,  1,  0, -1,  0,  1, -1},
        { 0,  1, -1,  1,  0, -1,  0,  0, -1},
        {-1,  0,  1, -1,  1,  0, -1,  0,  0},
        { 1, -1,  0,  0, -1,  1,  0, -1,  0},
        { 0, -1,  0,  0, -1,  1,  1, -1,  0},
        { 0,  1, -1,  0,  0, -1,  1,  0, -1},
        { 0,  0,  1,  0,  1,  0,  1,  0,  0},
        { 1,  0, -1,  0,  0, -1,  0,  1, -1},
        { 0,  0,  1,  1,  0,  0,  0,  1,  0},
        {-1,  0,  0, -1,  0,  1, -1,  1,  0},
        {-1,  1,  0, -1,  0,  1, -1,  0,  0},
        { 1,  0, -1,  0,  1, -1,  0,  0, -1},
        { 0, -1,  1,  1, -1,  0,  0, -1,  0},
        { 0,  0, -1,  0,  1, -1,  1,  0, -1},
        { 0, -1,  1,  0, -1,  0,  1, -1,  0},
        { 0,  1,  0,  0,  0,  1,  1,  0,  0},
        {-1,  0,  0, -1,  1,  0, -1,  0,  1},
        { 0,  1,  0,  1,  0,  0,  0,  0,  1},
        { 1, -1,  0,  0, -1,  0,  0, -1,  1},
    };
    for (const auto& rotation : rotations)
    {
        ctx.rspace_operations.push_back(make_row_symmetry_operation(rotation));
    }
    return ctx;
}

void add_mgo_fractional_symmetry_operations(SymmetryContext& ctx)
{
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    const std::array<int, 2> signs{{-1, 1}};

    for (const auto& permutation : permutations)
    {
        for (const int sx : signs)
        {
            for (const int sy : signs)
            {
                for (const int sz : signs)
                {
                    const std::array<int, 3> sign{{sx, sy, sz}};
                    std::array<int, 9> rotation{};
                    for (int col = 0; col != 3; ++col)
                    {
                        rotation[3 * permutation[col] + col] = sign[col];
                    }

                    const Matrix3 col_cartesian_rotation(
                        rotation[0], rotation[1], rotation[2],
                        rotation[3], rotation[4], rotation[5],
                        rotation[6], rotation[7], rotation[8]);
                    const Matrix3 row_cartesian_rotation = col_cartesian_rotation.Transpose();
                    SymmetryOperation op;
                    op.rotation = ctx.lattice_available
                                      ? ctx.lattice_vectors * row_cartesian_rotation *
                                            ctx.lattice_vectors.Inverse()
                                      : row_cartesian_rotation;
                    op.translation = {0.0, 0.0, 0.0};
                    op.use_row_convention = true;
                    ctx.rspace_operations.push_back(op);
                }
            }
        }
    }
    assert(ctx.rspace_operations.size() == 48);
}

void set_mgo_primitive_structure(SymmetryContext& ctx,
                                 const std::map<atom_t, int>& atom_to_type,
                                 const std::map<atom_t, coord_t>& coord_frac)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});
    ctx.set_crystal_structure(pbc.latvec, pbc.G, atom_to_type, coord_frac);
}

void test_single_atom_qpoint_view_reduces_full_input_to_crystal_qstars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}},
                                {{0, {0.0, 0.0, 0.0}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
    assert(view.representatives.size() == 4);
    std::size_t n_members = 0;
    double weight_sum = 0.0;
    for (const auto& q : view.representatives)
    {
        n_members += view.members.at(q).size();
        weight_sum += view.weights.at(q);
    }
    assert(n_members == pbc.klist_full.size());
    assert(std::abs(weight_sum - 1.0) < 1e-12);
}

void test_mgo_qpoint_view_reduces_time_reversal_q_list_to_crystal_qstars_for_full_input()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    std::vector<int> map_q_ks(pbc.kfrac_list.size(), -1);
    for (std::size_t ik = 0; ik != pbc.kfrac_list.size(); ++ik)
    {
        const auto minus_k =
            restrict_fractional_coordinate(Vector3_Order<double>{-pbc.kfrac_list[ik].x,
                                                                  -pbc.kfrac_list[ik].y,
                                                                  -pbc.kfrac_list[ik].z});
        int minus_ik = -1;
        for (std::size_t jk = 0; jk != pbc.kfrac_list.size(); ++jk)
        {
            if (same_fractional_kpoint(pbc.kfrac_list[jk], minus_k, 1e-5))
            {
                minus_ik = static_cast<int>(jk);
                break;
            }
        }
        assert(minus_ik >= 0);
        map_q_ks[ik] = std::min(static_cast<int>(ik), minus_ik);
    }
    pbc.set_kq_mapping(map_q_ks);
    assert(pbc.klist_coul.size() == 14);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
    assert(view.representatives.size() == 4);
    std::size_t n_members = 0;
    double weight_sum = 0.0;
    for (const auto& q : view.representatives)
    {
        n_members += view.members.at(q).size();
        weight_sum += view.weights.at(q);
    }
    assert(n_members == pbc.klist_full.size());
    assert(std::abs(weight_sum - 1.0) < 1e-12);
}

void test_mgo_keeps_all_kspace_operations_for_full_qstars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});

    SpaceGroupSymOps signed_permutation_operations;
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    const std::array<int, 2> signs{{-1, 1}};
    for (const auto& permutation : permutations)
    {
        for (const int sx : signs)
        {
            for (const int sy : signs)
            {
                for (const int sz : signs)
                {
                    const std::array<int, 3> sign{{sx, sy, sz}};
                    std::array<int, 9> rotation{};
                    for (int col = 0; col != 3; ++col)
                    {
                        rotation[3 * permutation[col] + col] = sign[col];
                    }
                    signed_permutation_operations.push_back(
                        make_row_symmetry_operation(rotation));
                }
            }
        }
    }

    ctx.set_rspace_operations(signed_permutation_operations);
    assert(ctx.rspace_operations.size() == 48);
    int metric_preserving_ops = 0;
    for (const auto& op : ctx.rspace_operations)
    {
        if (preserves_lattice_metric(op.rotation, ctx.lattice_vectors, 1e-6))
        {
            ++metric_preserving_ops;
        }
    }
    assert(metric_preserving_ops == 48);

    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    assert(ctx.kstars.size() == 4);
    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == ctx.kstars.size());
}

void test_kstar_routes_prefer_improper_spatial_operation_over_time_reversal()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec,
                              pbc.G,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY, SpaceGroupSymOp::INVERSE});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    bool checked_minus_q_member = false;
    for (const auto& star : ctx.kstars)
    {
        if (same_fractional_kpoint(star.k_ibz, {0.0, 0.0, 0.0}, 1e-5))
        {
            continue;
        }
        const auto minus_rep =
            restrict_fractional_coordinate(Vector3_Order<double>{-star.k_ibz.x,
                                                                  -star.k_ibz.y,
                                                                  -star.k_ibz.z});
        for (const auto& member : star.members)
        {
            if (!same_fractional_kpoint(member.k_bz, minus_rep, 1e-5)
                || same_fractional_kpoint(member.k_bz, star.k_ibz, 1e-5))
            {
                continue;
            }
            assert(member.spatial_isym == 1);
            assert(!member.time_reversal);
            checked_minus_q_member = true;
        }
    }
    assert(checked_minus_q_member);
}

ComplexMatrix make_test_dense_operator(const int n)
{
    ComplexMatrix matrix(n, n);
    for (int row = 0; row != n; ++row)
    {
        for (int col = 0; col != n; ++col)
        {
            matrix(row, col) =
                std::complex<double>{0.1 * (row + 1) + 0.03 * (col + 1),
                                     0.07 * (row - col)};
        }
    }
    return matrix;
}

librpa_int::symmetry_atom_block_matrix_map_t dense_to_atom_blocks(
    const ComplexMatrix& matrix,
    const std::map<atom_t, size_t>& atom_nabf)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    std::map<atom_t, int> offsets;
    int offset = 0;
    for (const auto& [atom, nbf] : atom_nabf)
    {
        offsets[atom] = offset;
        offset += static_cast<int>(nbf);
    }
    for (const auto& [atom_i, n_i_size] : atom_nabf)
    {
        const int n_i = static_cast<int>(n_i_size);
        for (const auto& [atom_j, n_j_size] : atom_nabf)
        {
            const int n_j = static_cast<int>(n_j_size);
            ComplexMatrix block(n_i, n_j);
            for (int i = 0; i != n_i; ++i)
            {
                for (int j = 0; j != n_j; ++j)
                {
                    block(i, j) = matrix(offsets.at(atom_i) + i,
                                         offsets.at(atom_j) + j);
                }
            }
            blocks[atom_i][atom_j] = block;
        }
    }
    return blocks;
}

ComplexMatrix atom_blocks_to_dense(
    const librpa_int::symmetry_atom_block_matrix_map_t& blocks,
    const std::map<atom_t, size_t>& atom_nabf)
{
    std::map<atom_t, int> offsets;
    int total = 0;
    for (const auto& [atom, nbf] : atom_nabf)
    {
        offsets[atom] = total;
        total += static_cast<int>(nbf);
    }
    ComplexMatrix matrix(total, total);
    for (const auto& [atom_i, row_blocks] : blocks)
    {
        for (const auto& [atom_j, block] : row_blocks)
        {
            const int row_offset = offsets.at(atom_i);
            const int col_offset = offsets.at(atom_j);
            for (int i = 0; i != block.nr; ++i)
            {
                for (int j = 0; j != block.nc; ++j)
                {
                    matrix(row_offset + i, col_offset + j) = block(i, j);
                }
            }
        }
    }
    return matrix;
}

ComplexMatrix rotate_dense_operator_from_transform_matrix(
    const ComplexMatrix& matrix,
    const ComplexMatrix& transform,
    const bool use_time_reversal)
{
    if (use_time_reversal)
    {
        return conj(transform) * conj(matrix) * transpose(transform, false);
    }
    return transform * matrix * transpose(transform, true);
}

void test_dense_kspace_rotation_matrix_orders_atom_swap_blocks()
{
    SymmetryContext ctx;
    ctx.rspace_operations.push_back(SpaceGroupSymOp::IDENTITY);

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};

    SymmetryKAtomRotation rot_0;
    rot_0.atom_from = 0;
    rot_0.atom_to = 1;
    rot_0.atom_type = 0;
    rot_0.lmax = 0;
    rot_0.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    rot_0.bloch_rsh_rotations[0](0, 0) = {2.0, 0.5};

    SymmetryKAtomRotation rot_1;
    rot_1.atom_from = 1;
    rot_1.atom_to = 0;
    rot_1.atom_type = 0;
    rot_1.lmax = 0;
    rot_1.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    rot_1.bloch_rsh_rotations[0](0, 0) = {3.0, -0.25};

    member.atom_rotations = {rot_0, rot_1};

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};

    const auto rotation = build_symmetry_kspace_rotation_matrix(
        ctx, {layout}, member, atom_nabf, {0.0, 0.0, 0.0});

    ComplexMatrix expected(2, 2);
    expected.zero_out();
    expected(0, 1) = rot_0.bloch_rsh_rotations.at(0)(0, 0);
    expected(1, 0) = rot_1.bloch_rsh_rotations.at(0)(0, 0);
    assert_matrix_close(rotation, expected);
}

void test_dense_operator_rotation_matches_atom_blocks_for_asymmetric_swap_phases()
{
    SymmetryContext ctx;
    ctx.rspace_operations.push_back(SpaceGroupSymOp::IDENTITY);

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.125, 0.125, 0.125};

    SymmetryKAtomRotation rot_0;
    rot_0.atom_from = 0;
    rot_0.atom_to = 1;
    rot_0.atom_type = 0;
    rot_0.lmax = 0;
    rot_0.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    rot_0.bloch_rsh_rotations[0](0, 0) =
        std::polar(1.0, -3.0 * PI / 4.0);

    SymmetryKAtomRotation rot_1;
    rot_1.atom_from = 1;
    rot_1.atom_to = 0;
    rot_1.atom_type = 0;
    rot_1.lmax = 0;
    rot_1.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    rot_1.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};
    member.atom_rotations = {rot_0, rot_1};

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    ComplexMatrix source(2, 2);
    source(0, 0) = {1.0, 0.0};
    source(0, 1) = {2.0, 3.0};
    source(1, 0) = {4.0, -5.0};
    source(1, 1) = {6.0, 0.0};

    const auto source_blocks = dense_to_atom_blocks(source, atom_nw);
    const auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
        ctx, {layout}, member, source_blocks, atom_nw, member.k_bz, false, nullptr,
        &member.k_bz);
    const auto expected = atom_blocks_to_dense(rotated_blocks, atom_nw);
    const auto transform = build_symmetry_kspace_operator_transform_matrix(
        ctx, {layout}, member, atom_nw, member.k_bz, false, &member.k_bz);
    const auto actual = rotate_dense_operator_from_transform_matrix(source, transform, false);
    assert_matrix_close(actual, expected);
}

void test_equivalent_kpoint_gauge_respects_bloch_atom_phase_convention()
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(lattice,
                              lattice,
                              {{0, 0}},
                              {{0, {0.25, 0.0, 0.0}}});
    ctx.rspace_operations.push_back(SpaceGroupSymOp::IDENTITY);

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {-1.0 / 3.0, 0.0, 0.0};
    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};
    member.atom_rotations.push_back(atom_rotation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    const Vector3_Order<double> canonical_target{2.0 / 3.0, 0.0, 0.0};

    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const auto rotation_without_atom_phase = build_symmetry_kspace_rotation_matrix(
        ctx, {layout}, member, atom_nw, member.k_bz, false, &canonical_target);
    assert(fequal(rotation_without_atom_phase(0, 0),
                  std::complex<double>(1.0, 0.0),
                  std::complex<double>(1e-12, 0.0)));

    ctx.basis_convention.bloch_ratom = 1;
    const auto rotation_with_atom_phase = build_symmetry_kspace_rotation_matrix(
        ctx, {layout}, member, atom_nw, member.k_bz, false, &canonical_target);
    assert(fequal(rotation_with_atom_phase(0, 0),
                  std::complex<double>(0.0, 1.0),
                  std::complex<double>(1e-12, 0.0)));
}

void test_mgo_dense_kspace_rotation_matches_atom_block_rotation()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            1);
    ctx.build_kstar_member_rotations(1);

    AtomicBasis basis(std::vector<std::size_t>{4, 4});
    basis.set_l_shells({{0, 1}, {0, 1}});
    const auto layouts = basis.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 4}, {1, 4}};
    const auto matrix = make_test_dense_operator(8);
    const auto blocks = dense_to_atom_blocks(matrix, atom_nabf);

    for (const auto& star : ctx.kstars)
    {
        for (const auto& member : star.members)
        {
            const auto transform = build_symmetry_kspace_operator_transform_matrix(
                ctx, layouts, member, atom_nabf, star.k_ibz, member.time_reversal,
                &member.k_bz);
            const auto rotated_dense = rotate_dense_operator_from_transform_matrix(
                matrix, transform, member.time_reversal);
            const auto rotated_blocks = rotate_symmetry_kspace_operator_blocks(
                ctx, layouts, member, blocks, atom_nabf, star.k_ibz,
                member.time_reversal, nullptr, &member.k_bz);
            assert_matrix_close(rotated_dense,
                                atom_blocks_to_dense(rotated_blocks, atom_nabf),
                                1e-10);
        }
    }
}

void test_mgo_k333_irreducible_sector_matches_single()
{
    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}},
                                {{0, {0.0, 0.0, 0.0}}});
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == Rlist.size());
}

void test_rspace_sector_star_member_isym_maps_full_to_ir_r()
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(lattice,
                              lattice,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    const auto c4 = make_row_symmetry_operation({0,  1, 0,
                                                -1,  0, 0,
                                                 0,  0, 1});
    const auto c4_inv = make_row_symmetry_operation({0, -1, 0,
                                                     1,  0, 0,
                                                     0,  0, 1});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY, c4, c4_inv});
    ctx.irreducible_sector.clear();
    add_irreducible_sector_entry(ctx.irreducible_sector, 0, 0, {1, 0, 0});
    ctx.set_available();

    const Vector3_Order<int> period{3, 3, 1};
    const std::vector<Vector3_Order<int>> Rlist{
        {1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);

    const auto& members = sector_stars.at({0, 0}).at({1, 0, 0});
    assert(members.size() == 3);
    bool checked_non_identity = false;
    for (const auto& member : members)
    {
        const auto rotated = multiply_row_vector(
            Vector3_Order<double>(static_cast<double>(member.full_R.x),
                                  static_cast<double>(member.full_R.y),
                                  static_cast<double>(member.full_R.z)),
            ctx.rspace_operations.at(member.isym).rotation);
        const Vector3_Order<int> expected{
            static_cast<int>(std::llround(rotated.x)),
            static_cast<int>(std::llround(rotated.y)),
            static_cast<int>(std::llround(rotated.z))};
        assert(expected == Vector3_Order<int>(1, 0, 0));
        checked_non_identity = checked_non_identity || member.isym != 0;
    }
    assert(checked_non_identity);
}

void test_rspace_block_restore_uses_stored_operation_rotation_convention()
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(lattice,
                              lattice,
                              {{0, 0}},
                              {{0, {0.0, 0.0, 0.0}}});
    ctx.basis_convention = {-1,
                            0,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const auto c4 = make_row_symmetry_operation({0,  1, 0,
                                                -1,  0, 0,
                                                 0,  0, 1});
    const auto c4_inv = make_row_symmetry_operation({0, -1, 0,
                                                     1,  0, 0,
                                                     0,  0, 1});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY, c4, c4_inv});
    ctx.build_rsh_rotations(ctx.basis_convention, 1);
    ctx.irreducible_sector.clear();
    add_irreducible_sector_entry(ctx.irreducible_sector, 0, 0, {1, 0, 0});
    ctx.set_available();

    const Vector3_Order<int> period{3, 3, 1};
    const std::vector<Vector3_Order<int>> Rlist{
        {1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);

    const auto& members = sector_stars.at({0, 0}).at({1, 0, 0});
    const SymmetryRSpaceRestoreMember* non_identity_member = nullptr;
    for (const auto& member : members)
    {
        if (member.isym != 0)
        {
            non_identity_member = &member;
            break;
        }
    }
    assert(non_identity_member != nullptr);

    const std::vector<SpeciesBasisLayout> layouts{{"X", {1}}};
    ComplexMatrix block_ir(3, 3);
    block_ir(0, 0) = 1.0;
    block_ir(0, 1) = 0.2;
    block_ir(0, 2) = -0.3;
    block_ir(1, 0) = 0.4;
    block_ir(1, 1) = 2.0;
    block_ir(1, 2) = 0.5;
    block_ir(2, 0) = -0.6;
    block_ir(2, 1) = 0.7;
    block_ir(2, 2) = 3.0;

    const auto T = ctx.get_rotation_matrix(layouts, 0, non_identity_member->isym);
    const auto expected_full = transpose(T, false) * block_ir * conj(T);
    const auto restored = rotate_symmetry_rspace_block(
        ctx, layouts, non_identity_member->isym, 0, 0, block_ir);
    assert_matrix_close(restored, expected_full);
}

void test_mgo_k333_irreducible_sector_matches_both()
{
    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ctx);

    const Vector3_Order<int> period{3, 3, 3};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  0,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1,  1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  1,  1});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 1,  1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  1});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == 2 * 2 * Rlist.size());
}

void test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry()
{
    auto ctx = make_bn_shrink_symmetry_context();
    const Vector3_Order<int> period{2, 2, 2};
    const auto Rlist = construct_R_grid(period);
    const auto generated_sector =
        build_symmetry_rspace_irreducible_sector(ctx, Rlist);

    symmetry_irreducible_sector_t expected_sector;
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 0, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 0, 1, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 0, { 0,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1, -1});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1, -1,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, {-1,  0,  0});
    add_irreducible_sector_entry(expected_sector, 1, 1, { 0,  0,  0});
    assert(generated_sector == expected_sector);

    ctx.irreducible_sector = generated_sector;
    ctx.set_available();
    symmetry_rspace_sector_stars_t sector_stars;
    build_symmetry_rspace_sector_stars(ctx, period, Rlist, sector_stars);
    std::size_t restored_members = 0;
    for (const auto& pair_stars : sector_stars)
    {
        for (const auto& R_star : pair_stars.second)
        {
            restored_members += R_star.second.size();
        }
    }
    assert(restored_members == 2 * 2 * Rlist.size());
}

} // namespace

void test_spin_operations_identity_translation()
{
    SymmetryContext ctx;
    std::vector<SymmetryOperation> operations;
    operations.push_back(make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}));
    operations.push_back(make_row_symmetry_operation({0, -1, 0, 1, 0, 0, 0, 0, 1}));
    operations.push_back(make_row_symmetry_operation({-1, 0, 0, 0, -1, 0, 0, 0, -1}));
    ctx.set_rspace_operations(operations);

    assert(ctx.operation_pool.size() == 3);
    assert(ctx.spin_operations.size() == 6);
    const std::array<std::complex<double>, 4> identity_u{1.0, 0.0, 0.0, 1.0};
    for (std::size_t isym = 0; isym != 3; ++isym)
    {
        const auto& unitary_op = ctx.spin_operations[isym];
        assert(unitary_op.spatial_id == isym);
        assert(!unitary_op.antiunitary);
        assert(unitary_op.spin_source == SymmetrySpinActionSource::Identity);
        assert(unitary_op.spin_u == identity_u);

        const auto& antiunitary_op = ctx.spin_operations[3 + isym];
        assert(antiunitary_op.spatial_id == isym);
        assert(antiunitary_op.antiunitary);
        assert(antiunitary_op.spin_source == SymmetrySpinActionSource::Identity);
        assert(antiunitary_op.spin_u == identity_u);
    }

    // reciprocal_rotation matches the dual rotation used by
    // apply_space_group_rotation_to_kpoint; the atom-map cache slots stay empty.
    const Vector3_Order<double> kpoint{0.2, 0.3, 0.4};
    for (std::size_t isym = 0; isym != 3; ++isym)
    {
        const auto expected =
            apply_space_group_rotation_to_kpoint(ctx.rspace_operations[isym], kpoint);
        const auto actual =
            multiply_row_vector(kpoint, ctx.operation_pool[isym].reciprocal_rotation);
        assert(fequal(actual.x, expected.x));
        assert(fequal(actual.y, expected.y));
        assert(fequal(actual.z, expected.z));
        assert(ctx.operation_pool[isym].atom_map.empty());
        assert(ctx.operation_pool[isym].return_lattice.empty());
    }

    // The generated metadata passes its own validation and the rebuild is idempotent.
    validate_symmetry_spin_operations(ctx.spin_operations, ctx.operation_pool.size());
    ctx.ensure_operation_metadata();
    assert(ctx.operation_pool.size() == 3);
    assert(ctx.spin_operations.size() == 6);
}

void test_kstar_member_action_ids_resolve()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ctx;
    set_mgo_primitive_structure(ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    // Bypass set_rspace_operations on purpose: the operation metadata must be
    // rebuilt lazily at the start of the k-star generation.
    add_mgo_fractional_symmetry_operations(ctx);
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);

    assert(ctx.rspace_operations.size() == 48);
    assert(ctx.operation_pool.size() == 48);
    assert(ctx.spin_operations.size() == 96);
    assert(ctx.kstars.size() == 4);

    // Action keys are unique and each action references only its canonical operation.
    std::set<std::size_t> canonical_ids;
    for (const auto& action : ctx.kspace_actions)
    {
        assert(action.canonical_operation_id < ctx.spin_operations.size());
        assert(action.equivalent_operation_ids.size() == 1);
        assert(action.equivalent_operation_ids[0] == action.canonical_operation_id);
        canonical_ids.insert(action.canonical_operation_id);
    }
    assert(canonical_ids.size() == ctx.kspace_actions.size());

    // Every star member resolves to the spin operation matching its
    // (spatial_isym, time_reversal) key.
    for (const auto& star : ctx.kstars)
    {
        for (const auto& member : star.members)
        {
            const auto& action = ctx.kspace_actions.at(member.action_id);
            const auto& spin_operation =
                ctx.spin_operations.at(action.canonical_operation_id);
            assert(static_cast<int>(spin_operation.spatial_id) == member.spatial_isym);
            assert(spin_operation.antiunitary == member.time_reversal);
        }
    }

    // Same invariant for the full-kpoint member table, using a fixture where it
    // is populated; each entry must copy the action of its star member.
    PeriodicBoundaryData pbc_line;
    pbc_line.set_latvec({2.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 6.0, 0.0, 0.0,
    };
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 6.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, 0.0}},
    };
    pbc_line.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);

    SymmetryContext ctx_line;
    ctx_line.set_crystal_structure(pbc_line.latvec,
                                   pbc_line.G,
                                   {{0, 0}},
                                   {{0, {0.0, 0.0, 0.0}}});
    ctx_line.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx_line.set_available();
    ctx_line.build_periodic_mappings(pbc_line, pbc_line.Rlist);

    assert(ctx_line.full_kpoint_members.size() == 3);
    assert(!ctx_line.kspace_actions.empty());
    for (const auto& entry : ctx_line.full_kpoint_members)
    {
        const auto& action = ctx_line.kspace_actions.at(entry.action_id);
        const auto& spin_operation =
            ctx_line.spin_operations.at(action.canonical_operation_id);
        assert(static_cast<int>(spin_operation.spatial_id) == entry.spatial_isym);
        assert(spin_operation.antiunitary == entry.time_reversal);
        const auto& member =
            ctx_line.kstars.at(static_cast<std::size_t>(entry.star_list_index))
                .members.at(static_cast<std::size_t>(entry.member_index));
        assert(entry.action_id == member.action_id);
    }
}

void test_spin_operation_validation_rejects_bad_input()
{
    SymmetrySpinOperation bad_spatial_id;
    bad_spatial_id.spatial_id = 3;
    bool caught = false;
    try
    {
        validate_symmetry_spin_operations({bad_spatial_id}, 1);
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    if (!caught)
    {
        throw std::runtime_error(
            "validate_symmetry_spin_operations accepted an out-of-range spatial_id");
    }

    SymmetrySpinOperation non_unitary;
    non_unitary.spin_u = {2.0, 0.0, 0.0, 1.0};
    caught = false;
    try
    {
        validate_symmetry_spin_operations({non_unitary}, 1);
    }
    catch (const std::invalid_argument&)
    {
        caught = true;
    }
    if (!caught)
    {
        throw std::runtime_error(
            "validate_symmetry_spin_operations accepted a non-unitary spin_u");
    }
}

void test_spin_u_equal_pm_invariance()
{
    const double half_angle = 0.35;
    const std::complex<double> phase(std::cos(half_angle), std::sin(half_angle));
    const std::array<std::complex<double>, 4> u_z{std::conj(phase), 0.0, 0.0, phase};
    std::array<std::complex<double>, 4> minus_u_z;
    for (std::size_t i = 0; i != 4; ++i)
    {
        minus_u_z[i] = -u_z[i];
    }
    // Rotation about a different axis by the same angle.
    const std::array<std::complex<double>, 4> u_x{
        std::cos(half_angle), std::complex<double>(0.0, -std::sin(half_angle)),
        std::complex<double>(0.0, -std::sin(half_angle)), std::cos(half_angle)};

    assert(symmetry_spin_u_equal(u_z, u_z));
    assert(symmetry_spin_u_equal(u_z, minus_u_z));
    assert(!symmetry_spin_u_equal(u_z, u_x));
}

PeriodicBoundaryData make_cubic_mesh_pbc(const Vector3_Order<int>& mesh)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac(mesh))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(mesh.x, mesh.y, mesh.z, kvecs);
    return pbc;
}

SymmetryContext make_single_atom_cubic_context(const PeriodicBoundaryData& pbc,
                                               const std::vector<SymmetryOperation>& ops)
{
    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec, pbc.G, {{0, 0}}, {{0, {0.0, 0.0, 0.0}}});
    ctx.set_rspace_operations(ops);
    return ctx;
}

void build_ctx_periodic_mappings(SymmetryContext& ctx, const PeriodicBoundaryData& pbc)
{
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
}

void test_explicit_unitary_only_list_widens_kstars()
{
    // 2D square lattice, single atom, 4x4x1 grid, spatial group {E, mirror_x}.
    // Orbit counting on (kx, ky) in {0, 1/4, 1/2, 3/4}^2 (verified with an
    // independent enumerator): the default grey table {E, mx, Theta, Theta.mx}
    // acts as (kx, ky) -> (+-kx, +-ky), giving 9 stars (4 singletons, 4 pairs,
    // 1 quartet); the explicit unitary-only table {E, mx} acts as
    // (kx, ky) -> (+-kx, ky), giving 12 stars (8 singletons, 4 pairs).
    auto pbc = make_cubic_mesh_pbc({4, 4, 1});
    const std::vector<SymmetryOperation> ops{
        make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}),
        make_row_symmetry_operation({-1, 0, 0, 0, 1, 0, 0, 0, 1}),
    };

    auto ctx_grey = make_single_atom_cubic_context(pbc, ops);
    build_ctx_periodic_mappings(ctx_grey, pbc);
    assert(!ctx_grey.has_explicit_spin_operations);
    assert(ctx_grey.kstars.size() == 9);
    assert(ctx_grey.count_kstar_members() == 16);
    bool grey_has_tr_member = false;
    for (const auto& star : ctx_grey.kstars)
    {
        for (const auto& member : star.members)
        {
            grey_has_tr_member = grey_has_tr_member || member.time_reversal;
        }
    }
    assert(grey_has_tr_member);

    auto ctx_unitary = make_single_atom_cubic_context(pbc, ops);
    std::vector<SymmetrySpinOperation> unitary_table(2);
    unitary_table[0].spatial_id = 0;
    unitary_table[1].spatial_id = 1;
    ctx_unitary.set_symmetry_spin_operations(unitary_table, false);
    build_ctx_periodic_mappings(ctx_unitary, pbc);
    assert(ctx_unitary.has_explicit_spin_operations);
    assert(ctx_unitary.spin_operations.size() == 2);
    assert(ctx_unitary.kstars.size() == 12);
    assert(ctx_unitary.count_kstar_members() == 16);
    for (const auto& star : ctx_unitary.kstars)
    {
        for (const auto& member : star.members)
        {
            assert(!member.time_reversal);
        }
    }
}

void test_antiunitary_translation_k_action_ignores_translation()
{
    // AFM-type anti-translation: spatial {E|(1/2,0,0)} enters the k-star table
    // once as unitary (k -> k, the translation drops out of the k mapping) and
    // once as antiunitary Theta{E|t} (k -> -k). On the 4x1x1 grid the stars are
    // {0}, {1/2}, {1/4, 3/4}.
    auto pbc = make_cubic_mesh_pbc({4, 1, 1});
    std::vector<SymmetryOperation> ops;
    ops.push_back(make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}));
    auto anti_translation = ops.front();
    anti_translation.translation = {0.5, 0.0, 0.0};
    ops.push_back(anti_translation);

    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec, pbc.G,
                              {{0, 0}, {1, 0}},
                              {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}});
    ctx.set_rspace_operations(ops);
    std::vector<SymmetrySpinOperation> spin_table(2);
    spin_table[0].spatial_id = 1; // {E|t}, unitary: k -> k
    spin_table[1].spatial_id = 1; // Theta{E|t}: k -> -k
    spin_table[1].antiunitary = true;
    ctx.set_symmetry_spin_operations(spin_table, false);
    build_ctx_periodic_mappings(ctx, pbc);

    assert(ctx.kstars.size() == 3);
    const SymmetryKStar* two_member_star = nullptr;
    for (const auto& star : ctx.kstars)
    {
        if (star.members.size() == 2)
        {
            two_member_star = &star;
        }
    }
    assert(two_member_star != nullptr);
    const SymmetryKStarMember* unitary_member = nullptr;
    const SymmetryKStarMember* tr_member = nullptr;
    for (const auto& member : two_member_star->members)
    {
        if (member.time_reversal)
        {
            tr_member = &member;
        }
        else
        {
            unitary_member = &member;
        }
    }
    assert(unitary_member != nullptr && tr_member != nullptr);
    // k and -k share one star; the antiunitary member sits at exactly -k and
    // routes through the anti-translation (spatial_isym == 1, eta = true).
    assert(fequal(unitary_member->k_bz.x, 0.25));
    assert(fequal(tr_member->k_bz.x, 0.75));
    assert(same_fractional_kpoint(tr_member->k_bz, unitary_member->k_bz * -1.0, 1e-8));
    assert(unitary_member->spatial_isym == 1);
    assert(tr_member->spatial_isym == 1);
    // Two deduplicated actions with keys (1, false) and (1, true).
    assert(ctx.kspace_actions.size() == 2);
    for (const auto& action : ctx.kspace_actions)
    {
        const auto& spin_op = ctx.spin_operations.at(action.canonical_operation_id);
        assert(spin_op.spatial_id == 1);
    }
}

void test_inversion_member_is_unitary_not_time_reversal()
{
    // Simple cubic {E, I} with the default grey table: inversion (unitary) and
    // time reversal (antiunitary) both map k to -k, but the route must pick the
    // unitary inversion, which precedes Theta in the table.
    // NOTE: the grid must contain k with -k =/= k (mod G); on a 2x2x2 grid
    // every k is self-inverse and the route check would be vacuous, so a 4x1x1
    // slice is used instead.
    auto pbc = make_cubic_mesh_pbc({4, 1, 1});
    const std::vector<SymmetryOperation> ops{
        make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}),
        make_row_symmetry_operation({-1, 0, 0, 0, -1, 0, 0, 0, -1}),
    };
    auto ctx = make_single_atom_cubic_context(pbc, ops);
    build_ctx_periodic_mappings(ctx, pbc);

    assert(ctx.kstars.size() == 3);
    bool found_inversion_member = false;
    for (const auto& star : ctx.kstars)
    {
        for (const auto& member : star.members)
        {
            assert(!member.time_reversal);
            if (member.spatial_isym == 1)
            {
                found_inversion_member = true;
                assert(same_fractional_kpoint(member.k_bz, star.k_ibz * -1.0, 1e-8));
            }
        }
    }
    assert(found_inversion_member);
}

void test_pure_spin_duplicates_dedup_actions()
{
    // Two spin operations sharing the (spatial_id, antiunitary) geometry but
    // differing in SU(2) must collapse into one geometric action and leave the
    // stars (member counts = weights) untouched.
    auto pbc = make_cubic_mesh_pbc({2, 2, 1});
    const std::vector<SymmetryOperation> ops{
        make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}),
    };

    auto ctx_single = make_single_atom_cubic_context(pbc, ops);
    std::vector<SymmetrySpinOperation> single_table(1);
    single_table[0].spatial_id = 0;
    ctx_single.set_symmetry_spin_operations(single_table, false);
    build_ctx_periodic_mappings(ctx_single, pbc);

    auto ctx_dup = make_single_atom_cubic_context(pbc, ops);
    std::vector<SymmetrySpinOperation> dup_table(2);
    dup_table[0].spatial_id = 0;
    dup_table[1].spatial_id = 0;
    // Pure-spin rotation about z by 2*0.35 rad: same geometry as the identity.
    const double half_angle = 0.35;
    dup_table[1].spin_u = {std::complex<double>(std::cos(half_angle), -std::sin(half_angle)),
                           0.0,
                           0.0,
                           std::complex<double>(std::cos(half_angle), std::sin(half_angle))};
    dup_table[1].spin_source = SymmetrySpinActionSource::ExplicitSpinSpace;
    ctx_dup.set_symmetry_spin_operations(dup_table, false);
    build_ctx_periodic_mappings(ctx_dup, pbc);

    // One deduplicated action collecting both spin operations.
    assert(ctx_dup.kspace_actions.size() == 1);
    const auto& action = ctx_dup.kspace_actions.front();
    assert(action.canonical_operation_id == 0);
    assert(action.equivalent_operation_ids.size() == 2);
    assert(action.equivalent_operation_ids[0] == 0);
    assert(action.equivalent_operation_ids[1] == 1);
    assert(ctx_single.kspace_actions.size() == 1);
    assert(ctx_single.kspace_actions.front().equivalent_operation_ids.size() == 1);

    // Stars identical to the single-operation case.
    assert(ctx_dup.kstars.size() == ctx_single.kstars.size());
    assert(ctx_dup.count_kstar_members() == ctx_single.count_kstar_members());
    for (std::size_t istar = 0; istar != ctx_dup.kstars.size(); ++istar)
    {
        const auto& star_dup = ctx_dup.kstars[istar];
        const auto& star_single = ctx_single.kstars[istar];
        assert(same_fractional_kpoint(star_dup.k_ibz, star_single.k_ibz, 1e-8));
        assert(star_dup.members.size() == star_single.members.size());
        for (std::size_t imember = 0; imember != star_dup.members.size(); ++imember)
        {
            const auto& member_dup = star_dup.members[imember];
            const auto& member_single = star_single.members[imember];
            assert(member_dup.spatial_isym == member_single.spatial_isym);
            assert(member_dup.time_reversal == member_single.time_reversal);
            assert(same_fractional_kpoint(member_dup.k_bz, member_single.k_bz, 1e-8));
        }
    }
}

void test_grey_expansion_matches_legacy_stars()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, 0.5, 0.5,
                    0.5, 0.0, 0.5,
                    0.5, 0.5, 0.0});

    std::vector<double> kvecs;
    for (const auto& kfrac : build_uniform_kmesh_frac({3, 3, 3}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(kvec.x * TWO_PI);
        kvecs.push_back(kvec.y * TWO_PI);
        kvecs.push_back(kvec.z * TWO_PI);
    }
    pbc.set_kgrids_kvec(3, 3, 3, kvecs);

    SymmetryContext ops_ctx;
    set_mgo_primitive_structure(ops_ctx,
                                {{0, 0}, {1, 1}},
                                {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
    add_mgo_fractional_symmetry_operations(ops_ctx);
    const auto mgo_operations = ops_ctx.rspace_operations;

    const auto build_mgo_ctx = [&mgo_operations]() {
        SymmetryContext ctx;
        set_mgo_primitive_structure(ctx,
                                    {{0, 0}, {1, 1}},
                                    {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.5, 0.5}}});
        ctx.set_rspace_operations(mgo_operations);
        return ctx;
    };

    auto ctx_legacy = build_mgo_ctx();
    build_ctx_periodic_mappings(ctx_legacy, pbc);

    auto ctx_grey = build_mgo_ctx();
    std::vector<SymmetrySpinOperation> unitary_table(mgo_operations.size());
    for (std::size_t isym = 0; isym != mgo_operations.size(); ++isym)
    {
        unitary_table[isym].spatial_id = isym;
    }
    ctx_grey.set_symmetry_spin_operations(unitary_table, true);
    build_ctx_periodic_mappings(ctx_grey, pbc);

    // The grey expansion of the all-unitary table reproduces the legacy default
    // spin table entry by entry.
    assert(ctx_grey.spin_operations.size() == ctx_legacy.spin_operations.size());
    for (std::size_t iop = 0; iop != ctx_grey.spin_operations.size(); ++iop)
    {
        const auto& expanded = ctx_grey.spin_operations[iop];
        const auto& legacy = ctx_legacy.spin_operations[iop];
        assert(expanded.spatial_id == legacy.spatial_id);
        assert(expanded.antiunitary == legacy.antiunitary);
        assert(expanded.spin_u == legacy.spin_u);
        assert(expanded.spin_source == legacy.spin_source);
    }

    // K-stars and deduplicated actions are identical, member by member.
    assert(ctx_grey.kstars.size() == ctx_legacy.kstars.size());
    for (std::size_t istar = 0; istar != ctx_grey.kstars.size(); ++istar)
    {
        const auto& star_grey = ctx_grey.kstars[istar];
        const auto& star_legacy = ctx_legacy.kstars[istar];
        assert(same_fractional_kpoint(star_grey.k_ibz, star_legacy.k_ibz, 1e-8));
        assert(star_grey.members.size() == star_legacy.members.size());
        for (std::size_t imember = 0; imember != star_grey.members.size(); ++imember)
        {
            const auto& member_grey = star_grey.members[imember];
            const auto& member_legacy = star_legacy.members[imember];
            assert(member_grey.spatial_isym == member_legacy.spatial_isym);
            assert(member_grey.time_reversal == member_legacy.time_reversal);
            assert(member_grey.action_id == member_legacy.action_id);
            assert(same_fractional_kpoint(member_grey.k_bz, member_legacy.k_bz, 1e-8));
        }
    }
    assert(ctx_grey.kspace_actions.size() == ctx_legacy.kspace_actions.size());
    for (std::size_t iaction = 0; iaction != ctx_grey.kspace_actions.size(); ++iaction)
    {
        const auto& action_grey = ctx_grey.kspace_actions[iaction];
        const auto& action_legacy = ctx_legacy.kspace_actions[iaction];
        assert(action_grey.canonical_operation_id == action_legacy.canonical_operation_id);
        assert(action_grey.equivalent_operation_ids == action_legacy.equivalent_operation_ids);
    }
}

void test_explicit_operation_table_builds_offdiagonal_qstar()
{
    // Section 17 resolution 2 verification case: the q view derives from the
    // same explicit operation table as the k-stars. Non-high-symmetry
    // q = (1/4, 1/2, 0) on the square-lattice 4x4x1 grid with the explicit
    // unitary-only table {E, mirror_x}: its q-star is exactly
    // {q, mirror_x q} with mirror_x q = (3/4, 1/2, 0) (mod G).
    auto pbc = make_cubic_mesh_pbc({4, 4, 1});
    const std::vector<SymmetryOperation> ops{
        make_row_symmetry_operation({1, 0, 0, 0, 1, 0, 0, 0, 1}),
        make_row_symmetry_operation({-1, 0, 0, 0, 1, 0, 0, 0, 1}),
    };
    auto ctx = make_single_atom_cubic_context(pbc, ops);
    std::vector<SymmetrySpinOperation> unitary_table(2);
    unitary_table[0].spatial_id = 0;
    unitary_table[1].spatial_id = 1;
    ctx.set_symmetry_spin_operations(unitary_table, false);
    build_ctx_periodic_mappings(ctx, pbc);

    const auto view = build_symmetry_qpoint_view(ctx, pbc, true);
    assert(view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    assert(view.representatives.size() == 12);
    std::size_t n_members = 0;
    for (const auto& q : view.representatives)
    {
        n_members += view.members.at(q).size();
    }
    assert(n_members == 16);

    const Vector3_Order<double> q_target{Vector3_Order<double>{0.25, 0.5, 0.0} * pbc.G};
    const Vector3_Order<double> q_mirror{Vector3_Order<double>{0.75, 0.5, 0.0} * pbc.G};
    bool found = false;
    for (const auto& q_rep : view.representatives)
    {
        const auto& members = view.members.at(q_rep);
        bool has_target = false;
        for (const auto& member : members)
        {
            if ((member - q_target).norm() < 1e-8)
            {
                has_target = true;
            }
        }
        if (!has_target)
        {
            continue;
        }
        found = true;
        assert(members.size() == 2);
        bool has_mirror = false;
        for (const auto& member : members)
        {
            if ((member - q_mirror).norm() < 1e-8)
            {
                has_mirror = true;
            }
        }
        assert(has_mirror);
    }
    assert(found);
}

int main()
{
    test_symmetry_context_saves_fractional_row_operations();
    test_kspace_shell_rotations_use_direct_rotation();
    test_kstar_member_return_lattice_preserves_input_fractional_representative();
    test_kstar_member_atom_mapping_tolerates_text_coordinate_noise();
    test_kstar_member_atom_mapping_tolerates_si_fractional_roundoff();
    test_kspace_rotation_derivative_matches_bloch_phase_difference();
    test_species_basis_layout_keeps_shell_order();
    test_species_layouts_match_basis_dimensions();
    test_atomic_basis_builds_symmetry_species_layouts();
    test_periodic_mappings_store_full_kpoint_members();
    test_periodic_mappings_accept_kq_reduced_coulomb_grid();
    test_qpoint_view_uses_pbc_without_symmetry();
    test_qpoint_view_preserves_time_reversal_mapping_without_crystal_symmetry();
    test_qpoint_view_preserves_time_reversal_q_reduction_with_crystal_symmetry();
    test_qpoint_view_rejects_reduced_input_without_symmetry();
    test_qpoint_view_accepts_reduced_input_with_symmetry();
    test_qpoint_view_keeps_full_input_grid_without_parsed_crystal_symmetry();
    test_single_atom_qpoint_view_reduces_full_input_to_crystal_qstars();
    test_mgo_qpoint_view_reduces_time_reversal_q_list_to_crystal_qstars_for_full_input();
    test_mgo_keeps_all_kspace_operations_for_full_qstars();
    test_kstar_routes_prefer_improper_spatial_operation_over_time_reversal();
    test_dense_kspace_rotation_matrix_orders_atom_swap_blocks();
    test_dense_operator_rotation_matches_atom_blocks_for_asymmetric_swap_phases();
    test_equivalent_kpoint_gauge_respects_bloch_atom_phase_convention();
    test_mgo_dense_kspace_rotation_matches_atom_block_rotation();
    test_mgo_k333_irreducible_sector_matches_single();
    test_rspace_sector_star_member_isym_maps_full_to_ir_r();
    test_rspace_block_restore_uses_stored_operation_rotation_convention();
    test_mgo_k333_irreducible_sector_matches_both();
    test_bn_shrink_irreducible_sector_can_be_generated_from_symmetry();
    test_spin_operations_identity_translation();
    test_kstar_member_action_ids_resolve();
    test_spin_operation_validation_rejects_bad_input();
    test_spin_u_equal_pm_invariance();
    test_explicit_unitary_only_list_widens_kstars();
    test_antiunitary_translation_k_action_ignores_translation();
    test_inversion_member_is_unitary_not_time_reversal();
    test_pure_spin_duplicates_dedup_actions();
    test_grey_expansion_matches_legacy_stars();
    test_explicit_operation_table_builds_offdiagonal_qstar();
}
