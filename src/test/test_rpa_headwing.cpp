#include <array>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <valarray>

#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/qpoint_view.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "../utils/constants.h"

using librpa_int::ArrayDesc;
using librpa_int::AtomicBasis;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::BlacsCtxtHandler;
using librpa_int::C_ONE;
using librpa_int::ComplexMatrix;
using librpa_int::diele_func;
using librpa_int::init_local_mat;
using librpa_int::KPointBlacsParallelContext;
using librpa_int::KPointBlacsProcessShape;
using librpa_int::MAJOR;
using librpa_int::Matrix3;
using librpa_int::Matz;
using librpa_int::matrix_m;
using librpa_int::MeanField;
using librpa_int::PeriodicBoundaryData;
using librpa_int::SpeciesBasisLayout;
using librpa_int::SymmetryContext;
using librpa_int::SymmetryKAtomRotation;
using librpa_int::SymmetryKStarMember;
using librpa_int::SymmetryOperation;
using librpa_int::SymmetryQPointRestoreMode;
using librpa_int::TFGrids;
using librpa_int::Vector3_Order;
using librpa_int::atom_mapping;
using librpa_int::atom_t;
using librpa_int::build_symmetry_qpoint_view;

namespace
{

void test_rspace_symmetry_requires_complete_band_space()
{
    MeanField truncated(1, 1, 2, 3);
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, 3));

    MeanField complete(1, 1, 3, 3);
    assert(librpa_int::rspace_symmetry_has_complete_band_space(complete, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(complete, 2));
}

void assert_complex_close(const std::complex<double> &actual, const std::complex<double> &expected,
                          const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        assert(false);
    }
}

void require_double_close(const double actual, const double expected, const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

void test_kpoint_coordinate_mapping_selects_active_klist_from_full_source()
{
    const std::vector<Vector3_Order<double>> pyatb_full_kpoints{
        {0.0, 0.0, 0.0}, {0.125, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.375, 0.0, 0.0},
        {0.0, 0.125, 0.0}, {0.125, 0.125, 0.0}, {0.875, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> active_kpoints{
        {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.125, 0.125, 0.0}};

    const auto mapping = librpa_int::map_kpoints_by_coordinates(active_kpoints, pyatb_full_kpoints);

    assert((mapping == std::vector<int>{0, 2, 5}));

    const std::vector<Vector3_Order<double>> wrapped_active_kpoints{{-0.125, 0.0, 0.0}};
    const auto wrapped_mapping =
        librpa_int::map_kpoints_by_coordinates(wrapped_active_kpoints, pyatb_full_kpoints);
    assert((wrapped_mapping == std::vector<int>{6}));
}

void test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge()
{
    SymmetryContext ctx;
    librpa_int::SymmetryKStar gamma;
    gamma.k_ibz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_first;
    gamma_first.k_bz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_second;
    gamma_second.k_bz = {0.5, 0.0, 0.0};
    gamma.members = {gamma_first, gamma_second};
    librpa_int::SymmetryKStar quarter;
    quarter.k_ibz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_first;
    quarter_first.k_bz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_second;
    quarter_second.k_bz = {-0.25, 0.0, 0.0};
    quarter.members = {quarter_first, quarter_second};
    ctx.kstars = {gamma, quarter};

    const std::vector<Vector3_Order<double>> ibz_kpoints{{0.25, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> full_bz_kpoints{
        {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}};

    const auto mapping = librpa_int::map_symmetry_kstar_members_to_source_kpoints(
        ctx, ibz_kpoints, full_bz_kpoints);

    assert((mapping == std::vector<std::vector<int>>{{3, 1}, {2, 0}}));
}

void test_replace_rpa_response_headwing_replaces_only_singular_channels(
    const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>(0.01 * (i + 1) + 0.02 * (j + 1), 0.001 * (i - j));
        }
    }

    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{2.0, 0.0}, std::complex<double>{0.2, 0.1},
             std::complex<double>{0.3, -0.1}},
            {std::complex<double>{0.2, -0.1}, std::complex<double>{2.2, 0.0},
             std::complex<double>{0.4, 0.2}},
            {std::complex<double>{0.3, 0.1}, std::complex<double>{0.4, -0.2},
             std::complex<double>{2.4, 0.0}}},
        MAJOR::COL);
    const matrix_m<std::complex<double>> wing(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.11, 0.01}, std::complex<double>{0.12, 0.02},
             std::complex<double>{0.13, 0.03}},
            {std::complex<double>{0.21, 0.04}, std::complex<double>{0.22, 0.05},
             std::complex<double>{0.23, 0.06}},
            {std::complex<double>{0.31, 0.07}, std::complex<double>{0.32, 0.08},
             std::complex<double>{0.33, 0.09}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_headwing(response, head, wing, desc);

    const int head_row = desc.indx_g2l_r(0);
    const int head_col = desc.indx_g2l_c(0);
    if (head_row >= 0 && head_col >= 0)
    {
        assert_complex_close(response(head_row, head_col), std::complex<double>{2.2, 0.0}, 1e-12);
    }

    for (int lambda = 1; lambda != 4; ++lambda)
    {
        std::complex<double> expected_wing = 0.0;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            expected_wing += wing(lambda - 1, alpha);
        }
        expected_wing /= 3.0;

        const int row_body = desc.indx_g2l_r(lambda);
        const int col_head = desc.indx_g2l_c(0);
        if (row_body >= 0 && col_head >= 0)
        {
            assert_complex_close(response(row_body, col_head), expected_wing, 1e-12);
        }

        const int row_head = desc.indx_g2l_r(0);
        const int col_body = desc.indx_g2l_c(lambda);
        if (row_head >= 0 && col_body >= 0)
        {
            assert_complex_close(response(row_head, col_body), std::conj(expected_wing), 1e-12);
        }
    }

    const int body_row = desc.indx_g2l_r(2);
    const int body_col = desc.indx_g2l_c(3);
    if (body_row >= 0 && body_col >= 0)
    {
        assert_complex_close(response(body_row, body_col), std::complex<double>{0.11, -0.001},
                             1e-12);
    }
}

void test_replace_rpa_response_head_only_keeps_numeric_wings(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    matrix_m<std::complex<double>> original(4, 4, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        for (int j = 0; j != 4; ++j)
        {
            const auto value =
                std::complex<double>(0.1 * (i + 1) + 0.01 * (j + 1), 0.001 * (i - j));
            original(i, j) = value;
            if (ilo < 0) continue;
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) = value;
        }
    }

    const matrix_m<std::complex<double>> chi0v_head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.21, 0.0}, std::complex<double>{0.01, 0.02},
             std::complex<double>{-0.03, 0.04}},
            {std::complex<double>{0.05, -0.01}, std::complex<double>{0.24, 0.0},
             std::complex<double>{0.07, 0.03}},
            {std::complex<double>{-0.02, -0.04}, std::complex<double>{0.08, -0.03},
             std::complex<double>{0.27, 0.0}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_head_only(response, chi0v_head, desc);

    const auto expected_head = (chi0v_head(0, 0) + chi0v_head(1, 1) + chi0v_head(2, 2)) / 3.0;
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            const auto expected = (i == 0 && j == 0) ? expected_head : original(i, j);
            assert_complex_close(response(ilo, jlo), expected, 1e-12);
        }
    }
}

void test_head_only_trace_logdet_can_use_reduced_response(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(3, 3, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>{0.02 * (i + 1) + 0.01 * (j + 1), 0.002 * (i - j)};
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);

    const matrix_m<std::complex<double>> dense_response(
        std::vector<std::vector<std::complex<double>>>{
            {{0.03, 0.0}, {0.04, -0.002}, {0.05, -0.004}},
            {{0.05, 0.002}, {0.06, 0.0}, {0.07, -0.002}},
            {{0.07, 0.004}, {0.08, 0.002}, {0.09, 0.0}}},
        MAJOR::COL);
    std::complex<double> trace = 0.0;
    for (int i = 0; i != 3; ++i)
    {
        trace += dense_response(i, i);
    }
    const auto a = 1.0 - dense_response(0, 0);
    const auto b = -dense_response(0, 1);
    const auto c = -dense_response(0, 2);
    const auto d = -dense_response(1, 0);
    const auto e = 1.0 - dense_response(1, 1);
    const auto f = -dense_response(1, 2);
    const auto g = -dense_response(2, 0);
    const auto h = -dense_response(2, 1);
    const auto i = 1.0 - dense_response(2, 2);
    const auto det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const auto expected = trace + std::log(det);

    assert_complex_close(actual, expected, 1e-12);

    ArrayDesc desc_full(blacs_h);
    desc_full.init_square_blk(5, 5, 0, 0);
    auto full_response = init_local_mat<std::complex<double>>(desc_full, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc_full.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc_full.indx_g2l_c(j);
            if (jlo < 0) continue;
            full_response(ilo, jlo) = dense_response(i, j);
        }
    }
    const auto full_actual =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(full_response, desc_full);

    assert_complex_close(full_actual, actual, 1e-12);
}

void test_rpa_trace_log_average_uses_directional_head_and_wing()
{
    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.2, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.3, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.4, 0.0}}},
        MAJOR::COL);
    const std::complex<double> body{0.1, 0.0};
    const std::array<std::complex<double>, 3> wing{std::complex<double>{0.05, 0.0},
                                                   std::complex<double>{0.02, 0.0},
                                                   std::complex<double>{0.01, 0.0}};
    const std::complex<double> body_inv = 1.0 / (1.0 - body);
    const matrix_m<std::complex<double>> schur_l(
        std::vector<std::vector<std::complex<double>>>{
            {1.0 - head(0, 0) - std::conj(wing[0]) * body_inv * wing[0],
             -std::conj(wing[0]) * body_inv * wing[1], -std::conj(wing[0]) * body_inv * wing[2]},
            {-std::conj(wing[1]) * body_inv * wing[0],
             1.0 - head(1, 1) - std::conj(wing[1]) * body_inv * wing[1],
             -std::conj(wing[1]) * body_inv * wing[2]},
            {-std::conj(wing[2]) * body_inv * wing[0], -std::conj(wing[2]) * body_inv * wing[1],
             1.0 - head(2, 2) - std::conj(wing[2]) * body_inv * wing[2]}},
        MAJOR::COL);
    const std::vector<double> qx{1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0};
    const std::vector<double> qz{0.0, 0.0};
    const std::vector<double> weights{0.2, 0.6};
    const std::complex<double> trace_body = body;
    const std::complex<double> logdet_body = std::log(1.0 - body);

    const auto actual = librpa_int::compute_rpa_chi0v_headwing_trace_log_average(
        head, schur_l, trace_body, logdet_body, qx, qy, qz, weights);
    const auto direct_trace_log = [&](const double nx, const double ny, const double nz)
    {
        const auto directional_head = nx * (nx * head(0, 0) + ny * head(0, 1) + nz * head(0, 2)) +
                                      ny * (nx * head(1, 0) + ny * head(1, 1) + nz * head(1, 2)) +
                                      nz * (nx * head(2, 0) + ny * head(2, 1) + nz * head(2, 2));
        const auto directional_wing = nx * wing[0] + ny * wing[1] + nz * wing[2];
        const auto direct_det = (1.0 - directional_head) * (1.0 - body) -
                                std::conj(directional_wing) * directional_wing;
        return directional_head + body + std::log(direct_det);
    };
    const auto expected = weights[0] * direct_trace_log(qx[0], qy[0], qz[0]) +
                          weights[1] * direct_trace_log(qx[1], qy[1], qz[1]);

    assert_complex_close(actual, expected, 1e-12);
}

void test_rpa_headwing_regular_body_start_channel()
{
    librpa_int::RpaHeadwingSettings settings;

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 1;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 4;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 4);
}

void test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice()
{
    librpa_int::PeriodicBoundaryData pbc;
    pbc.latvec = librpa_int::Matrix3(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0);
    pbc.G = librpa_int::Matrix3(0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.2);

    const double vol_3d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, false);
    require_double_close(vol_3d, std::abs(pbc.G.Det()), 1e-14);

    const double vol_2d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, true);
    const double expected_2d = std::abs(pbc.G.e11 * pbc.G.e22 - pbc.G.e12 * pbc.G.e21);
    require_double_close(vol_2d, expected_2d, 1e-14);

    pbc.set_period(4, 4, 4);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, false),
                         vol_3d / 64.0, 1e-14);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, true),
                         vol_2d / 64.0, 1e-14);
}

void test_strict_2d_headwing_prefactors_use_inplane_area()
{
    constexpr double area = 15.0;
    require_double_close(librpa_int::strict_2d_head_prefactor(area), librpa_int::TWO_PI / area,
                         1e-14);
    require_double_close(librpa_int::strict_2d_wing_prefactor(area),
                         2.0 * std::sqrt(librpa_int::TWO_PI / area), 1e-14);
}

void test_strict_2d_auxiliary_normalization_is_computed_from_basis_metadata()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({19.390653825130212, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 30.0});

    constexpr double multipole_norm_squared = 2205.0673846924301;
    const auto normalization = librpa_int::strict_2d_coulomb_head_normalization(
        pbc, multipole_norm_squared);

    require_double_close(normalization.inplane_area_bohr2, 19.390653825130212, 1e-13);
    require_double_close(normalization.auxiliary_head_coefficient,
                         8978.8175111265446, 1e-10);
    require_double_close(normalization.pw_to_auxiliary_scale,
                         37.802423070695596, 1e-12);

    bool rejected = false;
    try
    {
        (void)librpa_int::strict_2d_coulomb_head_normalization(pbc, 0.0);
    }
    catch (const std::logic_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_gamma_cell_uses_physical_reciprocal_measure()
{
    constexpr double internal_q = 0.2;
    constexpr double internal_area = 0.07;
    require_double_close(librpa_int::strict_2d_physical_q(internal_q),
                         librpa_int::TWO_PI * internal_q, 1e-14);
    require_double_close(librpa_int::strict_2d_physical_gamma_cell_area(internal_area),
                         librpa_int::TWO_PI * librpa_int::TWO_PI * internal_area, 1e-14);
}

void test_strict_2d_radial_integrals_match_analytic_values()
{
    const std::complex<double> a{2.0, 0.0};
    constexpr double qmax = 0.5;
    const auto expected_i0 = (1.0 - std::log(2.0)) / 4.0;
    const auto expected_i1 = (0.5 - 1.0 + std::log(2.0)) / 8.0;

    assert_complex_close(librpa_int::strict_2d_radial_i0(a, qmax), expected_i0, 1e-14);
    assert_complex_close(librpa_int::strict_2d_radial_i1(a, qmax), expected_i1, 1e-14);
}

void test_strict_2d_radial_integrals_are_stable_at_zero_and_small_a()
{
    constexpr double qmax = 0.3;
    assert_complex_close(librpa_int::strict_2d_radial_i0(0.0, qmax), qmax * qmax / 2.0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(0.0, qmax), qmax * qmax * qmax / 3.0,
                         1e-15);

    const std::complex<double> small_a{1.0e-10, -2.0e-10};
    const auto expected_i0 = qmax * qmax / 2.0 - small_a * std::pow(qmax, 3) / 3.0;
    const auto expected_i1 = std::pow(qmax, 3) / 3.0 - small_a * std::pow(qmax, 4) / 4.0;
    assert_complex_close(librpa_int::strict_2d_radial_i0(small_a, qmax), expected_i0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(small_a, qmax), expected_i1, 1e-15);
}

void test_strict_2d_inverse_head_average_has_linear_q_screening()
{
    const std::complex<double> a{1.7, 0.0};
    constexpr double qmax = 0.2;
    const auto inverse_head_average =
        2.0 * librpa_int::strict_2d_radial_i0(a, qmax) / (qmax * qmax);
    const auto old_2d_formula = 1.0 / a;

    assert(std::abs(inverse_head_average - 1.0) < 0.2);
    assert(std::abs(inverse_head_average - old_2d_formula) > 0.1);
}

void test_strict_2d_finite_q_reference_matches_head_and_schur_limits()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    head(0, 0) = 1.8;
    head(0, 1) = 0.12;
    head(1, 0) = 0.12;
    head(1, 1) = 1.4;
    head(2, 2) = 1.0;
    lind(0, 0) = 1.65;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.30;
    lind(2, 2) = 1.0;

    const auto reference = librpa_int::strict_2d_finite_q_reference(head, lind, 3.0, 4.0);
    const double qx = 3.0 / 5.0;
    const double qy = 4.0 / 5.0;
    const auto expected_eps_coefficient =
        qx * (qx * head(0, 0) + qy * head(0, 1)) + qy * (qx * head(1, 0) + qy * head(1, 1)) - 1.0;
    const auto expected_a =
        qx * (qx * lind(0, 0) + qy * lind(0, 1)) + qy * (qx * lind(1, 0) + qy * lind(1, 1)) - 1.0;

    assert_complex_close(reference.epsilon_minus_identity_over_q, expected_eps_coefficient, 1e-14);
    assert_complex_close(reference.schur_a, expected_a, 1e-14);
    assert_complex_close(reference.wc_head_limit, -librpa_int::TWO_PI * expected_a, 1e-14);
}

void test_strict_2d_schur_coefficient_removes_identity()
{
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    lind(0, 0) = 1.4;
    lind(1, 1) = 1.9;
    lind(2, 2) = 1.0;

    constexpr double qx = 0.6;
    constexpr double qy = 0.8;
    const auto expected = 0.4 * qx * qx + 0.9 * qy * qy;
    assert_complex_close(librpa_int::strict_2d_schur_coefficient(lind, qx, qy), expected, 1e-14);
}

void test_strict_2d_screening_denominator_must_stay_on_physical_branch()
{
    librpa_int::validate_strict_2d_screening_denominator({0.7, 1.0e-12}, 0.4);

    bool rejected_zero = false;
    try
    {
        librpa_int::validate_strict_2d_screening_denominator(-2.5, 0.4);
    }
    catch (const std::logic_error &)
    {
        rejected_zero = true;
    }
    assert(rejected_zero);

    bool rejected_negative = false;
    try
    {
        librpa_int::validate_strict_2d_screening_denominator(-3.0, 0.4);
    }
    catch (const std::logic_error &)
    {
        rejected_negative = true;
    }
    assert(rejected_negative);
}

void test_strict_2d_gw_uses_full_coulomb_at_all_q()
{
    librpa_int::validate_strict_2d_gw_coulomb_choices(false, false, true);
    librpa_int::validate_strict_2d_gw_coulomb_choices(true, true, true);

    bool rejected_cut_coulomb_finite_q = false;
    try
    {
        librpa_int::validate_strict_2d_gw_coulomb_choices(true, true, false);
    }
    catch (const std::logic_error &)
    {
        rejected_cut_coulomb_finite_q = true;
    }
    if (!rejected_cut_coulomb_finite_q)
    {
        std::cerr << "strict 2D GW accepted non-Ewald finite-q Wc legs" << std::endl;
        std::abort();
    }

    bool rejected_cut_coulomb_basis = false;
    try
    {
        librpa_int::validate_strict_2d_gw_coulomb_choices(true, false, true);
    }
    catch (const std::logic_error &)
    {
        rejected_cut_coulomb_basis = true;
    }
    if (!rejected_cut_coulomb_basis)
    {
        std::cerr << "strict 2D GW accepted a non-Ewald dielectric basis" << std::endl;
        std::abort();
    }
}

void test_strict_2d_gw_routes_gamma_through_complete_wc_average()
{
    const auto require_route = [](const bool condition, const char *message) {
        if (!condition)
        {
            std::cerr << message << std::endl;
            std::abort();
        }
    };
    require_route(librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, true),
                  "strict 2D Gamma must use the complete-Wc route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(false, 3, true, true, true),
                  "disabled head/wing replacement must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 2, true, true, true),
                  "non-full head/wing dielectric mode must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, false, true, true),
                  "3D dielectric calculations must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, false, true),
                  "finite q must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, false),
                  "missing head/wing data must keep the standard route");
}

void test_strict_2d_block_metrics_separate_head_wings_and_body()
{
    librpa_int::Strict2dBlockMetricSums sums;
    librpa_int::accumulate_strict_2d_block_metric(sums, 0, 0, {2.0, -1.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 0, 1, {3.0, 4.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 2, 0, {0.0, 6.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 1, 1, {5.0, 12.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 2, 2, {8.0, 15.0});

    const auto metrics = librpa_int::finalize_strict_2d_block_metrics(sums);
    require_double_close(metrics.head.real(), 2.0, 1e-15);
    require_double_close(metrics.head.imag(), -1.0, 1e-15);
    require_double_close(metrics.head_body_frobenius, 5.0, 1e-15);
    require_double_close(metrics.body_head_frobenius, 6.0, 1e-15);
    require_double_close(metrics.body_body_frobenius, std::sqrt(13.0 * 13.0 + 17.0 * 17.0), 1e-15);
}

void test_strict_2d_alpha_reference_averages_bare_coulomb()
{
    constexpr double alpha = 0.25;
    constexpr double radius = 0.4;
    const double gamma_area = librpa_int::PI * radius * radius;
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax(4, radius);
    matrix_m<std::complex<double>> regular_body_sqrt(1, 1, MAJOR::COL);
    regular_body_sqrt(0, 0) = 2.0;

    const auto alpha_wc = librpa_int::strict_2d_alpha_wc_average_coulomb_basis(
        alpha, regular_body_sqrt, weights, qmax, gamma_area);
    require_double_close(alpha_wc(0, 0).real(), (alpha - 1.0) * 4.0 * librpa_int::PI / radius,
                         1e-13);
    require_double_close(alpha_wc(0, 0).imag(), 0.0, 1e-15);
    require_double_close(std::abs(alpha_wc(0, 1)), 0.0, 1e-15);
    require_double_close(std::abs(alpha_wc(1, 0)), 0.0, 1e-15);
    require_double_close(alpha_wc(1, 1).real(), (alpha - 1.0) * 4.0, 1e-13);
}

void test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis()
{
    constexpr double scale = 5.0;

    matrix_m<std::complex<double>> pw_wc(3, 3, MAJOR::COL);
    pw_wc(0, 0) = {2.0, -0.5};
    pw_wc(0, 1) = {3.0, 4.0};
    pw_wc(0, 2) = {-1.0, 0.25};
    pw_wc(1, 0) = std::conj(pw_wc(0, 1));
    pw_wc(2, 0) = std::conj(pw_wc(0, 2));
    pw_wc(1, 1) = {5.0, 0.0};
    pw_wc(1, 2) = {0.75, -0.2};
    pw_wc(2, 1) = std::conj(pw_wc(1, 2));
    pw_wc(2, 2) = {7.0, 0.0};

    const auto auxiliary_wc =
        librpa_int::strict_2d_transform_pw_wc_to_auxiliary_basis(pw_wc, scale);
    assert_complex_close(auxiliary_wc(0, 0), scale * scale * pw_wc(0, 0), 1e-13);
    for (int i = 1; i != 3; ++i)
    {
        assert_complex_close(auxiliary_wc(0, i), scale * pw_wc(0, i), 1e-13);
        assert_complex_close(auxiliary_wc(i, 0), scale * pw_wc(i, 0), 1e-13);
        for (int j = 1; j != 3; ++j)
            assert_complex_close(auxiliary_wc(i, j), pw_wc(i, j), 1e-13);
    }
}

void test_strict_2d_wc_blocks_match_dense_finite_q_inverse()
{
    const std::complex<double> body{1.6, 0.0};
    const std::complex<double> left_wing{0.25, 0.04};
    const std::complex<double> right_wing = std::conj(left_wing);
    const std::complex<double> head_coefficient{0.9, 0.0};
    const std::complex<double> body_inv = 1.0 / body;
    const std::complex<double> bw_direction = body_inv * left_wing;
    const std::complex<double> wb_direction = right_wing * body_inv;
    const std::complex<double> schur_a = head_coefficient - right_wing * body_inv * left_wing;
    constexpr double q = 0.17;
    constexpr double cut_body_sqrt = 1.3;

    matrix_m<std::complex<double>> body_inv_matrix(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> wb(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv_matrix(0, 0) = body_inv;
    bw(0, 0) = bw_direction;
    wb(0, 0) = wb_direction;
    cut_sqrt(0, 0) = cut_body_sqrt;

    const auto actual =
        librpa_int::strict_2d_wc_blocks_at_q(body_inv_matrix, bw, wb, schur_a, cut_sqrt, q);

    const std::complex<double> eps00 = 1.0 + q * head_coefficient;
    const std::complex<double> eps01 = std::sqrt(q) * right_wing;
    const std::complex<double> eps10 = std::sqrt(q) * left_wing;
    const std::complex<double> determinant = eps00 * body - eps01 * eps10;
    const std::complex<double> inv00 = body / determinant;
    const std::complex<double> inv01 = -eps01 / determinant;
    const std::complex<double> inv10 = -eps10 / determinant;
    const std::complex<double> inv11 = eps00 / determinant;
    const double head_sqrt = std::sqrt(librpa_int::TWO_PI / q);

    assert_complex_close(actual(0, 0), head_sqrt * head_sqrt * (inv00 - 1.0), 1e-13);
    assert_complex_close(actual(0, 1), head_sqrt * (inv01 * cut_body_sqrt), 1e-13);
    assert_complex_close(actual(1, 0), cut_body_sqrt * inv10 * head_sqrt, 1e-13);
    assert_complex_close(actual(1, 1), cut_body_sqrt * cut_body_sqrt * (inv11 - 1.0), 1e-13);
}

void test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(1, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 1, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.72;
    bw_cart(0, 0) = {0.18, 0.03};
    bw_cart(0, 1) = {-0.07, 0.02};
    wb_cart(0, 0) = std::conj(bw_cart(0, 0));
    wb_cart(1, 0) = std::conj(bw_cart(0, 1));
    lind(0, 0) = 1.55;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.25;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.4;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax{0.21, 0.14, 0.21, 0.14};
    double gamma_area = 0.0;
    for (std::size_t i = 0; i != qmax.size(); ++i)
        gamma_area += weights[i] * qmax[i] * qmax[i] / 2.0;

    const auto analytic = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

    matrix_m<std::complex<double>> numeric(2, 2, MAJOR::COL);
    constexpr int radial_points = 200000;
    for (std::size_t idir = 0; idir != qx.size(); ++idir)
    {
        matrix_m<std::complex<double>> bw_direction(1, 1, MAJOR::COL);
        matrix_m<std::complex<double>> wb_direction(1, 1, MAJOR::COL);
        bw_direction(0, 0) = bw_cart(0, 0) * qx[idir] + bw_cart(0, 1) * qy[idir];
        wb_direction(0, 0) = wb_cart(0, 0) * qx[idir] + wb_cart(1, 0) * qy[idir];
        const auto a = librpa_int::strict_2d_schur_coefficient(lind, qx[idir], qy[idir]);
        const double dq = qmax[idir] / radial_points;
        for (int ir = 0; ir != radial_points; ++ir)
        {
            const double q = (ir + 0.5) * dq;
            const auto point = librpa_int::strict_2d_wc_blocks_at_q(body_inv, bw_direction,
                                                                    wb_direction, a, cut_sqrt, q);
            const double measure = weights[idir] * q * dq / gamma_area;
            for (int i = 0; i != 2; ++i)
                for (int j = 0; j != 2; ++j) numeric(i, j) += measure * point(i, j);
        }
    }

    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j) assert_complex_close(analytic(i, j), numeric(i, j), 2e-11);
    assert(std::abs(analytic(0, 1)) < 1e-13);
    assert(std::abs(analytic(1, 0)) < 1e-13);
}

void test_strict_2d_wc_blocks_have_finite_small_q_limits()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> wb(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.75;
    bw(0, 0) = {0.12, 0.03};
    wb(0, 0) = std::conj(bw(0, 0));
    cut_sqrt(0, 0) = 1.25;
    const std::complex<double> a{0.6, 0.0};
    constexpr double q = 1.0e-9;

    const auto wc = librpa_int::strict_2d_wc_blocks_at_q(body_inv, bw, wb, a, cut_sqrt, q);
    assert_complex_close(wc(0, 0), -librpa_int::TWO_PI * a, 3e-9);
    assert_complex_close(wc(1, 0), -std::sqrt(librpa_int::TWO_PI) * cut_sqrt(0, 0) * bw(0, 0),
                         3e-9);
    assert_complex_close(wc(0, 1), -std::sqrt(librpa_int::TWO_PI) * wb(0, 0) * cut_sqrt(0, 0),
                         3e-9);
    assert_complex_close(wc(1, 1), cut_sqrt(0, 0) * (body_inv(0, 0) - 1.0) * cut_sqrt(0, 0), 3e-9);
}

void test_strict_2d_wc_average_is_covariant_under_regular_body_rotation()
{
    matrix_m<std::complex<double>> body_inv(2, 2, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(2, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 2, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(2, 2, MAJOR::COL);
    body_inv(0, 0) = 0.70;
    body_inv(0, 1) = 0.04;
    body_inv(1, 0) = 0.04;
    body_inv(1, 1) = 0.82;
    bw_cart(0, 0) = 0.13;
    bw_cart(0, 1) = -0.05;
    bw_cart(1, 0) = 0.08;
    bw_cart(1, 1) = 0.11;
    wb_cart = bw_cart.get_transpose(true);
    lind(0, 0) = 1.4;
    lind(0, 1) = 0.06;
    lind(1, 0) = 0.06;
    lind(1, 1) = 1.7;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.15;
    cut_sqrt(0, 1) = 0.03;
    cut_sqrt(1, 0) = 0.03;
    cut_sqrt(1, 1) = 0.95;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax(4, 0.18);
    const double gamma_area = librpa_int::TWO_PI * 0.18 * 0.18 / 2.0;
    const auto wc = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

    constexpr double angle = 0.37;
    matrix_m<std::complex<double>> rotation(2, 2, MAJOR::COL);
    rotation(0, 0) = std::cos(angle);
    rotation(0, 1) = -std::sin(angle);
    rotation(1, 0) = std::sin(angle);
    rotation(1, 1) = std::cos(angle);
    const auto rotation_h = rotation.get_transpose(true);
    const auto body_rotated = rotation_h * body_inv * rotation;
    const auto bw_rotated = rotation_h * bw_cart;
    const auto wb_rotated = wb_cart * rotation;
    const auto cut_rotated = rotation_h * cut_sqrt * rotation;
    const auto wc_rotated = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_rotated, bw_rotated, wb_rotated, lind, cut_rotated, qx, qy, weights, qmax, gamma_area);

    matrix_m<std::complex<double>> full_rotation(3, 3, MAJOR::COL);
    full_rotation(0, 0) = 1.0;
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j) full_rotation(i + 1, j + 1) = rotation(i, j);
    const auto expected = full_rotation.get_transpose(true) * wc * full_rotation;
    for (int i = 0; i != 3; ++i)
        for (int j = 0; j != 3; ++j) assert_complex_close(wc_rotated(i, j), expected(i, j), 2e-13);
}

void test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(1, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 1, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.76;
    bw_cart(0, 0) = {0.14, 0.02};
    bw_cart(0, 1) = {-0.05, 0.01};
    wb_cart(0, 0) = std::conj(bw_cart(0, 0));
    wb_cart(1, 0) = std::conj(bw_cart(0, 1));
    lind(0, 0) = 1.45;
    lind(0, 1) = 0.04;
    lind(1, 0) = 0.04;
    lind(1, 1) = 1.30;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.2;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::array<int, 4> meshes{12, 14, 16, 20};
    double previous_weighted_norm = std::numeric_limits<double>::infinity();

    for (const int mesh : meshes)
    {
        const double gamma_area = 1.0 / static_cast<double>(mesh * mesh);
        const double radial_extent = std::sqrt(2.0 * gamma_area / librpa_int::TWO_PI);
        const std::vector<double> qmax(4, radial_extent);
        const auto average = librpa_int::strict_2d_average_wc_coulomb_basis(
            body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

        double average_norm_squared = 0.0;
        for (int i = 0; i != average.nr(); ++i)
            for (int j = 0; j != average.nc(); ++j)
                average_norm_squared += std::norm(average(i, j));
        const double average_norm = std::sqrt(average_norm_squared);
        assert(std::isfinite(average_norm));
        assert(average_norm < 10.0);

        const double weighted_norm = average_norm / static_cast<double>(mesh * mesh);
        assert(weighted_norm < previous_weighted_norm);
        previous_weighted_norm = weighted_norm;
    }
}

void test_rpa_chi0v_wing_desc_uses_global_rows(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(10, 10, 0, 0);

    ArrayDesc desc_full_wing(blacs_h);
    desc_full_wing.init(11, 3, desc_body.mb(), 1, 0, 0);

    const auto desc_wing = librpa_int::make_rpa_chi0v_wing_desc(
        desc_body, 1, desc_full_wing.m_loc(), desc_full_wing.n_loc());

    if (desc_body.nprows() > 1)
    {
        assert(desc_full_wing.m_loc() < desc_full_wing.m());
    }
    assert(desc_wing.m() == 11);
    assert(desc_wing.n() == 3);
    assert(desc_wing.mb() == desc_body.mb());
    assert(desc_wing.nb() == 1);
    assert(desc_wing.m_loc() == desc_full_wing.m_loc());
    assert(desc_wing.n_loc() == desc_full_wing.n_loc());
}

void test_headwing_spin_weights()
{
    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 2, false) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(2, false) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, true) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, true) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, false) - 0.375) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, false) - 2.0) < 1e-12);
}

void test_wing_cartesian_gram_is_invariant_under_row_phases()
{
    ComplexMatrix wing(2, 3);
    wing(0, 0) = {1.0, 1.0};
    wing(0, 1) = {2.0, -1.0};
    wing(0, 2) = {0.0, -1.0};
    wing(1, 0) = {0.5, -2.0};
    wing(1, 1) = {-1.0, 0.25};
    wing(1, 2) = {3.0, 0.5};

    ComplexMatrix phased = wing;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        phased(0, alpha) *= std::complex<double>{0.0, 1.0};
        phased(1, alpha) *= std::complex<double>{-1.0, 0.0};
    }

    const auto gram = librpa_int::compute_wing_cartesian_gram(wing);
    const auto phased_gram = librpa_int::compute_wing_cartesian_gram(phased);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            assert_complex_close(gram.at(alpha).at(beta), phased_gram.at(alpha).at(beta), 1e-12);
        }
    }
    assert_complex_close(gram.at(0).at(0), {6.25, 0.0}, 1e-12);
}

void test_velocity_matrix_initialization()
{
    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 3, 4);

    assert(velocity.size() == 2);
    for (int ispin = 0; ispin != 2; ++ispin)
    {
        assert(velocity[ispin].size() == 3);
        for (int ik = 0; ik != 3; ++ik)
        {
            assert(velocity[ispin][ik].size() == 3);
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                assert(velocity[ispin][ik][alpha].nr == 4);
                assert(velocity[ispin][ik][alpha].nc == 4);
            }
        }
    }
}

void test_headwing_local_kpoints_prefers_kpoint_blacs_context()
{
    const auto all_k = librpa_int::headwing_local_kpoints(4, nullptr);
    assert((all_k == std::vector<int>{0, 1, 2, 3}));

    if (librpa_int::get_mpi_size(MPI_COMM_WORLD) != 4) return;

    KPointBlacsProcessShape shape(2, 2, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 4);
    const auto local_k = librpa_int::headwing_local_kpoints(4, &kctx);

    if (kctx.kpoint_group_id() == 0)
        assert((local_k == std::vector<int>{0, 2}));
    else
        assert((local_k == std::vector<int>{1, 3}));

    const auto mismatched = librpa_int::headwing_local_kpoints(5, &kctx);
    assert((mismatched == std::vector<int>{0, 1, 2, 3, 4}));
}

void test_accumulate_wing_mu_for_pair_matches_original_formula()
{
    const std::vector<double> omega{0.5, 1.25};
    const std::array<std::complex<double>, 3> velocity{std::complex<double>{0.2, -0.1},
                                                       std::complex<double>{-0.3, 0.4},
                                                       std::complex<double>{0.15, 0.05}};
    const std::complex<double> c_mn{0.7, -0.2};
    const double egap = 1.8;
    const double factor1 = 0.6;
    const double factor2 = 0.125;

    std::array<std::complex<double>, 6> accumulated{};
    librpa_int::accumulate_wing_mu_for_pair(omega, velocity, c_mn, egap, factor1, factor2,
                                            accumulated.data());

    for (std::size_t iomega = 0; iomega != omega.size(); ++iomega)
    {
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            const auto denom = omega[iomega] * omega[iomega] + egap * egap;
            const auto expected = factor1 * std::conj(c_mn * velocity[alpha]) / denom +
                                  factor2 * c_mn * velocity[alpha] / denom;
            assert_complex_close(accumulated[iomega * 3 + alpha], expected, 1e-12);
        }
    }
}

SymmetryKStarMember make_headwing_wfc_atom_swap_member(
    const std::complex<double> &rot_0, const std::complex<double> &rot_1)
{
    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};

    SymmetryKAtomRotation atom_0;
    atom_0.atom_from = 0;
    atom_0.atom_to = 1;
    atom_0.atom_type = 0;
    atom_0.lmax = 0;
    atom_0.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_0.bloch_rsh_rotations[0](0, 0) = rot_0;

    SymmetryKAtomRotation atom_1;
    atom_1.atom_from = 1;
    atom_1.atom_to = 0;
    atom_1.atom_type = 0;
    atom_1.lmax = 0;
    atom_1.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_1.bloch_rsh_rotations[0](0, 0) = rot_1;

    member.atom_rotations = {atom_0, atom_1};
    return member;
}

void test_headwing_wfc_restore_applies_atom_permutation()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const auto member = make_headwing_wfc_atom_swap_member(
        {2.0, 0.5}, {3.0, -0.25});

    ComplexMatrix wfc_ibz(1, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), wfc_ibz(0, 1) * std::complex<double>{3.0, -0.25},
                         1e-12);
    assert_complex_close(wfc_bz(0, 1), wfc_ibz(0, 0) * std::complex<double>{2.0, 0.5},
                         1e-12);
}

void test_headwing_wfc_restore_applies_time_reversal()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};
    member.time_reversal = true;
    SymmetryKAtomRotation atom;
    atom.atom_from = 0;
    atom.atom_to = 0;
    atom.atom_type = 0;
    atom.lmax = 0;
    atom.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom.bloch_rsh_rotations[0](0, 0) = {0.25, 0.75};
    member.atom_rotations.push_back(atom);

    ComplexMatrix wfc_ibz(1, 1);
    wfc_ibz(0, 0) = {0.6, -0.35};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), std::conj(wfc_ibz(0, 0)) * std::complex<double>{0.25, 0.75},
                         1e-12);
}

void test_headwing_velocity_restore_uses_inverse_spatial_route()
{
    SymmetryContext ctx;
    ctx.lattice_vectors.Identity();
    SymmetryOperation operation;
    operation.rotation = Matrix3(0.0, -1.0, 0.0,
                                 1.0,  0.0, 0.0,
                                 0.0,  0.0, 1.0);
    ctx.rspace_operations = {operation};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    std::array<ComplexMatrix, 3> velocity_ibz;
    for (auto &component : velocity_ibz) component.create(1, 1);
    velocity_ibz[0](0, 0) = {1.0, 2.0};
    velocity_ibz[1](0, 0) = {3.0, -1.0};
    velocity_ibz[2](0, 0) = {-0.5, 0.25};

    const auto velocity_bz = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, false);
    assert_complex_close(velocity_bz[0](0, 0), -velocity_ibz[1](0, 0), 1e-12);
    assert_complex_close(velocity_bz[1](0, 0), velocity_ibz[0](0, 0), 1e-12);
    assert_complex_close(velocity_bz[2](0, 0), velocity_ibz[2](0, 0), 1e-12);

    member.time_reversal = true;
    const auto velocity_bz_tr = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, true);
    assert_complex_close(velocity_bz_tr[0](0, 0), std::conj(velocity_ibz[1](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[1](0, 0), -std::conj(velocity_ibz[0](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[2](0, 0), -std::conj(velocity_ibz[2](0, 0)), 1e-12);
}

void test_headwing_direct_full_bz_velocity_selects_kstar_member()
{
    librpa_int::velocity_matrix_t velocity_full;
    librpa_int::initialize_velocity_matrix(velocity_full, 1, 2, 1);
    velocity_full[0][0][0](0, 0) = {1.0, 0.0};
    velocity_full[0][1][0](0, 0) = {2.0, 0.0};
    velocity_full[0][1][1](0, 0) = {3.0, 0.0};
    velocity_full[0][1][2](0, 0) = {4.0, 0.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &velocity = librpa_int::direct_full_bz_velocity_for_kstar_member(
        velocity_full, member_source_ik, 0, 0, 0);
    assert_complex_close(velocity[0](0, 0), {2.0, 0.0}, 1e-12);
    assert_complex_close(velocity[1](0, 0), {3.0, 0.0}, 1e-12);
    assert_complex_close(velocity[2](0, 0), {4.0, 0.0}, 1e-12);
}

void test_headwing_direct_full_bz_wfc_selects_same_kstar_member()
{
    MeanField wfc_full(1, 2, 1, 1);
    auto &wfc_k0 = wfc_full.get_eigenvectors()[0][0][0];
    wfc_k0.create(1, 1);
    wfc_k0(0, 0) = {1.0, 0.0};
    auto &wfc_k1 = wfc_full.get_eigenvectors()[0][0][1];
    wfc_k1.create(1, 1);
    wfc_k1(0, 0) = {0.0, 1.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &wfc = librpa_int::direct_full_bz_wfc_for_kstar_member(
        wfc_full, member_source_ik, 0, 0, 0, 0);
    assert_complex_close(wfc(0, 0), {0.0, 1.0}, 1e-12);
}

RI::Tensor<double> make_single_value_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void compare_local_blacs_matrices(
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &actual,
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &expected,
    const double tolerance)
{
    assert(actual.first.m() == expected.first.m());
    assert(actual.first.n() == expected.first.n());
    assert(actual.first.m_loc() == expected.first.m_loc());
    assert(actual.first.n_loc() == expected.first.n_loc());
    for (int i = 0; i != actual.first.m_loc(); ++i)
    {
        for (int j = 0; j != actual.first.n_loc(); ++j)
        {
            assert_complex_close(actual.second(i, j), expected.second(i, j), tolerance);
        }
    }
}

void test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(
    const BlacsCtxtHandler &blacs_h)
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};

    auto member = make_headwing_wfc_atom_swap_member({0.0, 1.0}, {-1.0, 0.0});
    member.k_bz = {0.5, 0.0, 0.0};

    MeanField mf_ibz(1, 1, 2, 2, 1);
    auto &wfc_ibz = mf_ibz.get_eigenvectors()[0][0][0];
    wfc_ibz.create(2, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};
    wfc_ibz(1, 0) = {0.3, 0.5};
    wfc_ibz(1, 1) = {-0.8, -0.1};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, &member.k_bz);

    MeanField mf_full(1, 1, 2, 2, 1);
    auto &wfc_full = mf_full.get_eigenvectors()[0][0][0];
    wfc_full.create(2, 2);
    for (int ib = 0; ib != 2; ++ib)
    {
        for (int iao = 0; iao != 2; ++iao)
        {
            wfc_full(ib, iao) = wfc_bz(ib, iao);
        }
    }

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc(std::vector<size_t>{1, 1});
    AtomicBasis basis_abf(std::vector<size_t>{1, 1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac_ibz{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df_ibz(mf_ibz, velocity, kfrac_ibz, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                      pbc, librpa_int::global::mpi_comm_global_h, blacs_h);
    diele_func df_full(mf_full, velocity, {member.k_bz}, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                       pbc, librpa_int::global::mpi_comm_global_h, blacs_h);

    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = make_single_value_tensor(1.0);
    Cs_IJ[0][{1, {0, 0, 0}}] = make_single_value_tensor(-0.35);
    Cs_IJ[0][{1, {1, 0, 0}}] = make_single_value_tensor(0.42);

    std::vector<std::vector<const ComplexMatrix *>> restored_wfc_ptrs(
        1, std::vector<const ComplexMatrix *>(1, &wfc_bz));
    const auto restored = df_ibz.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz, &restored_wfc_ptrs);
    const auto full = df_full.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz);

    compare_local_blacs_matrices(restored, full, 1e-12);
}

void test_kblacs_transform_matches_original_transform(const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 2, 1);
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(2, 2);
    wfc(0, 0) = {0.7, 0.2};
    wfc(0, 1) = {-0.3, 0.4};
    wfc(1, 0) = {0.5, -0.1};
    wfc(1, 1) = {0.9, 0.3};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);

    auto tensor_data = std::make_shared<std::valarray<double>>(4);
    (*tensor_data)[0] = 1.0;
    (*tensor_data)[1] = 0.2;
    (*tensor_data)[2] = -0.4;
    (*tensor_data)[3] = 0.8;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_data);

    auto original = df.transform_Cs2mnk(0, 0, Cs_IJ);
    auto kblacs = df.transform_Cs2mnk_kblacs(0, 0, Cs_IJ, blacs_h, kfrac[0]);

    assert(original.first.m() == kblacs.first.m());
    assert(original.first.n() == kblacs.first.n());
    assert(original.first.m_loc() == kblacs.first.m_loc());
    assert(original.first.n_loc() == kblacs.first.n_loc());
    for (int i = 0; i != original.first.m_loc(); ++i)
    {
        for (int j = 0; j != original.first.n_loc(); ++j)
        {
            assert_complex_close(kblacs.second(i, j), original.second(i, j), 1e-12);
        }
    }
}

void test_transform_Cs2mnk_can_keep_spin_channels_separate(const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(2, 1, 2, 2, 1);
    auto &wfc_up = mf.get_eigenvectors()[0][0][0];
    auto &wfc_dn = mf.get_eigenvectors()[1][0][0];
    wfc_up.create(2, 2);
    wfc_dn.create(2, 2);
    wfc_up(0, 0) = {0.7, 0.2};
    wfc_up(0, 1) = {-0.3, 0.4};
    wfc_up(1, 0) = {0.5, -0.1};
    wfc_up(1, 1) = {0.9, 0.3};
    wfc_dn(0, 0) = {0.2, -0.6};
    wfc_dn(0, 1) = {0.8, 0.1};
    wfc_dn(1, 0) = {-0.4, 0.5};
    wfc_dn(1, 1) = {0.6, -0.2};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 2, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);

    auto tensor_data = std::make_shared<std::valarray<double>>(4);
    (*tensor_data)[0] = 1.0;
    (*tensor_data)[1] = 0.2;
    (*tensor_data)[2] = -0.4;
    (*tensor_data)[3] = 0.8;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_data);

    const auto all_spin = df.transform_Cs2mnk(0, 0, Cs_IJ);
    const auto spin_up = df.transform_Cs2mnk(0, 0, Cs_IJ, 0);
    const auto spin_dn = df.transform_Cs2mnk(0, 0, Cs_IJ, 1);

    bool spin_channels_differ = false;
    for (int i = 0; i != all_spin.first.m_loc(); ++i)
    {
        for (int j = 0; j != all_spin.first.n_loc(); ++j)
        {
            assert_complex_close(all_spin.second(i, j), spin_up.second(i, j) + spin_dn.second(i, j),
                                 1e-12);
            spin_channels_differ =
                spin_channels_differ || std::abs(spin_up.second(i, j) - spin_dn.second(i, j)) > 1e-12;
        }
    }
    assert(spin_channels_differ);
}

void test_head_initialization_does_not_require_coulomb_diagonalization(
    const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 1);
    mf.get_eigenvals()[0](0, 0) = -0.5;
    mf.get_eigenvals()[0](0, 1) = 0.5;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_weight()[0](0, 1) = 0.0;

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        velocity[0][0][alpha](1, 0) = std::complex<double>{0.1 * (alpha + 1), 0.0};
        velocity[0][0][alpha](0, 1) = std::complex<double>{0.1 * (alpha + 1), 0.0};
    }

    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    const atpair_k_cplx_mat_t empty_vq;

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 1, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);

    df.init(0.0, empty_vq);
    df.cal_head();
    assert(df.get_head_vec().size() == 1);
}

void test_strict_2d_gamma_quadrature_is_ready_after_wing_initialization(
    const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 1);
    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.0});
    const std::vector<double> kvecs{0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    librpa_int::PI,
                                    0.0,
                                    librpa_int::PI,
                                    0.0,
                                    0.0,
                                    librpa_int::PI,
                                    librpa_int::PI,
                                    0.0};
    pbc.set_kgrids_kvec(2, 2, 1, kvecs);
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    const atpair_k_cplx_mat_t empty_vq;

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 1, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);
    df.configure_strict_2d_coulomb_head(true, 1.0 / (4.0 * librpa_int::PI));
    assert(df.use_2d_dielectric);
    require_double_close(df.get_strict_2d_pw_to_auxiliary_scale(), 1.0, 1e-14);
    df.init_wing(0.0, empty_vq);

    const double average = df.get_strict_2d_bare_coulomb_gamma_average();
    assert(std::isfinite(average));
    assert(average > 0.0);
}

void add_scalar_wq_block(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const atom_t atom_i,
    const atom_t atom_j,
    const Vector3_Order<double> &q,
    const std::complex<double> value)
{
    auto &block = wq[atom_i][atom_j][q];
    block = matrix_m<std::complex<double>>(1, 1, MAJOR::ROW);
    block(0, 0) = value;
}

librpa_int::symmetry_atom_block_matrix_map_t scalar_wq_to_blocks(
    const std::map<atom_t, std::map<atom_t, std::complex<double>>> &values)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    for (const auto &[atom_i, row] : values)
    {
        for (const auto &[atom_j, value] : row)
        {
            blocks[atom_i][atom_j] = ComplexMatrix(1, 1);
            blocks[atom_i][atom_j](0, 0) = value;
        }
    }
    return blocks;
}

void add_scalar_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const Vector3_Order<double> &q,
    const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, block] : row)
        {
            add_scalar_wq_block(wq, atom_i, atom_j, q, block(0, 0));
        }
    }
}

PeriodicBoundaryData make_wq_full_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0};
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_wq_reduced_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0};
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}};
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);
    return pbc;
}

SymmetryContext make_two_atom_inversion_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}, {1, 0}},
        {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};

    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    inversion.translation = {0.0, 0.0, 0.0};

    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

void assert_wq_rspace_maps_close(
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &actual,
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &expected)
{
    for (const auto &[atom_i, expected_row] : expected)
    {
        assert(actual.count(atom_i) != 0);
        for (const auto &[atom_j, expected_Rs] : expected_row)
        {
            assert(actual.at(atom_i).count(atom_j) != 0);
            for (const auto &[R, expected_block] : expected_Rs)
            {
                assert(actual.at(atom_i).at(atom_j).count(R) != 0);
                const auto &actual_block = actual.at(atom_i).at(atom_j).at(R);
                if (std::abs(actual_block(0, 0) - expected_block(0, 0)) >= 1e-12)
                {
                    std::cerr << "atom_pair=(" << atom_i << "," << atom_j << ") R=("
                              << R.x << "," << R.y << "," << R.z << ")" << std::endl;
                }
                assert_complex_close(actual_block(0, 0), expected_block(0, 0), 1e-12);
            }
        }
    }
}

void test_wq_to_wr_symmetry_reduced_q_matches_full_bz()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
    add_scalar_wq_blocks(wq_sym, q_gamma_sym, gamma_blocks);
    add_scalar_wq_blocks(wq_sym, q_rep_sym, rep_blocks);
    const double symmetry_collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
    {
        for (auto &[atom_j, q_blocks] : row)
        {
            for (auto &[q, block] : q_blocks)
            {
                block *= symmetry_collective_scale;
            }
        }
    }

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

void test_wq_to_wr_symmetry_collective_handles_empty_local_rank()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        const auto gamma_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
            {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks({
            {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
            {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
            {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(actual.empty());
    }
}

Matz dense_wq_from_scalar_blocks(const librpa_int::symmetry_atom_block_matrix_map_t &blocks,
                                 const ArrayDesc &desc)
{
    Matz mat(desc.m_loc(), desc.n_loc(), MAJOR::COL);
    for (int i_local = 0; i_local < desc.m_loc(); ++i_local)
    {
        const int atom_i = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local < desc.n_loc(); ++j_local)
        {
            const int atom_j = desc.indx_l2g_c(j_local);
            mat(i_local, j_local) = blocks.at(static_cast<atom_t>(atom_i))
                                        .at(static_cast<atom_t>(atom_j))(0, 0);
        }
    }
    return mat;
}

void assert_dense_wq_rspace_maps_close(
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &actual,
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &expected)
{
    for (const auto &[freq, expected_Rs] : expected)
    {
        assert(actual.count(freq) != 0);
        for (const auto &[R, expected_mat] : expected_Rs)
        {
            assert(actual.at(freq).count(R) != 0);
            const auto diff = actual.at(freq).at(R) - expected_mat;
            double max_abs = 0.0;
            for (int i = 0; i < diff.nr(); ++i)
            {
                for (int j = 0; j < diff.nc(); ++j)
                {
                    max_abs = std::max(max_abs, std::abs(diff(i, j)));
                }
            }
            if (max_abs >= 1e-12)
            {
                std::cerr << "freq=" << freq << " R=(" << R.x << "," << R.y << "," << R.z
                          << ") max_abs=" << max_abs << std::endl;
                for (int i = 0; i < diff.nr(); ++i)
                {
                    for (int j = 0; j < diff.nc(); ++j)
                    {
                        std::cerr << "  (" << i << "," << j << ") actual="
                                  << actual.at(freq).at(R)(i, j) << " expected="
                                  << expected_mat(i, j) << " diff=" << diff(i, j) << std::endl;
                    }
                }
            }
            assert(max_abs < 1e-12);
        }
    }
}

void test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);
    const auto qpoint_view = build_symmetry_qpoint_view(ctx, pbc_sym, true);
    assert(qpoint_view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};
    ArrayDesc ad_Wc(blacs_h);
    ad_Wc.init(2, 2, 2, 2, 0, 0);

    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});

    constexpr double freq = 0.25;
    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_sym;
    wq_sym[freq][pbc_sym.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_sym[freq][pbc_sym.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);

    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_full;
    wq_full[freq][pbc_full.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_full[freq][pbc_full.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    wq_full[freq][pbc_full.klist.at(2)] =
        dense_wq_from_scalar_blocks(inversion_minus_blocks, ad_Wc);

    const auto expected = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_full, pbc_full, false);
    const auto actual = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_sym, pbc_sym, false,
        &qpoint_view, &ctx, &basis_abf, &ad_Wc);

    assert_dense_wq_rspace_maps_close(actual, expected);
}

void test_gamma_only_dense_wq_fourier_weight_scales_as_inverse_bvk_cells()
{
    constexpr double frequency = 0.25;
    const std::complex<double> gamma_value{2.4, -0.3};
    const std::array<int, 4> meshes{12, 14, 16, 20};

    for (const int mesh : meshes)
    {
        PeriodicBoundaryData pbc;
        pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        std::vector<double> kvecs;
        kvecs.reserve(static_cast<std::size_t>(3 * mesh * mesh));
        for (int ix = 0; ix != mesh; ++ix)
        {
            for (int iy = 0; iy != mesh; ++iy)
            {
                kvecs.push_back(librpa_int::TWO_PI * ix / mesh);
                kvecs.push_back(librpa_int::TWO_PI * iy / mesh);
                kvecs.push_back(0.0);
            }
        }
        pbc.set_kgrids_kvec(mesh, mesh, 1, kvecs);

        std::map<double, std::map<Vector3_Order<double>, Matz>> wq;
        if (librpa_int::global::mpi_comm_global_h.is_root())
        {
            Matz gamma(1, 1, MAJOR::COL);
            gamma(0, 0) = gamma_value;
            wq[frequency][pbc.klist.at(0)] = gamma;
        }

        const auto wr =
            librpa_int::FT_Wc_freq_q(librpa_int::global::mpi_comm_global_h, wq, pbc, false);
        if (librpa_int::global::mpi_comm_global_h.is_root())
        {
            const Vector3_Order<int> center{0, 0, 0};
            const auto expected = gamma_value / static_cast<double>(mesh * mesh);
            assert_complex_close(wr.at(frequency).at(center)(0, 0), expected, 1e-13);
        }
        else
        {
            assert(wr.empty());
        }
    }
}

}  // namespace

int main(int argc, char *argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io(false, "stdout", false);

    {
        BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
        blacs_h.init();
        blacs_h.set_square_grid();

        test_replace_rpa_response_headwing_replaces_only_singular_channels(blacs_h);
        test_rspace_symmetry_requires_complete_band_space();
        test_kpoint_coordinate_mapping_selects_active_klist_from_full_source();
        test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge();
        test_replace_rpa_response_head_only_keeps_numeric_wings(blacs_h);
        test_head_only_trace_logdet_can_use_reduced_response(blacs_h);
        test_rpa_trace_log_average_uses_directional_head_and_wing();
        test_rpa_headwing_regular_body_start_channel();
        test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice();
        test_strict_2d_headwing_prefactors_use_inplane_area();
        test_strict_2d_auxiliary_normalization_is_computed_from_basis_metadata();
        test_strict_2d_gamma_cell_uses_physical_reciprocal_measure();
        test_strict_2d_radial_integrals_match_analytic_values();
        test_strict_2d_radial_integrals_are_stable_at_zero_and_small_a();
        test_strict_2d_inverse_head_average_has_linear_q_screening();
        test_strict_2d_finite_q_reference_matches_head_and_schur_limits();
        test_strict_2d_schur_coefficient_removes_identity();
        test_strict_2d_screening_denominator_must_stay_on_physical_branch();
        test_strict_2d_gw_uses_full_coulomb_at_all_q();
        test_strict_2d_gw_routes_gamma_through_complete_wc_average();
        test_strict_2d_block_metrics_separate_head_wings_and_body();
        test_strict_2d_alpha_reference_averages_bare_coulomb();
        test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis();
        test_strict_2d_wc_blocks_match_dense_finite_q_inverse();
        test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature();
        test_strict_2d_wc_blocks_have_finite_small_q_limits();
        test_strict_2d_wc_average_is_covariant_under_regular_body_rotation();
        test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks();
        test_rpa_chi0v_wing_desc_uses_global_rows(blacs_h);
        test_headwing_spin_weights();
        test_wing_cartesian_gram_is_invariant_under_row_phases();
        test_velocity_matrix_initialization();
        test_headwing_local_kpoints_prefers_kpoint_blacs_context();
        test_accumulate_wing_mu_for_pair_matches_original_formula();
        test_headwing_wfc_restore_applies_atom_permutation();
        test_headwing_wfc_restore_applies_time_reversal();
        test_headwing_velocity_restore_uses_inverse_spatial_route();
        test_headwing_direct_full_bz_velocity_selects_kstar_member();
        test_headwing_direct_full_bz_wfc_selects_same_kstar_member();
        test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(blacs_h);
        test_kblacs_transform_matches_original_transform(blacs_h);
        test_transform_Cs2mnk_can_keep_spin_channels_separate(blacs_h);
        test_head_initialization_does_not_require_coulomb_diagonalization(blacs_h);
        test_strict_2d_gamma_quadrature_is_ready_after_wing_initialization(blacs_h);
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_gamma_only_dense_wq_fourier_weight_scales_as_inverse_bvk_cells();
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
