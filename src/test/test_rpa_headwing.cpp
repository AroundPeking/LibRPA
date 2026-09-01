#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <valarray>

#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/gw.h"
#include "../core/qpoint_view.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "../utils/constants.h"

using librpa_int::ArrayDesc;
using librpa_int::atom_mapping;
using librpa_int::atom_t;
using librpa_int::AtomicBasis;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::BlacsCtxtHandler;
using librpa_int::build_symmetry_qpoint_view;
using librpa_int::C_ONE;
using librpa_int::ComplexMatrix;
using librpa_int::diele_func;
using librpa_int::init_local_mat;
using librpa_int::KPointBlacsParallelContext;
using librpa_int::KPointBlacsProcessShape;
using librpa_int::MAJOR;
using librpa_int::Matrix3;
using librpa_int::matrix_m;
using librpa_int::Matz;
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

namespace
{

void test_complex_spacetime_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::force_complex_spacetime_diagnostic_requested(nullptr));
    assert(!librpa_int::force_complex_spacetime_diagnostic_requested(""));
    assert(librpa_int::force_complex_spacetime_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::force_complex_spacetime_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_complex_spacetime_storage_does_not_change_non_soc_spin_weight()
{
    if (std::abs(librpa_int::chi0_spacetime_spin_scale(1, 1) - 2.0) >= 1.0e-15 ||
        std::abs(librpa_int::chi0_spacetime_spin_scale(2, 1) - 1.0) >= 1.0e-15 ||
        std::abs(librpa_int::chi0_spacetime_spin_scale(1, 2) - 1.0) >= 1.0e-15)
        throw std::runtime_error(
            "complex chi0 storage must not change the physical non-SOC spin weight");
}

void test_direct_compressed_sigc_diagnostic_requires_explicit_shrink_path()
{
    assert(!librpa_int::direct_compressed_sigc_diagnostic_requested(nullptr));
    assert(!librpa_int::direct_compressed_sigc_diagnostic_requested(""));
    assert(librpa_int::direct_compressed_sigc_diagnostic_requested("enabled"));
    assert(librpa_int::should_contract_sigc_in_compressed_abfs(true, "enabled"));
    assert(!librpa_int::should_contract_sigc_in_compressed_abfs(true, nullptr));

    bool rejected_without_shrink = false;
    try
    {
        (void)librpa_int::should_contract_sigc_in_compressed_abfs(false, "enabled");
    }
    catch (const std::invalid_argument &)
    {
        rejected_without_shrink = true;
    }
    assert(rejected_without_shrink);

    bool rejected_invalid_value = false;
    try
    {
        (void)librpa_int::direct_compressed_sigc_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_chi0_rspace_symmetry_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::disable_chi0_rspace_symmetry_diagnostic_requested(nullptr));
    assert(!librpa_int::disable_chi0_rspace_symmetry_diagnostic_requested(""));
    assert(librpa_int::disable_chi0_rspace_symmetry_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::disable_chi0_rspace_symmetry_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_gamma_shrink_transform_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::use_gamma_shrink_transform_diagnostic_requested(nullptr));
    assert(!librpa_int::use_gamma_shrink_transform_diagnostic_requested(""));
    assert(librpa_int::use_gamma_shrink_transform_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::use_gamma_shrink_transform_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_chi0_qspace_symmetry_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::disable_chi0_qspace_symmetry_diagnostic_requested(nullptr));
    assert(!librpa_int::disable_chi0_qspace_symmetry_diagnostic_requested(""));
    assert(librpa_int::disable_chi0_qspace_symmetry_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::disable_chi0_qspace_symmetry_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_rpa_finite_q_matrix_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::rpa_finite_q_matrix_diagnostic_requested(nullptr));
    assert(!librpa_int::rpa_finite_q_matrix_diagnostic_requested(""));
    assert(librpa_int::rpa_finite_q_matrix_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::rpa_finite_q_matrix_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_distributed_hermiticity_metrics_on_one_rank(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(3, 3, 0, 0);
    auto matrix = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    const matrix_m<std::complex<double>> dense(
        std::vector<std::vector<std::complex<double>>>{
            {{1.0, 0.0}, {0.2, 0.3}, {-0.1, 0.05}},
            {{0.2, -0.3}, {2.0, 0.0}, {0.4, -0.2}},
            {{-0.1, -0.05}, {0.4, 0.2}, {3.0, 0.0}}},
        MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int iloc = desc.indx_g2l_r(i);
        for (int j = 0; j != 3; ++j)
        {
            const int jloc = desc.indx_g2l_c(j);
            matrix(iloc, jloc) = dense(i, j);
        }
    }

    const auto exact = librpa_int::distributed_hermiticity_metrics(matrix, desc);
    assert(exact.frobenius_norm > 0.0);
    assert(exact.antihermitian_frobenius_norm < 1.0e-14);
    assert(exact.antihermitian_max_abs < 1.0e-14);
    assert(exact.relative_frobenius_residual < 1.0e-14);

    matrix(desc.indx_g2l_r(0), desc.indx_g2l_c(2)) += std::complex<double>(0.25, 0.0);
    const auto perturbed = librpa_int::distributed_hermiticity_metrics(matrix, desc);
    assert(std::abs(perturbed.antihermitian_frobenius_norm - std::sqrt(0.125)) < 1.0e-14);
    assert(std::abs(perturbed.antihermitian_max_abs - 0.25) < 1.0e-14);
    assert(perturbed.relative_frobenius_residual > 0.0);
}

void test_sigc_rspace_symmetry_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::disable_sigc_rspace_symmetry_diagnostic_requested(nullptr));
    assert(!librpa_int::disable_sigc_rspace_symmetry_diagnostic_requested(""));
    assert(librpa_int::disable_sigc_rspace_symmetry_diagnostic_requested("enabled"));

    bool rejected_invalid_value = false;
    try
    {
        librpa_int::disable_sigc_rspace_symmetry_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected_invalid_value = true;
    }
    assert(rejected_invalid_value);
}

void test_single_q_member_diagnostic_disables_sigc_rspace_symmetry_restore()
{
    if (!librpa_int::should_use_sigc_rspace_symmetry(true, true, nullptr, nullptr))
        throw std::runtime_error("complete crystal q stars should keep Sigma_c symmetry restore");
    if (librpa_int::should_use_sigc_rspace_symmetry(true, true, nullptr,
                                                    "0,0.083333333333333333,0"))
        throw std::runtime_error(
            "a retained single q member breaks the crystal symmetry of Sigma_c");
    if (librpa_int::should_use_sigc_rspace_symmetry(true, true, "enabled", nullptr))
        throw std::runtime_error("the explicit Sigma_c symmetry diagnostic switch was ignored");
}

void test_strict_2d_qmember_diagnostic_selects_one_periodic_member()
{
    const auto require = [](const bool condition)
    {
        if (!condition) throw std::runtime_error("single q-member periodic selection regression");
    };
    const Vector3_Order<double> selected{0.0, 1.0 / 12.0, 0.0};
    const std::vector<Vector3_Order<double>> first_star{
        {0.0, 1.0 / 12.0, 0.0},  {0.0, -1.0 / 12.0, 0.0},        {1.0 / 12.0, 0.0, 0.0},
        {-1.0 / 12.0, 0.0, 0.0}, {1.0 / 12.0, -1.0 / 12.0, 0.0}, {-1.0 / 12.0, 1.0 / 12.0, 0.0},
    };

    for (const auto &q : first_star)
        require(librpa_int::strict_2d_qmember_diagnostic_keeps(q, selected, false));
    require(librpa_int::strict_2d_qmember_diagnostic_keeps(first_star.front(), selected, true));
    require(librpa_int::strict_2d_qmember_diagnostic_keeps(
        Vector3_Order<double>{0.0, -11.0 / 12.0, 0.0}, selected, true));
    for (std::size_t i = 1; i != first_star.size(); ++i)
        require(!librpa_int::strict_2d_qmember_diagnostic_keeps(first_star[i], selected, true));
}

void test_strict_2d_qmember_diagnostic_accepts_one_distributed_owner()
{
    const auto require = [](const bool condition)
    {
        if (!condition) throw std::runtime_error("single q-member MPI ownership regression");
    };
    for (const std::size_t local_count : {1U, 0U, 0U, 0U})
        require(librpa_int::strict_2d_qmember_diagnostic_selection_valid(local_count, 1, true));
    require(!librpa_int::strict_2d_qmember_diagnostic_selection_valid(0, 0, true));
    require(!librpa_int::strict_2d_qmember_diagnostic_selection_valid(2, 2, true));
    require(librpa_int::strict_2d_qmember_diagnostic_selection_valid(7, 0, false));
}

void test_wc_rf_output_collective_includes_ranks_without_local_frequency_blocks()
{
    if (!librpa_int::should_enter_wc_rf_collective(true, true) ||
        !librpa_int::should_enter_wc_rf_collective(true, false) ||
        librpa_int::should_enter_wc_rf_collective(false, true) ||
        librpa_int::should_enter_wc_rf_collective(false, false))
        throw std::runtime_error("Wc(R) output must keep every MPI rank in collective transforms");
}

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
        {0.0, 0.0, 0.0},   {0.125, 0.0, 0.0},   {0.25, 0.0, 0.0}, {0.375, 0.0, 0.0},
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

    const auto mapping =
        librpa_int::map_symmetry_kstar_members_to_source_kpoints(ctx, ibz_kpoints, full_bz_kpoints);

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
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, false), vol_3d / 64.0,
                         1e-14);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, true), vol_2d / 64.0,
                         1e-14);
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
    pbc.set_latvec({19.390653825130212, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0});

    constexpr double multipole_norm_squared = 2205.0673846924301;
    const auto normalization =
        librpa_int::strict_2d_coulomb_head_normalization(pbc, multipole_norm_squared);

    require_double_close(normalization.inplane_area_bohr2, 19.390653825130212, 1e-13);
    require_double_close(normalization.auxiliary_head_coefficient, 8978.8175111265446, 1e-10);
    require_double_close(normalization.pw_to_auxiliary_scale, 37.802423070695596, 1e-12);

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

void test_strict_2d_radial_log_integral_matches_numeric_quadrature()
{
    constexpr std::complex<double> a{0.37, 0.0};
    constexpr double qmax = 0.61;
    constexpr int n = 200000;
    const double dq = qmax / n;
    std::complex<double> numeric = 0.0;
    for (int i = 0; i != n; ++i)
    {
        const double q = (i + 0.5) * dq;
        numeric += q * std::log(1.0 + a * q) * dq;
    }
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(a, qmax), numeric, 2e-13);
}

void test_strict_2d_radial_log_integral_is_stable_at_zero_and_small_a()
{
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(0.0, 0.7), 0.0, 1e-15);
    constexpr std::complex<double> a{1.0e-10, 0.0};
    constexpr double qmax = 0.7;
    const auto expected = a * std::pow(qmax, 3) / 3.0 - a * a * std::pow(qmax, 4) / 8.0;
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(a, qmax), expected, 1e-24);
}

void test_strict_2d_rpa_trace_log_average_preserves_cell_normalization()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    head(0, 0) = -0.28;
    head(0, 1) = 0.03;
    head(1, 0) = 0.03;
    head(1, 1) = -0.16;
    lind(0, 0) = 1.41;
    lind(0, 1) = 0.04;
    lind(1, 0) = 0.04;
    lind(1, 1) = 1.23;
    lind(2, 2) = 1.0;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> angular_weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax{0.21, 0.14, 0.21, 0.14};
    double gamma_area = 0.0;
    for (std::size_t i = 0; i != qmax.size(); ++i)
        gamma_area += angular_weights[i] * qmax[i] * qmax[i] / 2.0;

    const std::complex<double> trace_body{0.12, 0.0};
    const std::complex<double> logdet_body = std::log(0.88);
    double weight_sum = 0.0;
    std::complex<double> body_part = 0.0;
    std::complex<double> head_part = 0.0;
    std::complex<double> schur_part = 0.0;
    const auto result = librpa_int::compute_strict_2d_rpa_chi0v_trace_log_average(
        head, lind, trace_body, logdet_body, qx, qy, angular_weights, qmax, gamma_area, &weight_sum,
        &body_part, &head_part, &schur_part);

    require_double_close(weight_sum, 1.0, 1e-14);
    assert_complex_close(body_part, trace_body + logdet_body, 1e-14);
    assert_complex_close(result, body_part + head_part + schur_part, 1e-14);
}

void test_strict_2d_rpa_trace_log_average_matches_dense_radial_quadrature()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    head(0, 0) = -0.28;
    head(0, 1) = 0.03;
    head(1, 0) = 0.03;
    head(1, 1) = -0.16;
    lind(0, 0) = 1.41;
    lind(0, 1) = 0.04;
    lind(1, 0) = 0.04;
    lind(1, 1) = 1.23;
    lind(2, 2) = 1.0;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> angular_weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax{0.21, 0.14, 0.21, 0.14};
    double gamma_area = 0.0;
    for (std::size_t i = 0; i != qmax.size(); ++i)
        gamma_area += angular_weights[i] * qmax[i] * qmax[i] / 2.0;

    const std::complex<double> trace_body{0.12, 0.0};
    const std::complex<double> logdet_body = std::log(0.88);
    const auto analytic = librpa_int::compute_strict_2d_rpa_chi0v_trace_log_average(
        head, lind, trace_body, logdet_body, qx, qy, angular_weights, qmax, gamma_area);

    std::complex<double> numeric = 0.0;
    constexpr int radial_points = 200000;
    for (std::size_t idir = 0; idir != qx.size(); ++idir)
    {
        const double nx = qx[idir];
        const double ny = qy[idir];
        const auto directional_head =
            nx * (nx * head(0, 0) + ny * head(0, 1)) + ny * (nx * head(1, 0) + ny * head(1, 1));
        const auto a = librpa_int::strict_2d_schur_coefficient(lind, nx, ny);
        const double dq = qmax[idir] / radial_points;
        for (int ir = 0; ir != radial_points; ++ir)
        {
            const double q = (ir + 0.5) * dq;
            const auto integrand =
                trace_body + logdet_body + directional_head * q + std::log(1.0 + a * q);
            numeric += angular_weights[idir] * q * dq * integrand / gamma_area;
        }
    }
    assert_complex_close(analytic, numeric, 2e-11);
}

void test_strict_2d_rpa_trace_log_average_rejects_invalid_geometry()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    lind(0, 0) = 1.2;
    lind(1, 1) = 1.2;
    lind(2, 2) = 1.0;
    const std::vector<double> qx{1.0};
    const std::vector<double> qy{0.0};
    const std::vector<double> weights{librpa_int::TWO_PI};
    const std::vector<double> qmax{0.2};
    const double gamma_area = librpa_int::PI * qmax[0] * qmax[0];

    const auto rejected =
        [&](const matrix_m<std::complex<double>> &test_head,
            const matrix_m<std::complex<double>> &test_lind, const std::vector<double> &test_qx,
            const std::vector<double> &test_qy, const std::vector<double> &test_weights,
            const std::vector<double> &test_qmax, const double test_area)
    {
        try
        {
            (void)librpa_int::compute_strict_2d_rpa_chi0v_trace_log_average(
                test_head, test_lind, 0.0, 0.0, test_qx, test_qy, test_weights, test_qmax,
                test_area);
        }
        catch (const std::logic_error &)
        {
            return true;
        }
        return false;
    };

    matrix_m<std::complex<double>> bad_head(2, 2, MAJOR::COL);
    assert(rejected(bad_head, lind, qx, qy, weights, qmax, gamma_area));
    assert(rejected(head, lind, {0.5}, {0.5}, weights, qmax, gamma_area));
    assert(rejected(head, lind, qx, qy, {}, qmax, gamma_area));
    assert(rejected(head, lind, qx, qy, weights, qmax, 0.0));

    auto bad_lind = lind.copy();
    bad_lind(0, 0) = -9.0;
    assert(rejected(head, bad_lind, qx, qy, weights, qmax, gamma_area));
}

void test_strict_2d_rpa_trace_log_route_is_qavg_only()
{
    librpa_int::RpaHeadwingSettings settings;
    settings.use_2d_dielectric = true;
    settings.rpa_headwing_mode = "qavg";
    assert(librpa_int::use_strict_2d_rpa_trace_log_average(settings));

    settings.use_2d_dielectric = false;
    assert(!librpa_int::use_strict_2d_rpa_trace_log_average(settings));
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
    const auto require_route = [](const bool condition, const char *message)
    {
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

void test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration()
{
    const auto require_condition = [](const bool condition, const char *message)
    {
        if (!condition)
        {
            std::cerr << message << std::endl;
            std::abort();
        }
    };
    require_condition(librpa_int::strict_2d_complete_wc_requested(true, 3, true),
                      "strict 2D complete-Wc request was not recognized");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(false, 3, true),
                      "disabled replacement was classified as strict 2D");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(true, 2, true),
                      "non-head/wing dielectric mode was classified as strict 2D");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(true, 3, false),
                      "3D dielectric mode was classified as strict 2D");

    librpa_int::validate_strict_2d_complete_wc_runtime(false, false, false);
    librpa_int::validate_strict_2d_complete_wc_runtime(true, true, true);

    bool rejected_missing_data = false;
    try
    {
        librpa_int::validate_strict_2d_complete_wc_runtime(true, false, true);
    }
    catch (const std::logic_error &)
    {
        rejected_missing_data = true;
    }
    require_condition(rejected_missing_data,
                      "strict 2D GW silently accepted missing analytic head/wing data");

    bool rejected_dense_wc = false;
    try
    {
        librpa_int::validate_strict_2d_complete_wc_runtime(true, true, false);
    }
    catch (const std::logic_error &)
    {
        rejected_dense_wc = true;
    }
    require_condition(rejected_dense_wc,
                      "strict 2D GW silently accepted a path without complete-Wc support");
}

void test_strict_2d_diagnostic_schema_and_qpoint_order_are_stable()
{
    const auto count_columns = [](const std::string &header)
    { return 1 + static_cast<int>(std::count(header.begin(), header.end(), ',')); };
    const auto finite_q_header = librpa_int::strict_2d_finite_q_diagnostics_header();
    const auto raw_gamma_chi0_header =
        librpa_int::strict_2d_raw_gamma_chi0_diagnostics_header();
    if (count_columns(finite_q_header) != 75 ||
        count_columns(raw_gamma_chi0_header) != 9 ||
        count_columns(librpa_int::strict_2d_gamma_wc_diagnostics_header()) != 19 ||
        count_columns(librpa_int::strict_2d_gamma_wc_transform_diagnostics_header()) != 12 ||
        finite_q_header.find(
            "coulomb_head_eigenvalue,q_coulomb_head_eigenvalue,current_basis_head_column") ==
            std::string::npos ||
        finite_q_header.find("chi0_head_real,chi0_head_imag,chi0_head_over_q2_real") ==
            std::string::npos ||
        finite_q_header.find("current_basis_p_head_real,current_basis_p_head_imag") ==
            std::string::npos ||
        finite_q_header.find("current_basis_chi0_head_real,current_basis_chi0_head_imag") ==
            std::string::npos ||
        finite_q_header.find("current_basis_wc_head_real,current_basis_wc_head_imag") ==
            std::string::npos ||
        finite_q_header.find(
            "analytic_finite_q_wc_head_real,analytic_finite_q_wc_head_imag,"
            "analytic_finite_q_wc_head_body_fro,analytic_finite_q_wc_body_head_fro,"
            "analytic_finite_q_wc_body_body_fro") == std::string::npos ||
        raw_gamma_chi0_header !=
            "ifreq,frequency,head_column,chi0_head_real,chi0_head_imag,"
            "chi0_head_body_fro,chi0_body_head_fro,chi0_body_body_fro,n_auxiliary")
    {
        std::cerr << "strict 2D diagnostic CSV schema changed unexpectedly" << std::endl;
        std::abort();
    }

    const std::array<double, 4> ascending{-2.0, 0.1, 3.0, 8.0};
    const std::array<double, 4> descending{8.0, 3.0, 0.1, -2.0};
    if (librpa_int::strict_2d_head_eigenvector_column(ascending.data(),
                                                      static_cast<int>(ascending.size())) != 3 ||
        librpa_int::strict_2d_head_eigenvector_column(descending.data(),
                                                      static_cast<int>(descending.size())) != 0)
    {
        std::cerr << "strict 2D diagnostic did not select the largest Coulomb channel" << std::endl;
        std::abort();
    }

    if (librpa_int::strict_2d_diagnostic_head_first_index(0, 3) != 1 ||
        librpa_int::strict_2d_diagnostic_head_first_index(1, 3) != 2 ||
        librpa_int::strict_2d_diagnostic_head_first_index(2, 3) != 3 ||
        librpa_int::strict_2d_diagnostic_head_first_index(3, 3) != 0 ||
        librpa_int::strict_2d_diagnostic_head_first_index(4, 3) != 4)
    {
        std::cerr << "strict 2D diagnostic did not remap an arbitrary head column first"
                  << std::endl;
        std::abort();
    }

    const std::vector<Vector3_Order<double>> qpoints{
        {0.1, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.1, 0.0}};
    const auto unchanged = librpa_int::strict_2d_diagnostic_qpoint_order(qpoints, false);
    if (!(unchanged == qpoints))
    {
        std::cerr << "disabled strict 2D diagnostics changed q-point order" << std::endl;
        std::abort();
    }
    const auto ordered = librpa_int::strict_2d_diagnostic_qpoint_order(qpoints, true);
    if (!librpa_int::is_gamma_point(ordered.front()) || ordered.size() != qpoints.size())
    {
        std::cerr << "strict 2D diagnostics did not place Gamma first" << std::endl;
        std::abort();
    }
}

void test_strict_2d_qshell_uses_minimum_image_q()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    const Vector3_Order<double> wrapped_q{11.0 / 12.0, 1.0 / 12.0, 0.0};
    const auto minimum_q = librpa_int::strict_2d_minimum_image_q(pbc, wrapped_q);
    require_double_close(minimum_q.x, -1.0 / 12.0, 1.0e-14);
    require_double_close(minimum_q.y, 1.0 / 12.0, 1.0e-14);
    require_double_close(minimum_q.z, 0.0, 1.0e-14);

    using librpa_int::classify_strict_2d_qshell;
    using librpa_int::Strict2dQshellRegion;
    const double first_q_norm = 1.0 / 12.0;
    assert(classify_strict_2d_qshell(0.0, first_q_norm) == Strict2dQshellRegion::gamma);
    assert(classify_strict_2d_qshell(first_q_norm, first_q_norm) == Strict2dQshellRegion::first);
    assert(classify_strict_2d_qshell(2.0 * first_q_norm, first_q_norm) ==
           Strict2dQshellRegion::rest);
}

void test_strict_2d_qradial_partitions_rest_exactly()
{
    using librpa_int::classify_strict_2d_qradial;
    using librpa_int::Strict2dQradialRegion;

    const double first_q_norm = 0.125;
    assert(classify_strict_2d_qradial(0.0, first_q_norm) ==
           Strict2dQradialRegion::gamma_or_first);
    assert(classify_strict_2d_qradial(first_q_norm, first_q_norm) ==
           Strict2dQradialRegion::gamma_or_first);
    assert(classify_strict_2d_qradial(1.5 * first_q_norm, first_q_norm) ==
           Strict2dQradialRegion::near);
    assert(classify_strict_2d_qradial(2.5 * first_q_norm, first_q_norm) ==
           Strict2dQradialRegion::near);
    assert(classify_strict_2d_qradial(2.5 * first_q_norm + 1.0e-9, first_q_norm) ==
           Strict2dQradialRegion::middle);
    assert(classify_strict_2d_qradial(4.5 * first_q_norm, first_q_norm) ==
           Strict2dQradialRegion::middle);
    assert(classify_strict_2d_qradial(4.5 * first_q_norm + 1.0e-9, first_q_norm) ==
           Strict2dQradialRegion::far);

    // The two numerically distinct lengths below are the same first shell in
    // the MoS2 N12 grid; the difference comes from rounded lattice data.
    const double mos2_first_q_norm = 0.1004876;
    const double mos2_equivalent_first_q_norm = 0.1004992;
    assert(classify_strict_2d_qradial(mos2_equivalent_first_q_norm, mos2_first_q_norm) ==
           Strict2dQradialRegion::gamma_or_first);

    assert(!librpa_int::strict_2d_qradial_is_corner(6.5 * first_q_norm, first_q_norm));
    assert(librpa_int::strict_2d_qradial_is_corner(
        (6.5 + 1.0e-9) * first_q_norm, first_q_norm));

    const double hbn_first_q_norm = 0.12777240809249668;
    assert(!librpa_int::strict_2d_qradial_is_corner(
        6.0 * hbn_first_q_norm, hbn_first_q_norm));
    assert(librpa_int::strict_2d_qradial_is_corner(
        6.928203230275509 * hbn_first_q_norm, hbn_first_q_norm));
}

void test_strict_2d_first_shell_wc_block_diagnostic_is_exactly_additive()
{
    using librpa_int::strict_2d_first_shell_wc_block_diagnostic;
    using librpa_int::strict_2d_wc_block_keeps;
    using librpa_int::Strict2dWcBlock;

    assert(strict_2d_first_shell_wc_block_diagnostic(nullptr) == Strict2dWcBlock::full);
    assert(strict_2d_first_shell_wc_block_diagnostic("") == Strict2dWcBlock::full);
    assert(strict_2d_first_shell_wc_block_diagnostic("head") == Strict2dWcBlock::head);
    assert(strict_2d_first_shell_wc_block_diagnostic("wing") == Strict2dWcBlock::wing);
    assert(strict_2d_first_shell_wc_block_diagnostic("body") == Strict2dWcBlock::body);

    constexpr int dimension = 4;
    constexpr int head_index = 2;
    for (int row = 0; row != dimension; ++row)
    {
        for (int column = 0; column != dimension; ++column)
        {
            const int kept_count =
                static_cast<int>(strict_2d_wc_block_keeps(Strict2dWcBlock::head, row, column,
                                                          head_index)) +
                static_cast<int>(strict_2d_wc_block_keeps(Strict2dWcBlock::wing, row, column,
                                                          head_index)) +
                static_cast<int>(strict_2d_wc_block_keeps(Strict2dWcBlock::body, row, column,
                                                          head_index));
            assert(kept_count == 1);

            const std::complex<double> value{1.0 + row, -2.0 - column};
            std::complex<double> reconstructed{0.0, 0.0};
            for (const auto block :
                 {Strict2dWcBlock::head, Strict2dWcBlock::wing, Strict2dWcBlock::body})
                if (strict_2d_wc_block_keeps(block, row, column, head_index))
                    reconstructed += value;
            if (reconstructed != value) std::abort();
        }
    }

    for (const auto block :
         {Strict2dWcBlock::head, Strict2dWcBlock::wing, Strict2dWcBlock::body})
        for (int row = 0; row != dimension; ++row)
            for (int column = 0; column != dimension; ++column)
                assert(strict_2d_wc_block_keeps(block, row, column, head_index) ==
                       strict_2d_wc_block_keeps(block, column, row, head_index));

    bool rejected = false;
    try
    {
        (void)strict_2d_first_shell_wc_block_diagnostic("head-body");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_alpha_wc_diagnostic_requires_explicit_reference()
{
    assert(!librpa_int::strict_2d_alpha_wc_diagnostic_requested(nullptr));
    assert(!librpa_int::strict_2d_alpha_wc_diagnostic_requested(""));
    assert(librpa_int::strict_2d_alpha_wc_diagnostic_requested("0.25"));

    bool rejected_other_value = false;
    try
    {
        librpa_int::strict_2d_alpha_wc_diagnostic_requested("0.5");
    }
    catch (const std::invalid_argument &)
    {
        rejected_other_value = true;
    }
    assert(rejected_other_value);
}

void test_strict_2d_first_shell_analytic_wc_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::strict_2d_first_shell_analytic_wc_diagnostic_requested(nullptr));
    assert(!librpa_int::strict_2d_first_shell_analytic_wc_diagnostic_requested(""));
    assert(librpa_int::strict_2d_first_shell_analytic_wc_diagnostic_requested("enabled"));

    bool rejected = false;
    try
    {
        librpa_int::strict_2d_first_shell_analytic_wc_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_finite_q_matrix_dump_selection_is_read_only_and_bounded()
{
    assert(!librpa_int::strict_2d_should_dump_finite_q_matrix(0, 0, true));
    assert(librpa_int::strict_2d_should_dump_finite_q_matrix(1, 0, false));
    assert(librpa_int::strict_2d_should_dump_finite_q_matrix(12, 0, false));
    assert(!librpa_int::strict_2d_should_dump_finite_q_matrix(13, 0, false));
    assert(!librpa_int::strict_2d_should_dump_finite_q_matrix(1, 1, false));

    bool rejected = false;
    try
    {
        (void)librpa_int::strict_2d_should_dump_finite_q_matrix(1, 0, false, 0);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_omega0_diagnostic_directory_is_explicit_and_normalized()
{
    assert(librpa_int::strict_2d_omega0_diagnostic_directory(nullptr).empty());
    assert(librpa_int::strict_2d_omega0_diagnostic_directory("").empty());
    assert(librpa_int::strict_2d_omega0_diagnostic_directory("omega0") == "omega0/");
    assert(librpa_int::strict_2d_omega0_diagnostic_directory("omega0/") == "omega0/");
}

void test_strict_2d_omega0_override_basis_modes_are_mutually_exclusive()
{
    const auto none = librpa_int::strict_2d_omega0_override_directories(nullptr, nullptr);
    assert(none.coulomb_basis.empty());
    assert(none.auxiliary_basis.empty());

    const auto auxiliary = librpa_int::strict_2d_omega0_override_directories(nullptr, "omega0_aux");
    assert(auxiliary.coulomb_basis.empty());
    assert(auxiliary.auxiliary_basis == "omega0_aux/");

    bool rejected = false;
    try
    {
        (void)librpa_int::strict_2d_omega0_override_directories("omega0_coul", "omega0_aux");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_omega0_override_reader_validates_shape_and_payload()
{
    const std::string path = "strict2d_omega0_override_test.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const char magic[8] = {'L', 'R', '2', 'D', 'W', 'C', '0', '1'};
        const std::int32_t rows = 2, cols = 2;
        const double values[8] = {1.0, 0.5, -2.0, 3.0, 4.0, -1.0, 0.25, 0.0};
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        output.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        output.write(reinterpret_cast<const char *>(values), sizeof(values));
    }
    const auto matrix = librpa_int::read_strict_2d_omega0_override_binary(path, 2);
    assert(matrix.size() == 4);
    assert_complex_close(matrix[0], {1.0, 0.5}, 1.0e-15);
    assert_complex_close(matrix[1], {-2.0, 3.0}, 1.0e-15);
    assert_complex_close(matrix[2], {4.0, -1.0}, 1.0e-15);
    assert_complex_close(matrix[3], {0.25, 0.0}, 1.0e-15);

    bool rejected_shape = false;
    try
    {
        (void)librpa_int::read_strict_2d_omega0_override_binary(path, 3);
    }
    catch (const std::runtime_error &)
    {
        rejected_shape = true;
    }
    assert(rejected_shape);
    std::remove(path.c_str());
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
        for (int j = 1; j != 3; ++j) assert_complex_close(auxiliary_wc(i, j), pw_wc(i, j), 1e-13);
    }
}

void test_strict_2d_regular_coulomb_legs_are_projected_to_the_gamma_basis()
{
    constexpr double inverse_sqrt_two = 0.70710678118654752440;
    matrix_m<std::complex<double>> coulomb_sqrt(2, 2, MAJOR::COL);
    coulomb_sqrt(0, 0) = 4.0;
    coulomb_sqrt(1, 1) = 1.0;

    matrix_m<std::complex<double>> eigenvectors(2, 2, MAJOR::COL);
    eigenvectors(0, 0) = inverse_sqrt_two;
    eigenvectors(0, 1) = inverse_sqrt_two;
    eigenvectors(1, 0) = inverse_sqrt_two;
    eigenvectors(1, 1) = -inverse_sqrt_two;

    const auto projected =
        librpa_int::strict_2d_project_operator_to_coulomb_basis(coulomb_sqrt, eigenvectors);
    require_double_close(projected(0, 0).real(), 2.5, 1e-14);
    require_double_close(projected(0, 1).real(), 1.5, 1e-14);
    require_double_close(projected(1, 0).real(), 1.5, 1e-14);
    require_double_close(projected(1, 1).real(), 2.5, 1e-14);
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

struct Point2d
{
    double x;
    double y;
};

double dot(const Point2d &point, const Point2d &normal)
{
    return point.x * normal.x + point.y * normal.y;
}

double cross(const Point2d &left, const Point2d &right)
{
    return left.x * right.y - left.y * right.x;
}

std::vector<Point2d> clip_polygon_halfplane(const std::vector<Point2d> &polygon,
                                            const Point2d &normal, const double bound)
{
    std::vector<Point2d> clipped;
    if (polygon.empty()) return clipped;
    Point2d previous = polygon.back();
    double previous_distance = dot(previous, normal) - bound;
    for (const auto &current : polygon)
    {
        const double current_distance = dot(current, normal) - bound;
        const bool previous_inside = previous_distance <= 1e-14;
        const bool current_inside = current_distance <= 1e-14;
        if (previous_inside != current_inside)
        {
            const double denominator = previous_distance - current_distance;
            if (std::abs(denominator) < 1e-18) std::abort();
            const double fraction = previous_distance / denominator;
            clipped.push_back({previous.x + fraction * (current.x - previous.x),
                               previous.y + fraction * (current.y - previous.y)});
        }
        if (current_inside) clipped.push_back(current);
        previous = current;
        previous_distance = current_distance;
    }
    return clipped;
}

std::pair<double, Point2d> polygon_area_centroid(const std::vector<Point2d> &polygon)
{
    if (polygon.size() < 3) return {0.0, {0.0, 0.0}};
    double twice_area = 0.0;
    Point2d weighted{0.0, 0.0};
    for (std::size_t i = 0; i != polygon.size(); ++i)
    {
        const auto &left = polygon[i];
        const auto &right = polygon[(i + 1) % polygon.size()];
        const double edge_cross = cross(left, right);
        twice_area += edge_cross;
        weighted.x += (left.x + right.x) * edge_cross;
        weighted.y += (left.y + right.y) * edge_cross;
    }
    if (!(twice_area > 0.0)) std::abort();
    return {0.5 * twice_area, {weighted.x / (3.0 * twice_area), weighted.y / (3.0 * twice_area)}};
}

std::vector<Point2d> gamma_cell_neighbors(const Point2d &g1, const Point2d &g2)
{
    std::vector<Point2d> neighbors;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            if (i != 0 || j != 0) neighbors.push_back({i * g1.x + j * g2.x, i * g1.y + j * g2.y});
    return neighbors;
}

std::vector<Point2d> gamma_voronoi_polygon(const Point2d &g1, const Point2d &g2)
{
    const double extent = 4.0 * std::max(std::hypot(g1.x, g1.y), std::hypot(g2.x, g2.y));
    std::vector<Point2d> polygon{
        {-extent, -extent}, {extent, -extent}, {extent, extent}, {-extent, extent}};
    for (const auto &neighbor : gamma_cell_neighbors(g1, g2))
        polygon = clip_polygon_halfplane(polygon, neighbor, 0.5 * dot(neighbor, neighbor));
    return polygon;
}

double gamma_cell_boundary(const Point2d &direction, const std::vector<Point2d> &neighbors)
{
    double qmax = std::numeric_limits<double>::infinity();
    for (const auto &neighbor : neighbors)
    {
        const double denominator = dot(direction, neighbor);
        if (denominator > 1e-14) qmax = std::min(qmax, 0.5 * dot(neighbor, neighbor) / denominator);
    }
    if (!(qmax > 0.0) || !std::isfinite(qmax)) std::abort();
    return qmax;
}

matrix_m<std::complex<double>> cartesian_gamma_subgrid_average(
    const matrix_m<std::complex<double>> &body_inv, const matrix_m<std::complex<double>> &bw_cart,
    const matrix_m<std::complex<double>> &wb_cart, const matrix_m<std::complex<double>> &lind,
    const matrix_m<std::complex<double>> &regular_body_sqrt, const std::vector<Point2d> &polygon,
    const int subdivisions, double &covered_area)
{
    if (subdivisions < 2) std::abort();
    double xmin = polygon.front().x, xmax = polygon.front().x;
    double ymin = polygon.front().y, ymax = polygon.front().y;
    for (const auto &point : polygon)
    {
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }
    const double dx = (xmax - xmin) / subdivisions;
    const double dy = (ymax - ymin) / subdivisions;
    const double gamma_area = polygon_area_centroid(polygon).first;
    matrix_m<std::complex<double>> average(body_inv.nr() + 1, body_inv.nc() + 1, MAJOR::COL);
    covered_area = 0.0;

    for (int ix = 0; ix != subdivisions; ++ix)
        for (int iy = 0; iy != subdivisions; ++iy)
        {
            const double xlo = xmin + ix * dx;
            const double xhi = xlo + dx;
            const double ylo = ymin + iy * dy;
            const double yhi = ylo + dy;
            auto cell = clip_polygon_halfplane(polygon, {1.0, 0.0}, xhi);
            cell = clip_polygon_halfplane(cell, {-1.0, 0.0}, -xlo);
            cell = clip_polygon_halfplane(cell, {0.0, 1.0}, yhi);
            cell = clip_polygon_halfplane(cell, {0.0, -1.0}, -ylo);
            const auto area_centroid = polygon_area_centroid(cell);
            const double area = area_centroid.first;
            const auto centroid = area_centroid.second;
            if (area == 0.0) continue;
            const double q = std::hypot(centroid.x, centroid.y);
            if (!(q > 1e-14)) std::abort();
            const double qx = centroid.x / q;
            const double qy = centroid.y / q;
            matrix_m<std::complex<double>> bw_direction(body_inv.nr(), 1, MAJOR::COL);
            matrix_m<std::complex<double>> wb_direction(1, body_inv.nc(), MAJOR::COL);
            for (int i = 0; i != body_inv.nr(); ++i)
            {
                bw_direction(i, 0) = bw_cart(i, 0) * qx + bw_cart(i, 1) * qy;
                wb_direction(0, i) = wb_cart(0, i) * qx + wb_cart(1, i) * qy;
            }
            const auto a = librpa_int::strict_2d_schur_coefficient(lind, qx, qy);
            const auto point = librpa_int::strict_2d_wc_blocks_at_q(
                body_inv, bw_direction, wb_direction, a, regular_body_sqrt, q);
            for (int i = 0; i != average.nr(); ++i)
                for (int j = 0; j != average.nc(); ++j)
                    average(i, j) += area * point(i, j) / gamma_area;
            covered_area += area;
        }
    return average;
}

double submatrix_frobenius(const matrix_m<std::complex<double>> &matrix, const int row_start,
                           const int column_start)
{
    double squared = 0.0;
    for (int i = row_start; i != matrix.nr(); ++i)
        for (int j = column_start; j != matrix.nc(); ++j) squared += std::norm(matrix(i, j));
    return std::sqrt(squared);
}

double matrix_hermiticity_residual(const matrix_m<std::complex<double>> &matrix)
{
    double residual = 0.0;
    for (int i = 0; i != matrix.nr(); ++i)
        for (int j = 0; j != matrix.nc(); ++j)
            residual = std::max(residual, std::abs(matrix(i, j) - std::conj(matrix(j, i))));
    return residual;
}

double wing_frobenius(const matrix_m<std::complex<double>> &matrix, const bool head_body)
{
    double squared = 0.0;
    for (int i = 1; i != matrix.nr(); ++i)
        squared += head_body ? std::norm(matrix(0, i)) : std::norm(matrix(i, 0));
    return std::sqrt(squared);
}

void test_strict_2d_wc_cell_average_matches_cartesian_voronoi_subgrid()
{
    matrix_m<std::complex<double>> body_inv(2, 2, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(2, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 2, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> regular_body_sqrt(2, 2, MAJOR::COL);
    body_inv(0, 0) = 0.72;
    body_inv(0, 1) = {0.03, 0.01};
    body_inv(1, 0) = std::conj(body_inv(0, 1));
    body_inv(1, 1) = 0.81;
    bw_cart(0, 0) = {0.18, 0.03};
    bw_cart(0, 1) = {-0.07, 0.02};
    bw_cart(1, 0) = {0.09, -0.01};
    bw_cart(1, 1) = {0.11, 0.04};
    wb_cart = bw_cart.get_transpose(true);
    lind(0, 0) = 1.55;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.25;
    lind(2, 2) = 1.0;
    regular_body_sqrt(0, 0) = 1.4;
    regular_body_sqrt(0, 1) = 0.05;
    regular_body_sqrt(1, 0) = 0.05;
    regular_body_sqrt(1, 1) = 1.1;

    const Point2d g1{0.08702149160639744, 0.05024962446042246};
    const Point2d g2{0.0, 0.1004992489208449};
    const auto neighbors = gamma_cell_neighbors(g1, g2);
    const auto polygon = gamma_voronoi_polygon(g1, g2);
    const double gamma_area = polygon_area_centroid(polygon).first;
    require_double_close(gamma_area, std::abs(cross(g1, g2)), 1e-14);

    constexpr int nangle = 5000;
    std::vector<double> qx(nangle), qy(nangle), weights(nangle), qmax(nangle);
    for (int i = 0; i != nangle; ++i)
    {
        const double angle = librpa_int::TWO_PI * i / nangle;
        qx[i] = std::cos(angle);
        qy[i] = std::sin(angle);
        weights[i] = librpa_int::TWO_PI / nangle;
        qmax[i] = gamma_cell_boundary({qx[i], qy[i]}, neighbors);
    }
    const auto analytic = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, regular_body_sqrt, qx, qy, weights, qmax, gamma_area);

    double covered_area = 0.0;
    const auto grid80 = cartesian_gamma_subgrid_average(
        body_inv, bw_cart, wb_cart, lind, regular_body_sqrt, polygon, 80, covered_area);
    require_double_close(covered_area, gamma_area, 1e-14);
    if (matrix_hermiticity_residual(analytic) >= 1e-12 ||
        matrix_hermiticity_residual(grid80) >= 1e-12)
        std::abort();

    const double head_relative = std::abs(grid80(0, 0) - analytic(0, 0)) / std::abs(analytic(0, 0));
    const double body_relative =
        submatrix_frobenius(grid80 - analytic, 1, 1) / submatrix_frobenius(analytic, 1, 1);
    const double wing_absolute =
        std::max(wing_frobenius(grid80, true), wing_frobenius(grid80, false));
    if (head_relative >= 1e-4 || body_relative >= 1e-4 || wing_absolute >= 1e-10) std::abort();
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

void test_rpa_chi0v_wing_desc_matches_producer_layout(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(10, 10, 0, 0);

    ArrayDesc desc_full_wing_seed(blacs_h);
    desc_full_wing_seed.init_square_blk(11, 3, 0, 0);
    ArrayDesc desc_full_wing(blacs_h);
    desc_full_wing.init(11, 3, desc_body.mb(), desc_full_wing_seed.nb(), 0, 0);

    const auto desc_wing = librpa_int::make_rpa_chi0v_wing_desc(
        desc_body, 1, desc_full_wing.m_loc(), desc_full_wing.n_loc());

    if (desc_body.nprows() > 1)
    {
        assert(desc_full_wing.m_loc() < desc_full_wing.m());
    }
    assert(desc_wing.m() == 11);
    assert(desc_wing.n() == 3);
    assert(desc_wing.mb() == desc_body.mb());
    assert(desc_wing.nb() == desc_full_wing.nb());
    assert(desc_wing.m_loc() == desc_full_wing.m_loc());
    assert(desc_wing.n_loc() == desc_full_wing.n_loc());
    assert(std::equal(desc_wing.desc, desc_wing.desc + 9, desc_full_wing.desc));
    assert(desc_wing.l2g_r() == desc_full_wing.l2g_r());
    assert(desc_wing.l2g_c() == desc_full_wing.l2g_c());
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

SymmetryKStarMember make_headwing_wfc_atom_swap_member(const std::complex<double> &rot_0,
                                                       const std::complex<double> &rot_1)
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
    const auto member = make_headwing_wfc_atom_swap_member({2.0, 0.5}, {3.0, -0.25});

    ComplexMatrix wfc_ibz(1, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), wfc_ibz(0, 1) * std::complex<double>{3.0, -0.25}, 1e-12);
    assert_complex_close(wfc_bz(0, 1), wfc_ibz(0, 0) * std::complex<double>{2.0, 0.5}, 1e-12);
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
    operation.rotation = Matrix3(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    ctx.rspace_operations = {operation};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    std::array<ComplexMatrix, 3> velocity_ibz;
    for (auto &component : velocity_ibz) component.create(1, 1);
    velocity_ibz[0](0, 0) = {1.0, 2.0};
    velocity_ibz[1](0, 0) = {3.0, -1.0};
    velocity_ibz[2](0, 0) = {-0.5, 0.25};

    const auto velocity_bz =
        librpa_int::rotate_headwing_velocity_to_kstar_member(ctx, member, velocity_ibz, 1, false);
    assert_complex_close(velocity_bz[0](0, 0), -velocity_ibz[1](0, 0), 1e-12);
    assert_complex_close(velocity_bz[1](0, 0), velocity_ibz[0](0, 0), 1e-12);
    assert_complex_close(velocity_bz[2](0, 0), velocity_ibz[2](0, 0), 1e-12);

    member.time_reversal = true;
    const auto velocity_bz_tr =
        librpa_int::rotate_headwing_velocity_to_kstar_member(ctx, member, velocity_ibz, 1, true);
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
    const auto &wfc =
        librpa_int::direct_full_bz_wfc_for_kstar_member(wfc_full, member_source_ik, 0, 0, 0, 0);
    assert_complex_close(wfc(0, 0), {0.0, 1.0}, 1e-12);
}

void test_weighted_wfc_gram_comparison_is_phase_invariant_and_detects_band_swap()
{
    ComplexMatrix direct(2, 2);
    direct(0, 0) = {1.0, 0.0};
    direct(0, 1) = {0.0, 0.0};
    direct(1, 0) = {0.0, 0.0};
    direct(1, 1) = {1.0, 0.0};

    auto phased = direct;
    phased(0, 0) = {0.0, 1.0};
    phased(1, 1) = {-1.0, 0.0};
    const auto phase_metrics =
        librpa_int::compare_weighted_wfc_grams(direct, phased, {1.0, 2.0});
    assert(phase_metrics.difference_frobenius < 1e-12);
    assert(phase_metrics.relative_frobenius < 1e-12);
    assert(phase_metrics.maximum_absolute_difference < 1e-12);

    ComplexMatrix swapped(2, 2);
    swapped(0, 1) = {1.0, 0.0};
    swapped(1, 0) = {1.0, 0.0};
    const auto swap_metrics =
        librpa_int::compare_weighted_wfc_grams(direct, swapped, {1.0, 2.0});
    assert(swap_metrics.difference_frobenius > 1.0);
    assert(swap_metrics.relative_frobenius > 0.1);
}

void test_gw_gf_kstar_wfc_diagnostic_requires_explicit_enable()
{
    assert(!librpa_int::output_gw_gf_kstar_wfc_diagnostic_requested(nullptr));
    assert(!librpa_int::output_gw_gf_kstar_wfc_diagnostic_requested(""));
    assert(librpa_int::output_gw_gf_kstar_wfc_diagnostic_requested("enabled"));
    bool rejected = false;
    try
    {
        (void)librpa_int::output_gw_gf_kstar_wfc_diagnostic_requested("true");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);

    assert(!librpa_int::stop_after_gw_gf_kstar_wfc_diagnostic_requested(nullptr));
    assert(!librpa_int::stop_after_gw_gf_kstar_wfc_diagnostic_requested(""));
    assert(librpa_int::stop_after_gw_gf_kstar_wfc_diagnostic_requested("enabled"));
    rejected = false;
    try
    {
        (void)librpa_int::stop_after_gw_gf_kstar_wfc_diagnostic_requested("yes");
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_gw_gf_kstar_wfc_diagnostic_uses_meanfield_kpoint_count()
{
    assert(librpa_int::gw_gf_kstar_wfc_diagnostic_active_kpoints(19, 19, 19) == 19);
    for (const auto counts : {std::array<int, 3>{0, 19, 19},
                              std::array<int, 3>{19, 18, 19},
                              std::array<int, 3>{19, 19, 18}})
    {
        bool rejected = false;
        try
        {
            (void)librpa_int::gw_gf_kstar_wfc_diagnostic_active_kpoints(
                counts[0], counts[1], counts[2]);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);
    }
}

RI::Tensor<double> make_single_value_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void compare_local_blacs_matrices(
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &actual,
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &expected, const double tolerance)
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

    diele_func df_ibz(mf_ibz, velocity, kfrac_ibz, basis_wfc, basis_abf, omega, 2, 2, 1, 1, pbc,
                      librpa_int::global::mpi_comm_global_h, blacs_h);
    diele_func df_full(mf_full, velocity, {member.k_bz}, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                       pbc, librpa_int::global::mpi_comm_global_h, blacs_h);

    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = make_single_value_tensor(1.0);
    Cs_IJ[0][{1, {0, 0, 0}}] = make_single_value_tensor(-0.35);
    Cs_IJ[0][{1, {1, 0, 0}}] = make_single_value_tensor(0.42);

    std::vector<std::vector<const ComplexMatrix *>> restored_wfc_ptrs(
        1, std::vector<const ComplexMatrix *>(1, &wfc_bz));
    const auto restored =
        df_ibz.transform_Cs2mnk_kblacs(0, 0, Cs_IJ, blacs_h, member.k_bz, &restored_wfc_ptrs);
    const auto full = df_full.transform_Cs2mnk_kblacs(0, 0, Cs_IJ, blacs_h, member.k_bz);

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
            spin_channels_differ = spin_channels_differ ||
                                   std::abs(spin_up.second(i, j) - spin_dn.second(i, j)) > 1e-12;
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
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old &wq,
    const atom_t atom_i, const atom_t atom_j, const Vector3_Order<double> &q,
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
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old &wq,
    const Vector3_Order<double> &q, const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, block] : row)
        {
            add_scalar_wq_block(wq, atom_i, atom_j, q, block(0, 0));
        }
    }
}

void add_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old &wq,
    const Vector3_Order<double> &q, const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, source] : row)
        {
            auto &target = wq[atom_i][atom_j][q];
            target = matrix_m<std::complex<double>>(source.nr, source.nc, MAJOR::ROW);
            for (int i = 0; i != source.nr; ++i)
                for (int j = 0; j != source.nc; ++j) target(i, j) = source(i, j);
        }
    }
}

librpa_int::symmetry_atom_block_matrix_map_t make_bn_hermitian_wq_blocks(
    const std::vector<int> &n_by_atom, const double index)
{
    const int n = std::accumulate(n_by_atom.begin(), n_by_atom.end(), 0);
    ComplexMatrix dense(n, n);
    for (int i = 0; i != n; ++i)
    {
        dense(i, i) = {1.0 + 0.09 * index + 0.04 * (i + 1), 0.0};
        for (int j = i + 1; j != n; ++j)
        {
            dense(i, j) = {0.023 * (i + 1) * (j + 2) + 0.011 * index,
                           0.017 * (j - i) + 0.007 * index};
            dense(j, i) = std::conj(dense(i, j));
        }
    }

    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    std::vector<int> offsets{0};
    for (const int n_atom : n_by_atom) offsets.push_back(offsets.back() + n_atom);
    for (atom_t atom_i = 0; atom_i != 2; ++atom_i)
    {
        for (atom_t atom_j = 0; atom_j != 2; ++atom_j)
        {
            auto &block = blocks[atom_i][atom_j];
            block.create(n_by_atom.at(atom_i), n_by_atom.at(atom_j));
            for (int i = 0; i != n_by_atom.at(atom_i); ++i)
                for (int j = 0; j != n_by_atom.at(atom_j); ++j)
                    block(i, j) = dense(offsets.at(atom_i) + i, offsets.at(atom_j) + j);
        }
    }
    return blocks;
}

std::vector<std::vector<int>> make_bn_test_shells(const int basis_case)
{
    if (basis_case == 0) return {{0}, {0}};
    if (basis_case == 1) return {{0, 1}, {0, 1}};
    if (basis_case == 2)
        return {{0, 0, 0, 1, 1, 1, 2, 2}, {0, 0, 0, 1, 1, 1, 2, 2}};
    if (basis_case == 3)
        return {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3, 4, 4, 4},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                 4, 4, 4}};
    throw std::runtime_error("unknown BN basis test case");
}

std::vector<int> basis_counts_from_shells(const std::vector<std::vector<int>> &shells)
{
    std::vector<int> counts;
    for (const auto &atom_shells : shells)
    {
        int count = 0;
        for (const int l : atom_shells) count += 2 * l + 1;
        counts.push_back(count);
    }
    return counts;
}

PeriodicBoundaryData make_wq_full_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0, librpa_int::TWO_PI / 3.0, 0.0, 0.0, librpa_int::TWO_PI * 2.0 / 3.0,
        0.0, 0.0};
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_wq_reduced_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{0.0, 0.0, 0.0, librpa_int::TWO_PI / 3.0, 0.0, 0.0};
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}}, {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}};
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);
    return pbc;
}

void test_spacetime_fourier_phases_form_k_minus_q_convolution()
{
    constexpr int mesh = 3;
    const std::array<std::complex<double>, mesh> wc_q{
        std::complex<double>{1.2, -0.3},
        std::complex<double>{-0.4, 0.7},
        std::complex<double>{0.9, 0.2}};
    // These values already contain the 1/Nk weight carried by MeanField's Green function.
    const std::array<std::complex<double>, mesh> weighted_green_k{
        std::complex<double>{0.13, 0.04},
        std::complex<double>{-0.08, 0.02},
        std::complex<double>{0.05, -0.06}};

    std::array<std::complex<double>, mesh> wc_R{};
    std::array<std::complex<double>, mesh> green_R{};
    for (int iR = 0; iR != mesh; ++iR)
    {
        for (int iq = 0; iq != mesh; ++iq)
        {
            const double angle = -librpa_int::TWO_PI * static_cast<double>(iq * iR) / mesh;
            const std::complex<double> phase{std::cos(angle), std::sin(angle)};
            wc_R[iR] += wc_q[iq] * phase / static_cast<double>(mesh);
            green_R[iR] += weighted_green_k[iq] * phase;
        }
    }

    for (int ik = 0; ik != mesh; ++ik)
    {
        std::complex<double> spacetime_sigma_k{};
        std::complex<double> direct_k_minus_q{};
        std::complex<double> direct_k_plus_q{};
        for (int iR = 0; iR != mesh; ++iR)
        {
            const double angle = librpa_int::TWO_PI * static_cast<double>(ik * iR) / mesh;
            const std::complex<double> phase{std::cos(angle), std::sin(angle)};
            spacetime_sigma_k += wc_R[iR] * green_R[iR] * phase;
        }
        for (int iq = 0; iq != mesh; ++iq)
        {
            direct_k_minus_q += wc_q[iq] * weighted_green_k[(ik - iq + mesh) % mesh];
            direct_k_plus_q += wc_q[iq] * weighted_green_k[(ik + iq) % mesh];
        }
        assert_complex_close(spacetime_sigma_k, direct_k_minus_q, 1.0e-13);
        if (std::abs(spacetime_sigma_k - direct_k_plus_q) < 1.0e-5)
            throw std::runtime_error("synthetic data do not distinguish k-q from k+q");
    }
}

SymmetryContext make_two_atom_inversion_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    ctx.set_crystal_structure(lattice, lattice, {{0, 0}, {1, 0}},
                              {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};

    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    inversion.translation = {0.0, 0.0, 0.0};

    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations(
        {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M}, 0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

std::vector<SymmetryOperation> make_bn_hexagonal_operations()
{
    const std::array<std::array<int, 9>, 12> rotations{{
        {{1, 0, 0, 0, 1, 0, 0, 0, 1}},
        {{0, 1, 0, -1, -1, 0, 0, 0, -1}},
        {{-1, -1, 0, 1, 0, 0, 0, 0, 1}},
        {{1, 0, 0, 0, 1, 0, 0, 0, -1}},
        {{0, 1, 0, -1, -1, 0, 0, 0, 1}},
        {{-1, -1, 0, 1, 0, 0, 0, 0, -1}},
        {{-1, 0, 0, 1, 1, 0, 0, 0, 1}},
        {{1, 1, 0, 0, -1, 0, 0, 0, -1}},
        {{0, -1, 0, -1, 0, 0, 0, 0, 1}},
        {{-1, 0, 0, 1, 1, 0, 0, 0, -1}},
        {{1, 1, 0, 0, -1, 0, 0, 0, 1}},
        {{0, -1, 0, -1, 0, 0, 0, 0, -1}},
    }};
    std::vector<SymmetryOperation> operations;
    operations.reserve(rotations.size());
    for (const auto &rotation : rotations)
    {
        SymmetryOperation operation;
        operation.rotation =
            Matrix3(rotation[0], rotation[1], rotation[2], rotation[3], rotation[4], rotation[5],
                    rotation[6], rotation[7], rotation[8]);
        operation.translation = {0.0, 0.0, 0.0};
        operation.use_row_convention = true;
        operations.push_back(operation);
    }
    return operations;
}

void set_bn_hexagonal_lattice(PeriodicBoundaryData &pbc)
{
    const double sqrt_three = std::sqrt(3.0);
    pbc.set_latvec({0.5, -0.5 * sqrt_three, 0.0, 0.5, 0.5 * sqrt_three, 0.0, 0.0, 0.0, 8.0});
}

SymmetryContext make_bn_hexagonal_context(const PeriodicBoundaryData &pbc, const int max_l = 0)
{
    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec, pbc.G, {{0, 0}, {1, 1}},
                              {{0, {1.0 / 3.0, 2.0 / 3.0, 0.5}}, {1, {2.0 / 3.0, 1.0 / 3.0, 0.5}}});
    ctx.set_rspace_operations(make_bn_hexagonal_operations());
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations(
        {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M},
        max_l);
    ctx.build_kstar_member_rotations(max_l);
    return ctx;
}

PeriodicBoundaryData make_bn_hexagonal_full_pbc(const int mesh = 3)
{
    PeriodicBoundaryData pbc;
    set_bn_hexagonal_lattice(pbc);
    std::vector<double> kvecs;
    for (const auto &kfrac : librpa_int::build_uniform_kmesh_frac({mesh, mesh, 1}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.push_back(librpa_int::TWO_PI * kvec.x);
        kvecs.push_back(librpa_int::TWO_PI * kvec.y);
        kvecs.push_back(librpa_int::TWO_PI * kvec.z);
    }
    pbc.set_kgrids_kvec(mesh, mesh, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_bn_hexagonal_reduced_pbc(const SymmetryContext &ctx, const int mesh = 3)
{
    PeriodicBoundaryData pbc;
    set_bn_hexagonal_lattice(pbc);
    std::vector<double> kvecs_ibz;
    std::vector<std::vector<Vector3_Order<double>>> full_kstars;
    for (const auto &star : ctx.kstars)
    {
        const auto kvec = librpa_int::restrict_fractional_coordinate(star.k_ibz) * pbc.G;
        kvecs_ibz.push_back(librpa_int::TWO_PI * kvec.x);
        kvecs_ibz.push_back(librpa_int::TWO_PI * kvec.y);
        kvecs_ibz.push_back(librpa_int::TWO_PI * kvec.z);
        full_kstars.emplace_back();
        for (const auto &member : star.members)
        {
            full_kstars.back().push_back(librpa_int::restrict_fractional_coordinate(member.k_bz) *
                                         pbc.G);
        }
    }
    pbc.set_irreducible_kgrids_kvec(mesh, mesh, 1, kvecs_ibz, full_kstars);
    return pbc;
}

std::size_t find_fractional_kpoint_index(const std::vector<Vector3_Order<double>> &kpoints,
                                         const Vector3_Order<double> &target)
{
    for (std::size_t ik = 0; ik != kpoints.size(); ++ik)
    {
        if (librpa_int::same_fractional_kpoint(kpoints[ik], target, 1e-8))
        {
            return ik;
        }
    }
    throw std::runtime_error("failed to find a full-grid fractional k-point in the test");
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
                if (actual_block.nr() != expected_block.nr() ||
                    actual_block.nc() != expected_block.nc())
                    throw std::runtime_error("Wc(R) atom block dimensions differ");
                for (int i = 0; i != expected_block.nr(); ++i)
                {
                    for (int j = 0; j != expected_block.nc(); ++j)
                    {
                        if (std::abs(actual_block(i, j) - expected_block(i, j)) >= 1e-12)
                        {
                            std::cerr << "atom_pair=(" << atom_i << "," << atom_j << ") R=("
                                      << R.x << "," << R.y << "," << R.z << ") block=(" << i
                                      << "," << j << ")" << std::endl;
                        }
                        assert_complex_close(actual_block(i, j), expected_block(i, j), 1e-12);
                    }
                }
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
    const auto gamma_blocks = scalar_wq_to_blocks(
        {{0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}}, {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks(
        {{0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}}, {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
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
    const auto inversion_minus_blocks = scalar_wq_to_blocks(
        {{0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}}, {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry,
                              wq_full, dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
                              dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

void test_wq_to_wr_qmember_diagnostic_keeps_original_full_bz_weight()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);
    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    const auto gamma_blocks = scalar_wq_to_blocks(
        {{0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}}, {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks(
        {{0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}}, {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
    const double collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
        for (auto &[atom_j, q_blocks] : row)
            for (auto &[q, block] : q_blocks) block *= collective_scale;

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_selected_full;
    add_scalar_wq_blocks(wq_selected_full, pbc_full.klist.at(1), rep_blocks);
    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf,
                                                no_symmetry, wq_selected_full, dummy_tfg, pbc_full,
                                                pbc_full.Rlist, false, "", false);

    setenv("LIBRPA_STRICT2D_QMEMBER_DIAG", "0.3333333333333333,0,0", 1);
    const auto actual =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
                              dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);
    unsetenv("LIBRPA_STRICT2D_QMEMBER_DIAG");
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
        const auto gamma_blocks = scalar_wq_to_blocks(
            {{0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}}, {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks(
            {{0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}}, {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks(
            {{0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}}, {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry,
                              wq_full, dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
                              dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(actual.empty());
    }
}

void test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(const int mesh,
                                                               const int basis_case)
{
    const auto pbc_full = make_bn_hexagonal_full_pbc(mesh);
    const auto ctx_full = make_bn_hexagonal_context(pbc_full);
    const auto pbc_sym = make_bn_hexagonal_reduced_pbc(ctx_full, mesh);
    const auto shells = make_bn_test_shells(basis_case);
    const auto counts = basis_counts_from_shells(shells);
    const int max_l = *std::max_element(shells.front().begin(), shells.front().end());
    auto ctx = make_bn_hexagonal_context(pbc_sym, max_l);

    assert(ctx.count_kstar_members() == static_cast<std::size_t>(mesh * mesh));

    AtomicBasis basis_abf(
        std::vector<std::size_t>{static_cast<std::size_t>(counts.at(0)),
                                 static_cast<std::size_t>(counts.at(1))});
    basis_abf.set_l_shells(shells);
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{
        {0, static_cast<std::size_t>(counts.at(0))},
        {1, static_cast<std::size_t>(counts.at(1))}};
    const std::set<std::pair<atom_t, atom_t>> target_pairs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    const auto full_targets =
        librpa_int::build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, pbc_sym.kfrac_list);
    const bool use_full_targets = full_targets.size() == ctx.kstars.size();

    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto &star = ctx.kstars[istar];
        const auto mapping_iter =
            std::find_if(ctx.kstar_grid_mapping.begin(), ctx.kstar_grid_mapping.end(),
                         [istar](const librpa_int::SymmetryKStarGridMappingEntry &entry)
                         { return entry.star_list_index == static_cast<int>(istar); });
        assert(mapping_iter != ctx.kstar_grid_mapping.end());
        const double index = static_cast<double>(istar + 1);
        const auto blocks_ibz = make_bn_hermitian_wq_blocks(counts, index);
        add_wq_blocks(wq_sym, pbc_sym.klist.at(istar), blocks_ibz);

        const auto closure = librpa_int::build_symmetry_upper_atom_pair_closure(star, target_pairs);
        const auto symmetrized = librpa_int::symmetrize_symmetry_ibz_kspace_operator_blocks(
            ctx, layouts, star.k_ibz, blocks_ibz, atom_nabf, &closure);
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto target = librpa_int::restrict_fractional_coordinate(
                use_full_targets
                    ? full_targets.at(istar).at(imember)
                    : Vector3_Order<double>{pbc_sym.latvec *
                                            mapping_iter->member_q_bz_keys.at(imember)});
            const auto rotated = librpa_int::rotate_symmetry_kspace_operator_blocks(
                ctx, layouts, star.members[imember], symmetrized, atom_nabf, star.k_ibz,
                star.members[imember].time_reversal, &target_pairs, &target);
            const auto ifull = find_fractional_kpoint_index(pbc_full.kfrac_list, target);
            add_wq_blocks(wq_full, pbc_full.klist.at(ifull), rotated);
        }
    }

    const double collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
        for (auto &[atom_j, q_blocks] : row)
            for (auto &[q, block] : q_blocks) block *= collective_scale;

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry,
                              wq_full, dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
                              dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);
    assert_wq_rspace_maps_close(actual, expected);
}

void test_bn_qstar_wq_to_wr_matches_explicit_full_bz_on_odd_and_even_meshes()
{
    for (const int basis_case : {0, 1, 3})
    {
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(3, basis_case);
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(4, basis_case);
    }
}

ComplexMatrix restore_bn_test_wfc_to_member(const SymmetryContext &ctx,
                                            const std::vector<SpeciesBasisLayout> &layouts,
                                            const std::map<atom_t, size_t> &atom_nw,
                                            const librpa_int::SymmetryKStar &star,
                                            const SymmetryKStarMember &member,
                                            const ComplexMatrix &wfc_ibz)
{
    const auto rotation = librpa_int::build_symmetry_kspace_rotation_matrix(
        ctx, layouts, member, atom_nw, star.k_ibz, member.time_reversal, &member.k_bz);
    return member.time_reversal ? librpa_int::conj(wfc_ibz) * librpa_int::conj(rotation)
                                : wfc_ibz * rotation;
}

void test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(const int mesh,
                                                                     const int basis_case)
{
    const auto pbc_full = make_bn_hexagonal_full_pbc(mesh);
    const auto ctx_full = make_bn_hexagonal_context(pbc_full);
    const auto pbc_sym = make_bn_hexagonal_reduced_pbc(ctx_full, mesh);
    const auto shells = make_bn_test_shells(basis_case);
    const auto counts = basis_counts_from_shells(shells);
    const int max_l = *std::max_element(shells.front().begin(), shells.front().end());
    auto ctx = make_bn_hexagonal_context(pbc_sym, max_l);

    const int n_aos = std::accumulate(counts.begin(), counts.end(), 0);
    AtomicBasis basis_wfc(
        std::vector<std::size_t>{static_cast<std::size_t>(counts.at(0)),
                                 static_cast<std::size_t>(counts.at(1))});
    basis_wfc.set_l_shells(shells);
    const auto layouts = basis_wfc.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nw{
        {0, static_cast<std::size_t>(counts.at(0))},
        {1, static_cast<std::size_t>(counts.at(1))}};
    const int n_full_kpoints = mesh * mesh;

    MeanField mf_ibz(1, static_cast<int>(ctx.kstars.size()), n_aos, n_aos);
    MeanField mf_full(1, n_full_kpoints, n_aos, n_aos);
    mf_ibz.get_efermi() = 0.0;
    mf_full.get_efermi() = 0.0;
    std::vector<int> full_owner(static_cast<std::size_t>(n_full_kpoints), -1);

    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto &star = ctx.kstars[istar];
        const double star_weight =
            static_cast<double>(star.members.size()) / static_cast<double>(n_full_kpoints);
        for (int iband = 0; iband != n_aos; ++iband)
        {
            const bool occupied = iband < n_aos / 2;
            mf_ibz.get_eigenvals()[0](static_cast<int>(istar), iband) =
                (occupied ? -0.8 - 0.13 * iband : 0.6 + 0.17 * (iband - n_aos / 2)) +
                0.03 * static_cast<double>(istar);
            mf_ibz.get_weight()[0](static_cast<int>(istar), iband) =
                occupied ? 2.0 * star_weight : 0.0;
        }
        auto &wfc_ibz = mf_ibz.get_eigenvectors()[0][0][istar];
        wfc_ibz.create(n_aos, n_aos);
        for (int iband = 0; iband != n_aos; ++iband)
            for (int iao = 0; iao != n_aos; ++iao)
                wfc_ibz(iband, iao) =
                    std::complex<double>{
                        0.11 * (iband + 1) + 0.07 * (iao + 1),
                        0.013 * static_cast<double>((iband + 1) * (iao + 2)) +
                            0.009 * static_cast<double>(istar + 1)} /
                    static_cast<double>(n_aos);

        for (const auto &member : star.members)
        {
            const auto ifull = find_fractional_kpoint_index(
                pbc_full.kfrac_list, librpa_int::restrict_fractional_coordinate(member.k_bz));
            assert(full_owner[ifull] < 0);
            full_owner[ifull] = static_cast<int>(istar);
            for (int iband = 0; iband != n_aos; ++iband)
            {
                mf_full.get_eigenvals()[0](static_cast<int>(ifull), iband) =
                    mf_ibz.get_eigenvals()[0](static_cast<int>(istar), iband);
                mf_full.get_weight()[0](static_cast<int>(ifull), iband) =
                    iband < n_aos / 2 ? 2.0 / static_cast<double>(n_full_kpoints) : 0.0;
            }
            mf_full.get_eigenvectors()[0][0][ifull] =
                restore_bn_test_wfc_to_member(ctx, layouts, atom_nw, star, member, wfc_ibz);
        }
    }
    assert(std::find(full_owner.begin(), full_owner.end(), -1) == full_owner.end());

    const std::vector<double> taus{-0.35, 0.27};
    const auto &Rs = pbc_full.Rlist;
    const auto expected = mf_full.get_gf_cplx_imagtimes_Rs(0, 0, 0, pbc_full.kfrac_list, taus, Rs);
    const auto actual = librpa_int::get_symmetry_restored_gf_cplx_imagtimes_Rs(
        ctx, layouts, mf_ibz, 0, 0, 0, pbc_sym.kfrac_list, taus, Rs, atom_nw);

    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            const auto diff = actual.at(tau).at(R) - expected.at(tau).at(R);
            double max_abs = 0.0;
            for (int i = 0; i != diff.nr; ++i)
                for (int j = 0; j != diff.nc; ++j)
                    max_abs = std::max(max_abs, std::abs(diff(i, j)));
            if (max_abs >= 1e-12)
            {
                std::cerr << "BN k-star GF mismatch: mesh=" << mesh << " tau=" << tau << " R=" << R
                          << " basis_case=" << basis_case << " max_abs=" << max_abs << std::endl;
                throw std::runtime_error("BN k-star Green function differs from explicit full BZ");
            }
        }
    }
}

void test_bn_kstar_green_function_matches_explicit_full_bz_on_odd_and_even_meshes()
{
    for (const int basis_case : {0, 1, 2})
    {
        test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(3, basis_case);
        test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(4, basis_case);
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
            mat(i_local, j_local) =
                blocks.at(static_cast<atom_t>(atom_i)).at(static_cast<atom_t>(atom_j))(0, 0);
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
                        std::cerr << "  (" << i << "," << j
                                  << ") actual=" << actual.at(freq).at(R)(i, j)
                                  << " expected=" << expected_mat(i, j) << " diff=" << diff(i, j)
                                  << std::endl;
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

    const auto gamma_blocks = scalar_wq_to_blocks(
        {{0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}}, {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks(
        {{0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}}, {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});

    constexpr double freq = 0.25;
    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_sym;
    wq_sym[freq][pbc_sym.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_sym[freq][pbc_sym.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);

    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_full;
    wq_full[freq][pbc_full.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_full[freq][pbc_full.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);
    const auto inversion_minus_blocks = scalar_wq_to_blocks(
        {{0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}}, {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    wq_full[freq][pbc_full.klist.at(2)] =
        dense_wq_from_scalar_blocks(inversion_minus_blocks, ad_Wc);

    const auto expected =
        librpa_int::FT_Wc_freq_q(librpa_int::global::mpi_comm_global_h, wq_full, pbc_full, false);
    const auto actual =
        librpa_int::FT_Wc_freq_q(librpa_int::global::mpi_comm_global_h, wq_sym, pbc_sym, false,
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
        test_complex_spacetime_diagnostic_requires_explicit_enable();
        test_complex_spacetime_storage_does_not_change_non_soc_spin_weight();
        test_direct_compressed_sigc_diagnostic_requires_explicit_shrink_path();
        test_chi0_rspace_symmetry_diagnostic_requires_explicit_enable();
        test_gamma_shrink_transform_diagnostic_requires_explicit_enable();
        test_chi0_qspace_symmetry_diagnostic_requires_explicit_enable();
        test_rpa_finite_q_matrix_diagnostic_requires_explicit_enable();
        test_distributed_hermiticity_metrics_on_one_rank(blacs_h);
        test_sigc_rspace_symmetry_diagnostic_requires_explicit_enable();
        test_single_q_member_diagnostic_disables_sigc_rspace_symmetry_restore();
        test_strict_2d_qmember_diagnostic_selects_one_periodic_member();
        test_strict_2d_qmember_diagnostic_accepts_one_distributed_owner();
        test_wc_rf_output_collective_includes_ranks_without_local_frequency_blocks();
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
        test_strict_2d_radial_log_integral_matches_numeric_quadrature();
        test_strict_2d_radial_log_integral_is_stable_at_zero_and_small_a();
        test_strict_2d_rpa_trace_log_average_preserves_cell_normalization();
        test_strict_2d_rpa_trace_log_average_matches_dense_radial_quadrature();
        test_strict_2d_rpa_trace_log_average_rejects_invalid_geometry();
        test_strict_2d_rpa_trace_log_route_is_qavg_only();
        test_strict_2d_inverse_head_average_has_linear_q_screening();
        test_strict_2d_finite_q_reference_matches_head_and_schur_limits();
        test_strict_2d_schur_coefficient_removes_identity();
        test_strict_2d_screening_denominator_must_stay_on_physical_branch();
        test_strict_2d_gw_uses_full_coulomb_at_all_q();
        test_strict_2d_gw_routes_gamma_through_complete_wc_average();
        test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration();
        test_strict_2d_diagnostic_schema_and_qpoint_order_are_stable();
        test_strict_2d_qshell_uses_minimum_image_q();
        test_strict_2d_qradial_partitions_rest_exactly();
        test_strict_2d_first_shell_wc_block_diagnostic_is_exactly_additive();
        test_strict_2d_alpha_wc_diagnostic_requires_explicit_reference();
        test_strict_2d_first_shell_analytic_wc_diagnostic_requires_explicit_enable();
        test_strict_2d_finite_q_matrix_dump_selection_is_read_only_and_bounded();
        test_strict_2d_omega0_diagnostic_directory_is_explicit_and_normalized();
        test_strict_2d_omega0_override_basis_modes_are_mutually_exclusive();
        test_strict_2d_omega0_override_reader_validates_shape_and_payload();
        test_strict_2d_block_metrics_separate_head_wings_and_body();
        test_strict_2d_alpha_reference_averages_bare_coulomb();
        test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis();
        test_strict_2d_regular_coulomb_legs_are_projected_to_the_gamma_basis();
        test_strict_2d_wc_blocks_match_dense_finite_q_inverse();
        test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature();
        test_strict_2d_wc_cell_average_matches_cartesian_voronoi_subgrid();
        test_strict_2d_wc_blocks_have_finite_small_q_limits();
        test_strict_2d_wc_average_is_covariant_under_regular_body_rotation();
        test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks();
        test_rpa_chi0v_wing_desc_matches_producer_layout(blacs_h);
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
        test_weighted_wfc_gram_comparison_is_phase_invariant_and_detects_band_swap();
        test_gw_gf_kstar_wfc_diagnostic_requires_explicit_enable();
        test_gw_gf_kstar_wfc_diagnostic_uses_meanfield_kpoint_count();
        test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(blacs_h);
        test_kblacs_transform_matches_original_transform(blacs_h);
        test_transform_Cs2mnk_can_keep_spin_channels_separate(blacs_h);
        test_head_initialization_does_not_require_coulomb_diagonalization(blacs_h);
        test_strict_2d_gamma_quadrature_is_ready_after_wing_initialization(blacs_h);
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_qmember_diagnostic_keeps_original_full_bz_weight();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_spacetime_fourier_phases_form_k_minus_q_convolution();
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_on_odd_and_even_meshes();
        test_bn_kstar_green_function_matches_explicit_full_bz_on_odd_and_even_meshes();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_gamma_only_dense_wq_fourier_weight_scales_as_inverse_bvk_cells();
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
