#include <cassert>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);

    if (librpa_int::global::size_global != 4)
        throw std::runtime_error("test_rpa_headwing_mpi requires 4 MPI processes");

    librpa_int::BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
    blacs_h.init();
    blacs_h.set_square_grid();

    constexpr int n_abf = 257;
    librpa_int::ArrayDesc desc_regular(blacs_h);
    desc_regular.init_square_blk(n_abf, n_abf, 0, 0);

    librpa_int::ArrayDesc desc_optimized(blacs_h);
    desc_optimized.init(n_abf, n_abf, 128, 128, 0, 0);
    auto sqrt_coulomb =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);

    assert(desc_regular.m_loc() == desc_optimized.m_loc());
    assert(desc_regular.n_loc() == desc_optimized.n_loc());
    assert(librpa_int::rpa_headwing_matrix_matches_descriptor(
        sqrt_coulomb, desc_optimized, desc_optimized));
    assert(!librpa_int::rpa_headwing_matrix_matches_descriptor(
        sqrt_coulomb, desc_regular, desc_optimized));
    auto row_major_sqrt_coulomb = sqrt_coulomb.copy();
    row_major_sqrt_coulomb.swap_to_row_major();
    assert(!librpa_int::rpa_headwing_matrix_matches_descriptor(
        row_major_sqrt_coulomb, desc_optimized, desc_optimized));

    constexpr int n_diag = 6;
    librpa_int::ArrayDesc desc_diag(blacs_h);
    desc_diag.init(n_diag, n_diag, 2, 2, 0, 0);
    auto hermitian =
        librpa_int::init_local_mat<std::complex<double>>(desc_diag, librpa_int::MAJOR::COL);
    for (int iloc = 0; iloc != hermitian.nr(); ++iloc)
    {
        const int i = desc_diag.indx_l2g_r(iloc);
        for (int jloc = 0; jloc != hermitian.nc(); ++jloc)
        {
            const int j = desc_diag.indx_l2g_c(jloc);
            hermitian(iloc, jloc) =
                i == j ? std::complex<double>(2.0 + i, 0.0)
                       : (i < j ? std::complex<double>(0.25 * (i + j + 1), 0.1 * (j - i))
                                : std::conj(std::complex<double>(0.25 * (i + j + 1),
                                                                 0.1 * (i - j))));
        }
    }
    const auto exact_metrics =
        librpa_int::distributed_hermiticity_metrics(hermitian, desc_diag);
    assert(exact_metrics.frobenius_norm > 0.0);
    assert(exact_metrics.antihermitian_frobenius_norm < 1.0e-14);
    assert(exact_metrics.antihermitian_max_abs < 1.0e-14);
    assert(exact_metrics.relative_frobenius_residual < 1.0e-14);

    const int iloc = desc_diag.indx_g2l_r(0);
    const int jloc = desc_diag.indx_g2l_c(1);
    if (iloc >= 0 && jloc >= 0) hermitian(iloc, jloc) += std::complex<double>(0.0, 0.5);
    const auto perturbed_metrics =
        librpa_int::distributed_hermiticity_metrics(hermitian, desc_diag);
    assert(std::abs(perturbed_metrics.antihermitian_frobenius_norm - std::sqrt(0.5)) < 1.0e-13);
    assert(std::abs(perturbed_metrics.antihermitian_max_abs - 0.5) < 1.0e-13);
    assert(perturbed_metrics.relative_frobenius_residual > 0.0);

    librpa_int::AtomicBasis atom_basis(std::vector<std::size_t>{2, 1});
    const std::array<double, 3> q_key{0.25, 0.0, 0.0};
    using tensor_map_t = std::map<
        int, std::map<std::pair<int, std::array<double, 3>>,
                      RI::Tensor<std::complex<double>>>>;
    tensor_map_t upper_atom_blocks;
    auto same_atom_data = std::make_shared<std::valarray<std::complex<double>>>(
        std::complex<double>(0.0), 4);
    (*same_atom_data)[0] = {1.0, 0.0};
    (*same_atom_data)[1] = {2.0, 1.0};
    (*same_atom_data)[2] = {3.0, 0.5};
    (*same_atom_data)[3] = {4.0, 0.0};
    upper_atom_blocks[0][{0, q_key}] =
        RI::Tensor<std::complex<double>>({2, 2}, same_atom_data);
    auto cross_atom_data = std::make_shared<std::valarray<std::complex<double>>>(
        std::complex<double>(0.0), 2);
    (*cross_atom_data)[0] = {5.0, 1.0};
    (*cross_atom_data)[1] = {6.0, 2.0};
    upper_atom_blocks[0][{1, q_key}] =
        RI::Tensor<std::complex<double>>({2, 1}, cross_atom_data);
    auto last_atom_data = std::make_shared<std::valarray<std::complex<double>>>(
        std::complex<double>(7.0, 0.0), 1);
    upper_atom_blocks[1][{1, q_key}] =
        RI::Tensor<std::complex<double>>({1, 1}, last_atom_data);

    librpa_int::ArrayDesc desc_collected(blacs_h);
    desc_collected.init(3, 3, 1, 1, 0, 0);
    auto collected = librpa_int::init_local_mat<std::complex<double>>(
        desc_collected, librpa_int::MAJOR::COL);
    librpa_int::collect_block_from_ALL_IJ_Tensor(
        collected, desc_collected, atom_basis, q_key, true, std::complex<double>(1.0),
        upper_atom_blocks, librpa_int::MAJOR::ROW);
    for (int iloc_collected = 0; iloc_collected != collected.nr(); ++iloc_collected)
    {
        const int i = desc_collected.indx_l2g_r(iloc_collected);
        for (int jloc_collected = 0; jloc_collected != collected.nc(); ++jloc_collected)
        {
            const int j = desc_collected.indx_l2g_c(jloc_collected);
            if (i == 0 && j == 1)
                assert(collected(iloc_collected, jloc_collected) ==
                       std::complex<double>(2.0, 1.0));
            if (i == 1 && j == 0)
                assert(collected(iloc_collected, jloc_collected) ==
                       std::complex<double>(3.0, 0.5));
            if (i == 0 && j == 2)
                assert(collected(iloc_collected, jloc_collected) ==
                       std::complex<double>(5.0, 1.0));
            if (i == 2 && j == 0)
                assert(collected(iloc_collected, jloc_collected) ==
                       std::complex<double>(5.0, -1.0));
        }
    }
    const auto collected_metrics =
        librpa_int::distributed_hermiticity_metrics(collected, desc_collected);
    assert(collected_metrics.antihermitian_max_abs > 1.0);

    librpa_int::ArrayDesc desc_two(blacs_h);
    desc_two.init(2, 2, 1, 1, 0, 0);
    auto coulomb =
        librpa_int::init_local_mat<std::complex<double>>(desc_two, librpa_int::MAJOR::COL);
    auto sqrt_coulomb_two = coulomb.copy();
    auto chi0 = coulomb.copy();
    for (int iloc_two = 0; iloc_two != coulomb.nr(); ++iloc_two)
    {
        const int i = desc_two.indx_l2g_r(iloc_two);
        for (int jloc_two = 0; jloc_two != coulomb.nc(); ++jloc_two)
        {
            const int j = desc_two.indx_l2g_c(jloc_two);
            if (i == j)
            {
                coulomb(iloc_two, jloc_two) = i == 0 ? 2.0 : 1.0;
                sqrt_coulomb_two(iloc_two, jloc_two) = i == 0 ? std::sqrt(2.0) : 1.0;
                chi0(iloc_two, jloc_two) = i == 0 ? -0.15 : -0.10;
            }
            else
            {
                chi0(iloc_two, jloc_two) =
                    i == 0 ? std::complex<double>(0.04, 0.03)
                           : std::complex<double>(0.04, -0.03);
            }
        }
    }

    auto v_chi0 = coulomb.copy();
    auto chi0_sqrt_v = coulomb.copy();
    auto symmetric_response = coulomb.copy();
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', 2, 2, 2, std::complex<double>(1.0), coulomb.ptr(), 1, 1, desc_two.desc,
        chi0.ptr(), 1, 1, desc_two.desc, std::complex<double>(0.0), v_chi0.ptr(), 1, 1,
        desc_two.desc);
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', 2, 2, 2, std::complex<double>(1.0), chi0.ptr(), 1, 1, desc_two.desc,
        sqrt_coulomb_two.ptr(), 1, 1, desc_two.desc, std::complex<double>(0.0),
        chi0_sqrt_v.ptr(), 1, 1, desc_two.desc);
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', 2, 2, 2, std::complex<double>(1.0), sqrt_coulomb_two.ptr(), 1, 1,
        desc_two.desc, chi0_sqrt_v.ptr(), 1, 1, desc_two.desc, std::complex<double>(0.0),
        symmetric_response.ptr(), 1, 1, desc_two.desc);

    assert(librpa_int::distributed_hermiticity_metrics(coulomb, desc_two)
               .relative_frobenius_residual < 1.0e-14);
    assert(librpa_int::distributed_hermiticity_metrics(chi0, desc_two)
               .relative_frobenius_residual < 1.0e-14);
    assert(librpa_int::distributed_hermiticity_metrics(v_chi0, desc_two)
               .relative_frobenius_residual > 1.0e-2);
    assert(librpa_int::distributed_hermiticity_metrics(symmetric_response, desc_two)
               .relative_frobenius_residual < 1.0e-14);

    const auto general_integrand =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(v_chi0, desc_two);
    const auto symmetric_integrand =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(symmetric_response, desc_two);
    const double k00 = -0.30;
    const double k11 = -0.10;
    const double k01_abs_squared = 2.0 * (0.04 * 0.04 + 0.03 * 0.03);
    const double eig_gap =
        std::sqrt((k00 - k11) * (k00 - k11) + 4.0 * k01_abs_squared);
    const double eig0 = 0.5 * (k00 + k11 - eig_gap);
    const double eig1 = 0.5 * (k00 + k11 + eig_gap);
    const double eigenvalue_integrand = eig0 + std::log(1.0 - eig0) +
                                       eig1 + std::log(1.0 - eig1);
    assert(std::abs(general_integrand.imag()) < 1.0e-13);
    assert(std::abs(symmetric_integrand.imag()) < 1.0e-13);
    assert(std::abs(general_integrand.real() - symmetric_integrand.real()) < 1.0e-13);
    assert(std::abs(symmetric_integrand.real() - eigenvalue_integrand) < 1.0e-13);

    auto positive_det_with_pivot =
        librpa_int::init_local_mat<std::complex<double>>(desc_two, librpa_int::MAJOR::COL);
    for (int iloc_two = 0; iloc_two != positive_det_with_pivot.nr(); ++iloc_two)
    {
        const int i = desc_two.indx_l2g_r(iloc_two);
        for (int jloc_two = 0; jloc_two != positive_det_with_pivot.nc(); ++jloc_two)
        {
            const int j = desc_two.indx_l2g_c(jloc_two);
            if (i == 0 && j == 0) positive_det_with_pivot(iloc_two, jloc_two) = 0.1;
            if (i == 0 && j == 1) positive_det_with_pivot(iloc_two, jloc_two) = {0.0, 1.0};
            if (i == 1 && j == 0) positive_det_with_pivot(iloc_two, jloc_two) = {0.0, 1.0};
            if (i == 1 && j == 1) positive_det_with_pivot(iloc_two, jloc_two) = 1.0;
        }
    }
    std::vector<int> pivot(std::max(1, desc_two.m_loc() * 10));
    int pivot_info = 0;
    const auto pivot_logdet = librpa_int::compute_pi_det_blacs_2d(
        positive_det_with_pivot, desc_two, pivot.data(), pivot_info);
    assert(pivot_info == 0);
    assert(std::abs(pivot_logdet.real() - std::log(1.1)) < 1.0e-13);
    assert(std::abs(pivot_logdet.imag()) < 1.0e-13);

    auto near_negative_det =
        librpa_int::init_local_mat<std::complex<double>>(desc_two, librpa_int::MAJOR::COL);
    near_negative_det.zero_out();
    const std::complex<double> first_negative_diagonal(-2.0, 1.0e-8);
    for (int i = 0; i != 2; ++i)
    {
        const int iloc_two = desc_two.indx_g2l_r(i);
        const int jloc_two = desc_two.indx_g2l_c(i);
        if (iloc_two >= 0 && jloc_two >= 0)
            near_negative_det(iloc_two, jloc_two) = i == 0 ? first_negative_diagonal : 3.0;
    }
    std::vector<int> negative_pivot(std::max(1, desc_two.m_loc() * 10));
    int negative_info = 0;
    const auto negative_logdet = librpa_int::compute_pi_det_blacs_2d(
        near_negative_det, desc_two, negative_pivot.data(), negative_info);
    const auto expected_negative_logdet = std::log(3.0 * first_negative_diagonal);
    assert(negative_info == 0);
    assert(std::abs(negative_logdet.real() - expected_negative_logdet.real()) < 1.0e-13);
    assert(std::abs(negative_logdet.imag() - expected_negative_logdet.imag()) < 1.0e-13);

    librpa_int::ArrayDesc desc_four_shifted(blacs_h);
    desc_four_shifted.init(4, 4, 2, 2, 1, 1);
    auto diagonal_positive = librpa_int::init_local_mat<std::complex<double>>(
        desc_four_shifted, librpa_int::MAJOR::COL);
    diagonal_positive.zero_out();
    const double diagonal_values[4] = {2.0, 3.0, 4.0, 5.0};
    for (int i = 0; i != 4; ++i)
    {
        const int iloc_four = desc_four_shifted.indx_g2l_r(i);
        const int jloc_four = desc_four_shifted.indx_g2l_c(i);
        if (iloc_four >= 0 && jloc_four >= 0)
            diagonal_positive(iloc_four, jloc_four) = diagonal_values[i];
    }
    std::vector<int> diagonal_pivot(std::max(1, desc_four_shifted.m_loc() * 10));
    int diagonal_info = 0;
    const auto diagonal_logdet = librpa_int::compute_pi_det_blacs_2d(
        diagonal_positive, desc_four_shifted, diagonal_pivot.data(), diagonal_info);
    assert(diagonal_info == 0);
    assert(std::abs(diagonal_logdet.real() - std::log(120.0)) < 1.0e-13);
    assert(std::abs(diagonal_logdet.imag()) < 1.0e-13);

    auto two_cross_process_pivots = librpa_int::init_local_mat<std::complex<double>>(
        desc_four_shifted, librpa_int::MAJOR::COL);
    two_cross_process_pivots.zero_out();
    const int permuted_row[4] = {2, 3, 0, 1};
    for (int i = 0; i != 4; ++i)
    {
        const int row = permuted_row[i];
        const int iloc_four = desc_four_shifted.indx_g2l_r(row);
        const int jloc_four = desc_four_shifted.indx_g2l_c(i);
        if (iloc_four >= 0 && jloc_four >= 0)
            two_cross_process_pivots(iloc_four, jloc_four) = diagonal_values[i];
    }
    std::vector<int> cross_pivot(std::max(1, desc_four_shifted.m_loc() * 10));
    int cross_info = 0;
    const auto cross_logdet = librpa_int::compute_pi_det_blacs_2d(
        two_cross_process_pivots, desc_four_shifted, cross_pivot.data(), cross_info);
    assert(cross_info == 0);
    assert(std::abs(cross_logdet.real() - std::log(120.0)) < 1.0e-13);
    assert(std::abs(cross_logdet.imag()) < 1.0e-13);

    blacs_h.exit();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
