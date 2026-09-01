#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);

    if (librpa_int::global::size_global != 4)
        throw std::runtime_error("test_rpa_headwing_mpi requires 4 MPI processes");

    librpa_int::BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
    blacs_h.init();
    blacs_h.set_square_grid();

    constexpr int n_abf = 512;
    librpa_int::ArrayDesc desc_regular(blacs_h);
    desc_regular.init_square_blk(n_abf, n_abf, 0, 0);

    librpa_int::ArrayDesc desc_optimized(blacs_h);
    desc_optimized.init(n_abf, n_abf, 128, 128, 0, 0);
    auto sqrt_coulomb =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);

    assert(desc_regular.m_loc() == desc_optimized.m_loc());
    assert(desc_regular.n_loc() == desc_optimized.n_loc());
    assert(librpa_int::rpa_headwing_matrix_matches_descriptor(sqrt_coulomb, desc_optimized,
                                                              desc_optimized));
    assert(!librpa_int::rpa_headwing_matrix_matches_descriptor(sqrt_coulomb, desc_regular,
                                                               desc_optimized));
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
                assert(collected(iloc_collected, jloc_collected) == std::complex<double>(5.0, 1.0));
            if (i == 2 && j == 0)
                assert(collected(iloc_collected, jloc_collected) ==
                       std::complex<double>(5.0, -1.0));
        }
    }
    const auto collected_metrics =
        librpa_int::distributed_hermiticity_metrics(collected, desc_collected);
    assert(collected_metrics.antihermitian_max_abs > 1.0);

    // Exercise the production full-BZ wing path with one k point distributed
    // over a real 2x2 k-point BLACS group.  Compare every auxiliary-basis wing
    // element against a dense transition sum, rather than comparing only the
    // Cartesian Gram matrix or a few projected rows.
    constexpr int wing_n_basis = 6;
    constexpr int wing_n_states = 6;
    constexpr int wing_n_abf = 5;
    librpa_int::MeanField wing_meanfield(1, 1, wing_n_states, wing_n_basis, 1);
    const double wing_energies[wing_n_states] = {-0.8, -0.4, 0.2, 0.5, 0.9, 1.4};
    for (int iband = 0; iband != wing_n_states; ++iband)
    {
        wing_meanfield.get_eigenvals()[0](0, iband) = wing_energies[iband];
        wing_meanfield.get_weight()[0](0, iband) = iband < 2 ? 2.0 : 0.0;
    }
    auto &wing_wfc = wing_meanfield.get_eigenvectors()[0][0][0];
    wing_wfc.create(wing_n_states, wing_n_basis);
    for (int iband = 0; iband != wing_n_states; ++iband)
        for (int iao = 0; iao != wing_n_basis; ++iao)
            wing_wfc(iband, iao) = iband == iao ? 1.0 : 0.0;

    librpa_int::velocity_matrix_t wing_velocity;
    librpa_int::initialize_velocity_matrix(wing_velocity, 1, 1, wing_n_states);
    for (int iocc = 0; iocc != 2; ++iocc)
    {
        for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
        {
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                const std::complex<double> value(
                    0.03 * (iunocc + 1) * (alpha + 1),
                    0.02 * (iocc + 1) * (alpha + 1));
                wing_velocity[0][0][alpha](iunocc, iocc) = value;
                wing_velocity[0][0][alpha](iocc, iunocc) = std::conj(value);
            }
        }
    }

    librpa_int::AtomicBasis wing_basis_wfc(std::vector<std::size_t>{wing_n_basis});
    librpa_int::AtomicBasis wing_basis_abf(std::vector<std::size_t>{wing_n_abf});
    librpa_int::PeriodicBoundaryData wing_pbc;
    const std::vector<librpa_int::Vector3_Order<double>> wing_kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> wing_omega{0.5, 1.25};

    auto wing_tensor_data = std::make_shared<std::valarray<double>>(
        0.0, wing_n_abf * wing_n_basis * wing_n_basis);
    for (int mu = 0; mu != wing_n_abf; ++mu)
    {
        for (int iao = 0; iao != wing_n_basis; ++iao)
        {
            for (int jao = 0; jao != wing_n_basis; ++jao)
            {
                (*wing_tensor_data)[(mu * wing_n_basis + iao) * wing_n_basis + jao] =
                    0.01 * (mu + 1) * (iao + 1) - 0.004 * (jao + 1) +
                    (iao == jao ? 0.08 : 0.0);
            }
        }
    }
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>>
        wing_cs_ij;
    wing_cs_ij[0][{0, {0, 0, 0}}] = RI::Tensor<double>(
        {static_cast<std::size_t>(wing_n_abf), static_cast<std::size_t>(wing_n_basis),
         static_cast<std::size_t>(wing_n_basis)},
        wing_tensor_data);

    librpa_int::KPointBlacsProcessShape wing_shape(1, 4, true);
    librpa_int::KPointBlacsParallelContext wing_kblacs(wing_shape, MPI_COMM_WORLD, 1);
    assert(wing_kblacs.blacs_nprows() == 2 && wing_kblacs.blacs_npcols() == 2);
    librpa_int::diele_func wing_df(
        wing_meanfield, wing_velocity, wing_kfrac, wing_basis_wfc, wing_basis_abf, wing_omega,
        wing_n_basis, wing_n_states, 1, wing_n_abf, wing_pbc,
        librpa_int::global::mpi_comm_global_h, blacs_h, &wing_kblacs);

    std::vector<librpa_int::ComplexMatrix> dense_c_mnk;
    dense_c_mnk.reserve(wing_n_abf);
    std::vector<std::complex<double>> distributed_wing_mu(
        wing_omega.size() * 3 * wing_n_abf, 0.0);
    for (int mu = 0; mu != wing_n_abf; ++mu)
    {
        const auto distributed_c = wing_df.transform_Cs2mnk_kblacs(
            0, mu, wing_cs_ij, wing_kblacs.blacs_h, wing_kfrac[0]);
        auto *distributed_wing_mu_for_mu =
            distributed_wing_mu.data() + mu * wing_omega.size() * 3;
        for (int iocc = 0; iocc != 2; ++iocc)
        {
            const int mloc = distributed_c.first.indx_g2l_r(iocc);
            if (mloc < 0) continue;
            for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
            {
                const int nloc = distributed_c.first.indx_g2l_c(iunocc);
                if (nloc < 0) continue;
                const std::array<std::complex<double>, 3> velocity_unocc_occ{
                    wing_velocity[0][0][0](iunocc, iocc),
                    wing_velocity[0][0][1](iunocc, iocc),
                    wing_velocity[0][0][2](iunocc, iocc)};
                librpa_int::accumulate_wing_mu_for_pair(
                    wing_omega, velocity_unocc_occ, distributed_c.second(mloc, nloc),
                    wing_energies[iunocc] - wing_energies[iocc], 1.0, 0.0,
                    distributed_wing_mu_for_mu);
            }
        }
        librpa_int::ComplexMatrix dense_c(wing_n_states, wing_n_states);
        for (int m = 0; m != wing_n_states; ++m)
        {
            for (int n = 0; n != wing_n_states; ++n)
            {
                const int mloc = distributed_c.first.indx_g2l_r(m);
                const int nloc = distributed_c.first.indx_g2l_c(n);
                std::complex<double> value = 0.0;
                if (mloc >= 0 && nloc >= 0) value = distributed_c.second(mloc, nloc);
                MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                              MPI_COMM_WORLD);
                dense_c(m, n) = value;
            }
        }
        dense_c_mnk.push_back(std::move(dense_c));
    }
    MPI_Allreduce(MPI_IN_PLACE, distributed_wing_mu.data(),
                  static_cast<int>(distributed_wing_mu.size()), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);

    for (std::size_t iomega = 0; iomega != wing_omega.size(); ++iomega)
    {
        for (int mu = 0; mu != wing_n_abf; ++mu)
        {
            std::array<std::complex<double>, 3> expected{};
            for (int iocc = 0; iocc != 2; ++iocc)
            {
                for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
                {
                    const double egap = wing_energies[iunocc] - wing_energies[iocc];
                    const double denominator =
                        wing_omega[iomega] * wing_omega[iomega] + egap * egap;
                    const auto c_mn = dense_c_mnk[mu](iocc, iunocc);
                    for (int alpha = 0; alpha != 3; ++alpha)
                    {
                        expected[alpha] +=
                            std::conj(c_mn * wing_velocity[0][0][alpha](iunocc, iocc)) /
                            denominator;
                    }
                }
            }
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                const auto actual = distributed_wing_mu
                    [mu * wing_omega.size() * 3 + iomega * 3 + alpha];
                assert(std::abs(actual - expected[alpha]) < 2.0e-12);
            }
        }
    }

    // Match the MoS2 runtime layout: four k-point groups with one BLACS rank
    // per group.  Each rank owns a disjoint subset of k points and the final
    // auxiliary wing is the global sum over those subsets.
    constexpr int kdist_nk = 4;
    librpa_int::MeanField kdist_meanfield(1, kdist_nk, wing_n_states, wing_n_basis, 1);
    librpa_int::velocity_matrix_t kdist_velocity;
    librpa_int::initialize_velocity_matrix(kdist_velocity, 1, kdist_nk, wing_n_states);
    std::vector<librpa_int::Vector3_Order<double>> kdist_kfrac;
    for (int ik = 0; ik != kdist_nk; ++ik)
    {
        kdist_kfrac.push_back({0.125 * ik, 0.0, 0.0});
        for (int iband = 0; iband != wing_n_states; ++iband)
        {
            kdist_meanfield.get_eigenvals()[0](ik, iband) = wing_energies[iband] + 0.01 * ik;
            kdist_meanfield.get_weight()[0](ik, iband) = iband < 2 ? 0.5 : 0.0;
        }
        auto &kdist_wfc = kdist_meanfield.get_eigenvectors()[0][0][ik];
        kdist_wfc.create(wing_n_states, wing_n_basis);
        for (int iband = 0; iband != wing_n_states; ++iband)
        {
            for (int iao = 0; iao != wing_n_basis; ++iao)
                kdist_wfc(iband, iao) = iband == iao ? 1.0 : 0.0;
        }
        for (int iocc = 0; iocc != 2; ++iocc)
        {
            for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
            {
                for (int alpha = 0; alpha != 3; ++alpha)
                {
                    kdist_velocity[0][ik][alpha](iunocc, iocc) =
                        std::complex<double>(0.02 * (ik + 1) * (iunocc + 1) * (alpha + 1),
                                             0.01 * (iocc + 1) * (alpha + 1));
                }
            }
        }
    }

    librpa_int::KPointBlacsParallelContext kdist_kblacs(
        librpa_int::KPointBlacsProcessShape(4, 1, true), MPI_COMM_WORLD, kdist_nk);
    assert(kdist_kblacs.process_shape().nprocs_kpoint == 4);
    assert(kdist_kblacs.process_shape().nprocs_blacs == 1);
    assert(kdist_kblacs.kpoints_local().size() == 1);
    librpa_int::diele_func kdist_df(
        kdist_meanfield, kdist_velocity, kdist_kfrac, wing_basis_wfc, wing_basis_abf, wing_omega,
        wing_n_basis, wing_n_states, 1, wing_n_abf, wing_pbc,
        librpa_int::global::mpi_comm_global_h, blacs_h, &kdist_kblacs);

    std::vector<std::complex<double>> kdist_wing_mu(
        wing_omega.size() * 3 * wing_n_abf, 0.0);
    for (const int ik : kdist_kblacs.kpoints_local())
    {
        for (int mu = 0; mu != wing_n_abf; ++mu)
        {
            const auto distributed_c = kdist_df.transform_Cs2mnk_kblacs(
                ik, mu, wing_cs_ij, kdist_kblacs.blacs_h, kdist_kfrac[ik]);
            auto *wing_for_mu = kdist_wing_mu.data() + mu * wing_omega.size() * 3;
            for (int iocc = 0; iocc != 2; ++iocc)
            {
                const int mloc = distributed_c.first.indx_g2l_r(iocc);
                for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
                {
                    const int nloc = distributed_c.first.indx_g2l_c(iunocc);
                    assert(mloc >= 0 && nloc >= 0);
                    const std::array<std::complex<double>, 3> velocity_unocc_occ{
                        kdist_velocity[0][ik][0](iunocc, iocc),
                        kdist_velocity[0][ik][1](iunocc, iocc),
                        kdist_velocity[0][ik][2](iunocc, iocc)};
                    librpa_int::accumulate_wing_mu_for_pair(
                        wing_omega, velocity_unocc_occ, distributed_c.second(mloc, nloc),
                        wing_energies[iunocc] - wing_energies[iocc], 1.0, 0.0, wing_for_mu);
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, kdist_wing_mu.data(), static_cast<int>(kdist_wing_mu.size()),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    for (int mu = 0; mu != wing_n_abf; ++mu)
    {
        for (std::size_t iomega = 0; iomega != wing_omega.size(); ++iomega)
        {
            std::array<std::complex<double>, 3> expected{};
            for (int ik = 0; ik != kdist_nk; ++ik)
            {
                for (int iocc = 0; iocc != 2; ++iocc)
                {
                    for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
                    {
                        const double c_mn =
                            (*wing_tensor_data)
                                [(mu * wing_n_basis + iocc) * wing_n_basis + iunocc] +
                            (*wing_tensor_data)
                                [(mu * wing_n_basis + iunocc) * wing_n_basis + iocc];
                        const double egap = wing_energies[iunocc] - wing_energies[iocc];
                        const double denominator =
                            wing_omega[iomega] * wing_omega[iomega] + egap * egap;
                        for (int alpha = 0; alpha != 3; ++alpha)
                            expected[alpha] +=
                                std::conj(c_mn * kdist_velocity[0][ik][alpha](iunocc, iocc)) /
                                denominator;
                    }
                }
            }
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                const auto actual = kdist_wing_mu
                    [mu * wing_omega.size() * 3 + iomega * 3 + alpha];
                assert(std::abs(actual - expected[alpha]) < 2.0e-12);
            }
        }
    }

    // Exercise the symmetry-restored member pairing used by cal_wing_symmetric.
    // Each rank owns one IBZ star.  Every star has two full-BZ members whose
    // direct WFC and velocity sources are deliberately stored in a nontrivial
    // order.  Band phases cancel only when the WFC and velocity come from the
    // same source, while an R != 0 coefficient block makes the member k target
    // observable as well.
    constexpr int member_n_bz = 2 * kdist_nk;
    librpa_int::MeanField member_wfc_full(1, member_n_bz, wing_n_states, wing_n_basis, 1);
    librpa_int::velocity_matrix_t member_velocity_full;
    librpa_int::initialize_velocity_matrix(member_velocity_full, 1, member_n_bz, wing_n_states);
    std::vector<std::vector<std::complex<double>>> member_band_phases(
        member_n_bz, std::vector<std::complex<double>>(wing_n_states));
    for (int ik_source = 0; ik_source != member_n_bz; ++ik_source)
    {
        auto &wfc = member_wfc_full.get_eigenvectors()[0][0][ik_source];
        wfc.create(wing_n_states, wing_n_basis);
        for (int iband = 0; iband != wing_n_states; ++iband)
        {
            const double angle = 0.07 * (ik_source + 1) * (iband + 1);
            member_band_phases[ik_source][iband] = std::polar(1.0, angle);
            for (int iao = 0; iao != wing_n_basis; ++iao)
                wfc(iband, iao) = iband == iao ? member_band_phases[ik_source][iband] : 0.0;
        }
        for (int iocc = 0; iocc != 2; ++iocc)
        {
            for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
            {
                for (int alpha = 0; alpha != 3; ++alpha)
                {
                    const std::complex<double> base_velocity(
                        0.015 * (ik_source + 1) * (iunocc + 1) * (alpha + 1),
                        0.008 * (iocc + 1) * (alpha + 1));
                    member_velocity_full[0][ik_source][alpha](iunocc, iocc) =
                        std::conj(member_band_phases[ik_source][iunocc]) * base_velocity *
                        member_band_phases[ik_source][iocc];
                }
            }
        }
    }

    const std::vector<std::vector<int>> member_source_ik{{3, 7}, {6, 0}, {5, 2}, {1, 4}};
    librpa_int::SymmetryContext member_symmetry;
    member_symmetry.available = true;
    for (int ik_ibz = 0; ik_ibz != kdist_nk; ++ik_ibz)
    {
        librpa_int::SymmetryKStar star;
        star.star_index = ik_ibz;
        star.k_ibz = kdist_kfrac[ik_ibz];
        for (int imember = 0; imember != 2; ++imember)
        {
            librpa_int::SymmetryKStarMember member;
            member.spatial_isym = imember;
            member.k_bz = {0.04 + 0.17 * ik_ibz + 0.09 * imember, 0.03 * (ik_ibz - imember), 0.0};
            star.members.push_back(member);
        }
        member_symmetry.kstars.push_back(std::move(star));
    }
    assert(member_symmetry.count_kstar_members() == member_n_bz);

    auto member_tensor_r1_data =
        std::make_shared<std::valarray<double>>(0.0, wing_n_abf * wing_n_basis * wing_n_basis);
    for (int mu = 0; mu != wing_n_abf; ++mu)
    {
        for (int iao = 0; iao != wing_n_basis; ++iao)
        {
            for (int jao = 0; jao != wing_n_basis; ++jao)
            {
                (*member_tensor_r1_data)[(mu * wing_n_basis + iao) * wing_n_basis + jao] =
                    0.006 * (mu + 1) * (iao + 2) + 0.003 * (jao + 1);
            }
        }
    }
    auto member_cs_ij = wing_cs_ij;
    member_cs_ij[0][{0, {1, 0, 0}}] = RI::Tensor<double>(
        {static_cast<std::size_t>(wing_n_abf), static_cast<std::size_t>(wing_n_basis),
         static_cast<std::size_t>(wing_n_basis)},
        member_tensor_r1_data);

    const double member_bz_weight_scale =
        static_cast<double>(kdist_nk) / static_cast<double>(member_n_bz);
    const double member_factor1 = (0.5 * member_bz_weight_scale) / 2.0 * (1.0 - 0.0 * member_n_bz);
    assert(std::abs(member_factor1 - 0.125) < 1.0e-15);
    std::vector<std::complex<double>> member_wing_mu(wing_omega.size() * 3 * wing_n_abf, 0.0);
    for (const int ik_ibz : kdist_kblacs.kpoints_local())
    {
        const auto &star =
            librpa_int::find_symmetry_kstar_for_ibz_kpoint(member_symmetry, kdist_kfrac[ik_ibz]);
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto &velocity = librpa_int::direct_full_bz_velocity_for_kstar_member(
                member_velocity_full, member_source_ik, 0, ik_ibz, imember);
            const auto &wfc = librpa_int::direct_full_bz_wfc_for_kstar_member(
                member_wfc_full, member_source_ik, 0, 0, ik_ibz, imember);
            const std::vector<std::vector<const librpa_int::ComplexMatrix *>> wfc_ptrs{{&wfc}};
            for (int mu = 0; mu != wing_n_abf; ++mu)
            {
                const auto distributed_c =
                    kdist_df.transform_Cs2mnk_kblacs(ik_ibz, mu, member_cs_ij, kdist_kblacs.blacs_h,
                                                     star.members[imember].k_bz, &wfc_ptrs, 0);
                auto *wing_for_mu = member_wing_mu.data() + mu * wing_omega.size() * 3;
                for (int iocc = 0; iocc != 2; ++iocc)
                {
                    const int mloc = distributed_c.first.indx_g2l_r(iocc);
                    for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
                    {
                        const int nloc = distributed_c.first.indx_g2l_c(iunocc);
                        assert(mloc >= 0 && nloc >= 0);
                        const std::array<std::complex<double>, 3> velocity_unocc_occ{
                            velocity[0](iunocc, iocc), velocity[1](iunocc, iocc),
                            velocity[2](iunocc, iocc)};
                        librpa_int::accumulate_wing_mu_for_pair(
                            wing_omega, velocity_unocc_occ, distributed_c.second(mloc, nloc),
                            wing_energies[iunocc] - wing_energies[iocc], member_factor1, 0.0,
                            wing_for_mu);
                    }
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, member_wing_mu.data(), static_cast<int>(member_wing_mu.size()),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    std::vector<std::complex<double>> member_expected(wing_omega.size() * 3 * wing_n_abf, 0.0);
    std::vector<std::complex<double>> member_mismatched(wing_omega.size() * 3 * wing_n_abf, 0.0);
    for (int ik_ibz = 0; ik_ibz != kdist_nk; ++ik_ibz)
    {
        const auto &star = member_symmetry.kstars[ik_ibz];
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const int ik_source = member_source_ik[ik_ibz][imember];
            const int mismatched_velocity_source = (ik_source + 1) % member_n_bz;
            const auto phase_k = std::polar(1.0, 2.0 * M_PI * star.members[imember].k_bz.x);
            for (int mu = 0; mu != wing_n_abf; ++mu)
            {
                auto *expected_for_mu = member_expected.data() + mu * wing_omega.size() * 3;
                auto *mismatched_for_mu = member_mismatched.data() + mu * wing_omega.size() * 3;
                for (int iocc = 0; iocc != 2; ++iocc)
                {
                    for (int iunocc = 2; iunocc != wing_n_states; ++iunocc)
                    {
                        const auto tensor_index = [=](const int iao, const int jao)
                        { return (mu * wing_n_basis + iao) * wing_n_basis + jao; };
                        const auto c_occ_unocc =
                            (*wing_tensor_data)[tensor_index(iocc, iunocc)] +
                            phase_k * (*member_tensor_r1_data)[tensor_index(iocc, iunocc)];
                        const auto c_unocc_occ =
                            (*wing_tensor_data)[tensor_index(iunocc, iocc)] +
                            phase_k * (*member_tensor_r1_data)[tensor_index(iunocc, iocc)];
                        const auto c_sym = c_occ_unocc + std::conj(c_unocc_occ);
                        const auto c_mn = std::conj(member_band_phases[ik_source][iocc]) * c_sym *
                                          member_band_phases[ik_source][iunocc];
                        std::array<std::complex<double>, 3> velocity_unocc_occ{};
                        std::array<std::complex<double>, 3> mismatched_velocity{};
                        for (int alpha = 0; alpha != 3; ++alpha)
                        {
                            velocity_unocc_occ[alpha] =
                                member_velocity_full[0][ik_source][alpha](iunocc, iocc);
                            mismatched_velocity[alpha] =
                                member_velocity_full[0][mismatched_velocity_source][alpha](iunocc,
                                                                                           iocc);
                        }
                        const double egap = wing_energies[iunocc] - wing_energies[iocc];
                        librpa_int::accumulate_wing_mu_for_pair(wing_omega, velocity_unocc_occ,
                                                                c_mn, egap, member_factor1, 0.0,
                                                                expected_for_mu);
                        librpa_int::accumulate_wing_mu_for_pair(wing_omega, mismatched_velocity,
                                                                c_mn, egap, member_factor1, 0.0,
                                                                mismatched_for_mu);
                    }
                }
            }
        }
    }

    double member_mismatch_norm = 0.0;
    for (std::size_t index = 0; index != member_expected.size(); ++index)
    {
        assert(std::abs(member_wing_mu[index] - member_expected[index]) < 2.0e-12);
        member_mismatch_norm += std::norm(member_mismatched[index] - member_expected[index]);
    }
    assert(std::sqrt(member_mismatch_norm) > 1.0e-4);

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

    // Exercise the complete distributed Schur correction used by the RPA
    // Gamma head/wing path.  The result must be invariant under arbitrary
    // row phases of the Coulomb eigenvectors when the body and wing are
    // transformed together.
    constexpr int n_body = 17;
    librpa_int::ArrayDesc desc_schur_body(blacs_h);
    desc_schur_body.init_square_blk(n_body, n_body, 0, 0);
    librpa_int::ArrayDesc desc_schur_wing_seed(blacs_h);
    desc_schur_wing_seed.init_square_blk(n_body, 3, 0, 0);
    librpa_int::ArrayDesc desc_schur_wing(blacs_h);
    desc_schur_wing.init(n_body, 3, desc_schur_body.mb(), desc_schur_wing_seed.nb(), 0, 0);

    librpa_int::matrix_m<std::complex<double>> dense_body(n_body, n_body,
                                                          librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> dense_wing(n_body, 3,
                                                          librpa_int::MAJOR::COL);
    for (int i = 0; i != n_body; ++i)
    {
        dense_body(i, i) = -0.15 - 0.002 * i;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            dense_wing(i, alpha) =
                std::complex<double>(0.003 * (i + 1) * (alpha + 1),
                                     0.002 * (i - 2 * alpha));
        }
        if (i + 1 < n_body)
        {
            const std::complex<double> coupling(0.004 * (i + 1), 0.001 * (i + 2));
            dense_body(i, i + 1) = coupling;
            dense_body(i + 1, i) = std::conj(coupling);
        }
    }

    auto distributed_schur_correction =
        [&](const librpa_int::matrix_m<std::complex<double>> &body_dense,
            const librpa_int::matrix_m<std::complex<double>> &wing_dense)
    {
        auto body_dist = librpa_int::init_local_mat<std::complex<double>>(
            desc_schur_body, librpa_int::MAJOR::COL);
        auto wing_dist = librpa_int::init_local_mat<std::complex<double>>(
            desc_schur_wing, librpa_int::MAJOR::COL);
        for (int iloc_body = 0; iloc_body != body_dist.nr(); ++iloc_body)
        {
            const int i = desc_schur_body.indx_l2g_r(iloc_body);
            for (int jloc_body = 0; jloc_body != body_dist.nc(); ++jloc_body)
            {
                const int j = desc_schur_body.indx_l2g_c(jloc_body);
                body_dist(iloc_body, jloc_body) = body_dense(i, j);
            }
        }
        for (int iloc_wing = 0; iloc_wing != wing_dist.nr(); ++iloc_wing)
        {
            const int i = desc_schur_wing.indx_l2g_r(iloc_wing);
            for (int jloc_wing = 0; jloc_wing != wing_dist.nc(); ++jloc_wing)
            {
                const int alpha = desc_schur_wing.indx_l2g_c(jloc_wing);
                wing_dist(iloc_wing, jloc_wing) = wing_dense(i, alpha);
            }
        }

        body_dist *= -1.0;
        for (int i = 0; i != n_body; ++i)
        {
            const int iloc_body = desc_schur_body.indx_g2l_r(i);
            const int jloc_body = desc_schur_body.indx_g2l_c(i);
            if (iloc_body >= 0 && jloc_body >= 0) body_dist(iloc_body, jloc_body) += 1.0;
        }
        librpa_int::invert_scalapack(body_dist, desc_schur_body);

        librpa_int::ArrayDesc desc_lam_3(blacs_h);
        desc_lam_3.init_square_blk(n_body, 3, 0, 0);
        librpa_int::ArrayDesc desc_3_3(blacs_h);
        desc_3_3.init_square_blk(3, 3, 0, 0);
        auto lam_3 = librpa_int::init_local_mat<std::complex<double>>(
            desc_lam_3, librpa_int::MAJOR::COL);
        auto correction_dist = librpa_int::init_local_mat<std::complex<double>>(
            desc_3_3, librpa_int::MAJOR::COL);
        librpa_int::ScalapackConnector::pgemm_f(
            'N', 'N', n_body, 3, n_body, std::complex<double>(1.0), body_dist.ptr(), 1, 1,
            desc_schur_body.desc, wing_dist.ptr(), 1, 1, desc_schur_wing.desc,
            std::complex<double>(0.0), lam_3.ptr(), 1, 1, desc_lam_3.desc);
        librpa_int::ScalapackConnector::pgemm_f(
            'C', 'N', 3, 3, n_body, std::complex<double>(1.0), wing_dist.ptr(), 1, 1,
            desc_schur_wing.desc, lam_3.ptr(), 1, 1, desc_lam_3.desc,
            std::complex<double>(0.0), correction_dist.ptr(), 1, 1, desc_3_3.desc);

        librpa_int::matrix_m<std::complex<double>> correction(3, 3,
                                                               librpa_int::MAJOR::COL);
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            const int iloc_alpha = desc_3_3.indx_g2l_r(alpha);
            for (int beta = 0; beta != 3; ++beta)
            {
                const int jloc_beta = desc_3_3.indx_g2l_c(beta);
                std::complex<double> value = 0.0;
                if (iloc_alpha >= 0 && jloc_beta >= 0)
                    value = correction_dist(iloc_alpha, jloc_beta);
                MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                              MPI_COMM_WORLD);
                correction(alpha, beta) = value;
            }
        }
        return correction;
    };

    const auto correction = distributed_schur_correction(dense_body, dense_wing);
    librpa_int::matrix_m<std::complex<double>> phased_body(n_body, n_body,
                                                           librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> phased_wing(n_body, 3,
                                                           librpa_int::MAJOR::COL);
    for (int i = 0; i != n_body; ++i)
    {
        const double phase_i = i % 2 == 0 ? 1.0 : -1.0;
        for (int j = 0; j != n_body; ++j)
        {
            const double phase_j = j % 2 == 0 ? 1.0 : -1.0;
            phased_body(i, j) = phase_i * dense_body(i, j) * phase_j;
        }
        for (int alpha = 0; alpha != 3; ++alpha)
            phased_wing(i, alpha) = phase_i * dense_wing(i, alpha);
    }
    const auto phased_correction = distributed_schur_correction(phased_body, phased_wing);
    for (int alpha = 0; alpha != 3; ++alpha)
        for (int beta = 0; beta != 3; ++beta)
            assert(std::abs(correction(alpha, beta) - phased_correction(alpha, beta)) < 2.0e-12);

    // Degenerate eigenspaces may be returned with a different unitary basis by
    // serial LAPACK and distributed ScaLAPACK.  Rotate two regular channels by
    // a genuinely complex unitary matrix and require the Schur invariant to
    // remain unchanged when body and wing are rotated together.
    librpa_int::matrix_m<std::complex<double>> regular_unitary(n_body, n_body,
                                                               librpa_int::MAJOR::COL);
    for (int i = 0; i != n_body; ++i) regular_unitary(i, i) = 1.0;
    constexpr int rotate_i = 3;
    constexpr int rotate_j = 11;
    const double rotation_angle = 0.37;
    const double rotation_phase = 0.41;
    const double rotation_cos = std::cos(rotation_angle);
    const std::complex<double> rotation_sin_phase =
        std::sin(rotation_angle) * std::exp(std::complex<double>(0.0, rotation_phase));
    regular_unitary(rotate_i, rotate_i) = rotation_cos;
    regular_unitary(rotate_j, rotate_j) = rotation_cos;
    regular_unitary(rotate_i, rotate_j) = rotation_sin_phase;
    regular_unitary(rotate_j, rotate_i) = -std::conj(rotation_sin_phase);
    const auto rotated_body =
        regular_unitary * dense_body * regular_unitary.get_transpose(true);
    const auto rotated_wing = regular_unitary * dense_wing;
    const auto rotated_correction = distributed_schur_correction(rotated_body, rotated_wing);
    for (int alpha = 0; alpha != 3; ++alpha)
        for (int beta = 0; beta != 3; ++beta)
            assert(std::abs(correction(alpha, beta) - rotated_correction(alpha, beta)) < 2.0e-12);

    // Reproduce the entire production chain with descriptors that differ:
    // Coulomb diagonalization -> sqrt(V) chi0 sqrt(V) -> wing projection ->
    // body extraction -> Schur correction.  The dense reference is built from
    // the known Coulomb eigensystem and is independent of PDSYEV row phases.
    librpa_int::matrix_m<std::complex<double>> dense_coulomb(n_abf, n_abf,
                                                             librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> dense_chi0(n_abf, n_abf,
                                                          librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> dense_wing_mu(n_abf, 3,
                                                             librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> known_sqrt_eigenvectors(
        n_abf, n_abf, librpa_int::MAJOR::COL);
    constexpr int dense_mix_block = n_abf / 4;
    constexpr double orthogonal_mix[4][4] = {
        {0.5, 0.5, 0.5, 0.5},
        {0.5, -0.5, 0.5, -0.5},
        {0.5, 0.5, -0.5, -0.5},
        {0.5, -0.5, -0.5, 0.5},
    };
    const auto coulomb_eigenvalue_for_mode = [](const int mode)
    {
        if (mode < 65) return -1.0 - 0.002 * mode;
        if (mode == n_abf - 1) return 8.0;
        return 1.0 + 0.004 * mode;
    };
    for (int iblock = 0; iblock != dense_mix_block; ++iblock)
    {
        int indices[4] = {iblock, iblock + dense_mix_block, iblock + 2 * dense_mix_block,
                          iblock + 3 * dense_mix_block};
        for (int irow = 0; irow != 4; ++irow)
        {
            for (int jrow = 0; jrow != 4; ++jrow)
            {
                double value = 0.0;
                for (int imode = 0; imode != 4; ++imode)
                {
                    const double eigenvalue = coulomb_eigenvalue_for_mode(indices[imode]);
                    value += orthogonal_mix[irow][imode] * eigenvalue *
                             orthogonal_mix[jrow][imode];
                }
                dense_coulomb(indices[irow], indices[jrow]) = value;
            }
            for (int imode = 0; imode != 4; ++imode)
            {
                const int mode = indices[imode];
                const double eigenvalue = coulomb_eigenvalue_for_mode(mode);
                if (eigenvalue <= 0.0) continue;
                known_sqrt_eigenvectors(indices[irow], n_abf - 1 - mode) =
                    orthogonal_mix[irow][imode] * std::sqrt(eigenvalue);
            }
        }
    }

    for (int i = 0; i != n_abf; ++i)
    {
        dense_chi0(i, i) = -0.02 - 0.00003 * i;
        if (i + 1 < n_abf)
        {
            const std::complex<double> coupling(2.0e-4, 5.0e-5);
            dense_chi0(i, i + 1) = coupling;
            dense_chi0(i + 1, i) = std::conj(coupling);
        }
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            dense_wing_mu(i, alpha) =
                std::complex<double>(2.0e-4 * (i + 1) * (alpha + 1), 1.0e-4 * (i - alpha));
        }
    }

    auto coulomb_dist =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);
    auto chi0_dist =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);
    for (int iloc_abf = 0; iloc_abf != coulomb_dist.nr(); ++iloc_abf)
    {
        const int i = desc_optimized.indx_l2g_r(iloc_abf);
        for (int jloc_abf = 0; jloc_abf != coulomb_dist.nc(); ++jloc_abf)
        {
            const int j = desc_optimized.indx_l2g_c(jloc_abf);
            coulomb_dist(iloc_abf, jloc_abf) = dense_coulomb(i, j);
            chi0_dist(iloc_abf, jloc_abf) = dense_chi0(i, j);
        }
    }

    auto coulomb_eigenvectors =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);
    std::vector<double> coulomb_eigenvalues(n_abf);
    std::size_t n_filtered = 0;
    auto sqrt_eigenvectors = librpa_int::power_hemat_blacs_real(
        coulomb_dist, desc_optimized, coulomb_eigenvectors, desc_optimized, n_filtered,
        coulomb_eigenvalues.data(), 0.5, 0.0);
    assert(n_filtered == 65);
    const int n_nonsingular = n_abf - static_cast<int>(n_filtered);

    auto chi0_times_sqrt =
        librpa_int::init_local_mat<std::complex<double>>(desc_optimized, librpa_int::MAJOR::COL);
    librpa_int::ArrayDesc desc_projected(blacs_h);
    desc_projected.init_square_blk(n_nonsingular, n_nonsingular, 0, 0);
    auto projected_response =
        librpa_int::init_local_mat<std::complex<double>>(desc_projected, librpa_int::MAJOR::COL);
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', n_abf, n_nonsingular, n_abf, std::complex<double>(1.0), chi0_dist.ptr(), 1, 1,
        desc_optimized.desc, sqrt_eigenvectors.ptr(), 1, 1, desc_optimized.desc,
        std::complex<double>(0.0), chi0_times_sqrt.ptr(), 1, 1, desc_optimized.desc);
    librpa_int::ScalapackConnector::pgemm_f('C', 'N', n_nonsingular, n_nonsingular, n_abf,
                                            std::complex<double>(1.0), sqrt_eigenvectors.ptr(), 1,
                                            1, desc_optimized.desc, chi0_times_sqrt.ptr(), 1, 1,
                                            desc_optimized.desc, std::complex<double>(0.0),
                                            projected_response.ptr(), 1, 1, desc_projected.desc);

    librpa_int::ArrayDesc desc_wing_mu(blacs_h);
    desc_wing_mu.init_square_blk(n_abf, 3, 0, 0);
    auto wing_mu_dist =
        librpa_int::init_local_mat<std::complex<double>>(desc_wing_mu, librpa_int::MAJOR::COL);
    for (int iloc_mu = 0; iloc_mu != wing_mu_dist.nr(); ++iloc_mu)
    {
        const int mu = desc_wing_mu.indx_l2g_r(iloc_mu);
        for (int jloc_alpha = 0; jloc_alpha != wing_mu_dist.nc(); ++jloc_alpha)
        {
            const int alpha = desc_wing_mu.indx_l2g_c(jloc_alpha);
            wing_mu_dist(iloc_mu, jloc_alpha) = dense_wing_mu(mu, alpha);
        }
    }

    const int n_projected_body = n_nonsingular - 1;
    librpa_int::ArrayDesc desc_projected_body(blacs_h);
    desc_projected_body.init_square_blk(n_projected_body, n_projected_body, 0, 0);
    librpa_int::ArrayDesc desc_projected_wing_seed(blacs_h);
    desc_projected_wing_seed.init_square_blk(n_projected_body, 3, 0, 0);
    librpa_int::ArrayDesc desc_projected_wing(blacs_h);
    desc_projected_wing.init(n_projected_body, 3, desc_projected_body.mb(),
                             desc_projected_wing_seed.nb(), 0, 0);
    auto projected_wing = librpa_int::init_local_mat<std::complex<double>>(desc_projected_wing,
                                                                           librpa_int::MAJOR::COL);
    const auto production_schur_wing_desc = librpa_int::make_rpa_chi0v_wing_desc(
        desc_projected_body, 0, projected_wing.nr(), projected_wing.nc());
    assert(std::equal(production_schur_wing_desc.desc, production_schur_wing_desc.desc + 9,
                      desc_projected_wing.desc));
    assert(production_schur_wing_desc.l2g_r() == desc_projected_wing.l2g_r());
    assert(production_schur_wing_desc.l2g_c() == desc_projected_wing.l2g_c());
    librpa_int::ScalapackConnector::pgemm_f(
        'C', 'N', n_projected_body, 3, n_abf, std::complex<double>(1.0), sqrt_eigenvectors.ptr(), 1,
        2, desc_optimized.desc, wing_mu_dist.ptr(), 1, 1, desc_wing_mu.desc,
        std::complex<double>(0.0), projected_wing.ptr(), 1, 1, desc_projected_wing.desc);

    auto projected_body = librpa_int::init_local_mat<std::complex<double>>(desc_projected_body,
                                                                           librpa_int::MAJOR::COL);
    librpa_int::ScalapackConnector::pgemr2d_f(
        n_projected_body, n_projected_body, projected_response.ptr(), 2, 2, desc_projected.desc,
        projected_body.ptr(), 1, 1, desc_projected_body.desc, blacs_h.ictxt);
    projected_body *= -1.0;
    for (int i = 0; i != n_projected_body; ++i)
    {
        const int iloc_body = desc_projected_body.indx_g2l_r(i);
        const int jloc_body = desc_projected_body.indx_g2l_c(i);
        if (iloc_body >= 0 && jloc_body >= 0) projected_body(iloc_body, jloc_body) += 1.0;
    }
    librpa_int::invert_scalapack(projected_body, desc_projected_body);

    librpa_int::ArrayDesc desc_projected_lam_3(blacs_h);
    desc_projected_lam_3.init_square_blk(n_projected_body, 3, 0, 0);
    librpa_int::ArrayDesc desc_projected_3_3(blacs_h);
    desc_projected_3_3.init_square_blk(3, 3, 0, 0);
    auto projected_lam_3 = librpa_int::init_local_mat<std::complex<double>>(desc_projected_lam_3,
                                                                            librpa_int::MAJOR::COL);
    auto projected_correction_dist = librpa_int::init_local_mat<std::complex<double>>(
        desc_projected_3_3, librpa_int::MAJOR::COL);
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', n_projected_body, 3, n_projected_body, std::complex<double>(1.0),
        projected_body.ptr(), 1, 1, desc_projected_body.desc, projected_wing.ptr(), 1, 1,
        production_schur_wing_desc.desc, std::complex<double>(0.0), projected_lam_3.ptr(), 1, 1,
        desc_projected_lam_3.desc);
    librpa_int::ScalapackConnector::pgemm_f(
        'C', 'N', 3, 3, n_projected_body, std::complex<double>(1.0), projected_wing.ptr(), 1, 1,
        production_schur_wing_desc.desc, projected_lam_3.ptr(), 1, 1, desc_projected_lam_3.desc,
        std::complex<double>(0.0), projected_correction_dist.ptr(), 1, 1, desc_projected_3_3.desc);

    librpa_int::matrix_m<std::complex<double>> projected_correction(3, 3, librpa_int::MAJOR::COL);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        const int iloc_alpha = desc_projected_3_3.indx_g2l_r(alpha);
        for (int beta = 0; beta != 3; ++beta)
        {
            const int jloc_beta = desc_projected_3_3.indx_g2l_c(beta);
            std::complex<double> value = 0.0;
            if (iloc_alpha >= 0 && jloc_beta >= 0)
                value = projected_correction_dist(iloc_alpha, jloc_beta);
            MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
            projected_correction(alpha, beta) = value;
        }
    }

    const auto expected_response =
        known_sqrt_eigenvectors.get_transpose(true) * dense_chi0 * known_sqrt_eigenvectors;
    const auto expected_wing_full = known_sqrt_eigenvectors.get_transpose(true) * dense_wing_mu;
    librpa_int::matrix_m<std::complex<double>> expected_identity_minus_body(
        n_projected_body, n_projected_body, librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> expected_wing(n_projected_body, 3,
                                                             librpa_int::MAJOR::COL);
    for (int i = 0; i != n_projected_body; ++i)
    {
        for (int j = 0; j != n_projected_body; ++j)
            expected_identity_minus_body(i, j) =
                (i == j ? 1.0 : 0.0) - expected_response(i + 1, j + 1);
        for (int alpha = 0; alpha != 3; ++alpha)
            expected_wing(i, alpha) = expected_wing_full(i + 1, alpha);
    }
    auto expected_body_inverse = expected_identity_minus_body.copy();
    std::vector<int> expected_pivots(n_projected_body);
    std::vector<std::complex<double>> expected_work(n_projected_body);
    int expected_info = 0;
    librpa_int::LapackConnector::getrf_f(
        n_projected_body, n_projected_body, expected_body_inverse.ptr(), n_projected_body,
        expected_pivots.data(), expected_info);
    assert(expected_info == 0);
    librpa_int::LapackConnector::getri_f(
        n_projected_body, expected_body_inverse.ptr(), n_projected_body,
        expected_pivots.data(), expected_work.data(), n_projected_body, expected_info);
    assert(expected_info == 0);
    const auto expected_correction =
        expected_wing.get_transpose(true) * expected_body_inverse * expected_wing;
    for (int alpha = 0; alpha != 3; ++alpha)
        for (int beta = 0; beta != 3; ++beta)
            assert(std::abs(projected_correction(alpha, beta) - expected_correction(alpha, beta)) <
                   2.0e-10);

    kdist_kblacs.finalize();
    wing_kblacs.finalize();
    blacs_h.exit();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
