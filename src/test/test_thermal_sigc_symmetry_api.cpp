#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../api/instance_manager.h"
#include "librpa.hpp"

namespace
{
using namespace librpa_int;
using Z = std::complex<double>;

std::vector<Z> run(int symmetry_flags, int nk, bool ibz_input = false,
                  bool irreducible_outputs = false)
{
    librpa::Handler h(MPI_COMM_WORLD);
    const auto ds = api::get_dataset_instance(h);
    const double lattice[]{4, 0, 0, 0, 5, 0, 0, 0, 6};
    const double pi = std::acos(-1.0);
    const double reciprocal[]{2 * pi / 4, 0, 0, 0, 2 * pi / 5, 0, 0, 0, 2 * pi / 6};
    h.set_latvec_and_G(lattice, reciprocal);
    h.set_atoms({0, 0}, {1, 0, 0, 3, 0, 0});
    h.set_ao_basis_wfc({1, 1}, {{0}, {0}});
    h.set_ao_basis_aux({1, 1}, {{0}, {0}});
    h.set_basis_convention(-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M,
                           LIBRPA_RSH_COEFF_1_M);
    const int rotations[]{1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1};
    const double translations[]{0, 0, 0, 1, 0, 0};
    h.set_symmetry_operations(2, 1, rotations, translations);
    std::vector<int> loaded_full_indices;
    if (ibz_input)
    {
        if (nk != 3)
            throw std::runtime_error("IBZ thermal GW test requires the three-point mesh");
        loaded_full_indices = {0, 1};
    }
    else
    {
        for (int ik = 0; ik < nk; ++ik) loaded_full_indices.push_back(ik);
    }
    std::vector<double> kpoints, energies, occupations, kweights;
    constexpr double BETA = 8, MU = 0.1;
    for (const int ik_full : loaded_full_indices)
    {
        kpoints.insert(kpoints.end(), {2 * pi * ik_full / (4 * nk), 0, 0});
        const double weight = 1.0 / nk;
        kweights.push_back(ibz_input && ik_full == 1 ? 2 * weight : weight);
        for (double e : {-0.2, 0.4})
        {
            energies.push_back(e + 0.03 * std::cos(2 * pi * ik_full / nk));
            occupations.push_back(2.0 / (1 + std::exp(BETA * (energies.back() - MU))));
        }
    }
    if (ibz_input)
        h.set_kgrids_kvec(nk, 1, 1, kpoints, kweights);
    else
        h.set_kgrids_kvec(nk, 1, 1, kpoints);
    const int n_loaded_kpoints = static_cast<int>(loaded_full_indices.size());
    if (ibz_input && (ds->pbc.klist.size() != 2 || ds->pbc.klist_full.size() != 3))
        throw std::runtime_error("IBZ k-grid input did not retain the complete BvK mesh");
    h.set_scf_dimension(1, n_loaded_kpoints, 2, 2);
    h.set_wg_ekb_efermi(1, n_loaded_kpoints, 2, occupations.data(), energies.data(), MU);
    h.set_fermi_dirac_reference(1 / BETA, MU, 2, 1e-12);
    const double a = 1 / std::sqrt(2.0);
    const Z orbitals[]{a, a, -a, a};
    for (int ik = 0; ik < n_loaded_kpoints; ++ik) h.set_wfc_packed(0, ik, 2, 2, orbitals);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        const int r[]{0, 0, 0};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
            {
                const double c = i == j ? 0.2 : 0.035;
                h.set_lri_coeff(LIBRPA_ROUTING_LIBRI, i, j, 1, 1, 1, r, &c);
                if (j < i) continue;
                const Z full = i == j ? 0.7 : 0.08;
                const Z cut = i == j ? 0.5 : 0.04;
                for (int ik = 0; ik < n_loaded_kpoints; ++ik)
                {
                    const double scale =
                        1 + 0.2 * std::cos(2 * pi * loaded_full_indices[ik] / nk);
                    const Z v = full * scale, vc = cut * scale;
                    h.set_aux_bare_coulomb_k_atom_pair_packed(ik, i, j, 1, 1, &v, 0);
                    h.set_aux_cut_coulomb_k_atom_pair_packed(ik, i, j, 1, 1, &vc, 0);
                }
            }
    }
    const std::vector<double> times{1, 3, 5, 7}, weights(4, 2);
    const std::vector<int> bosons{0, 1, -1, 2, -2}, fermions{0, -1, 1, -2, 2, -3, 3, -4};
    const auto transform =
        ThermalGWTransform::from_quadrature(BETA, times, weights, bosons, fermions);
    const auto b = transform.copy_bosonic_frequency_to_time();
    const auto f = transform.copy_fermionic_time_to_frequency();
    std::vector<double> br(b.size), bi(b.size), fr(f.size), fi(f.size);
    for (int i = 0; i < b.size; ++i)
    {
        br[i] = b.c[i].real();
        bi[i] = b.c[i].imag();
    }
    for (int i = 0; i < f.size; ++i)
    {
        fr[i] = f.c[i].real();
        fi[i] = f.c[i].imag();
    }
    h.set_external_thermal_gw_grid(BETA, 1, 2, 3, 1e-10, 4, 5, 8, times.data(), bosons.data(),
                                   fermions.data(), br.data(), bi.data(), fr.data(), fi.data());
    librpa::Options opts;
    opts.parallel_routing = LIBRPA_ROUTING_LIBRI;
    opts.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    opts.nfreq = 3;
    opts.ntau = 4;
    opts.use_symmetry_exx = symmetry_flags & 1 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    opts.use_symmetry_rpa = symmetry_flags & 2 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    opts.use_symmetry_gw = symmetry_flags & 4 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    opts.use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;
    opts.use_shrink_abfs = opts.use_shrink_chi = LIBRPA_SWITCH_OFF;
    opts.replace_w_head = LIBRPA_SWITCH_OFF;
    opts.use_scalapack_gw_wc = LIBRPA_SWITCH_ON;
    opts.use_cholesky_gw_wc = LIBRPA_SWITCH_OFF;
    opts.use_fullcoul_eps = LIBRPA_SWITCH_ON;
    opts.use_fullcoul_wc = opts.use_fullcoul_exx = LIBRPA_SWITCH_OFF;
    opts.libri_chi0_threshold_C = opts.libri_chi0_threshold_G = 0;
    opts.libri_g0w0_threshold_C = opts.libri_g0w0_threshold_G = opts.libri_g0w0_threshold_Wc = 0;
    opts.libri_exx_threshold_C = opts.libri_exx_threshold_D = opts.libri_exx_threshold_V = 0;
    h.build_g0w0_sigma(opts);
    if (nk == 3 && (symmetry_flags & 2) && ds->p_chi0->qpoint_view().representatives.size() != 2)
        throw std::runtime_error("q symmetry test did not reduce three q points to two");
    ds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(ds->blacs_h);
    std::vector<int> iks;
    if (irreducible_outputs && nk == 3)
        iks = {0, 1};
    else
        for (int ik = 0; ik < n_loaded_kpoints; ++ik) iks.push_back(ik);
    std::vector<Z> result;
    if (rank == 0)
        for (const int ik : iks)
            for (const auto &[frequency, diagonal] : ds->p_g0w0->sigc_diag_is_ik_f_KS.at(0).at(ik))
                result.insert(result.end(), diagonal.begin(), diagonal.end());
    opts.n_params_anacon = 4;
    opts.option_qpe_solver = 2;
    const auto exx = h.get_exx_pot_kgrid(opts, 1, iks, 0, 2);
    const auto qp = h.get_g0w0_qpe_kgrid(
        opts, 1, iks, 0, 2, std::vector<double>(2 * iks.size(), 0), exx);
    if (rank == 0)
    {
        for (double energy : exx) result.push_back(energy);
        for (double energy : qp.eqp) result.push_back(energy);
    }
    return result;
}
}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, status = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try
    {
        for (int nk : {1, 3})
        {
            const auto full = run(0, nk);
            for (int flags : {1, 2, 4, 3, 7})
            {
                const auto reduced = run(flags, nk);
                double error = 0, qp_error = 0, exx_error = 0, magnitude = 0;
                if (rank == 0)
                {
                    if (full.size() != 12 * nk || reduced.size() != full.size()) status = 1;
                    if (!status)
                        for (std::size_t i = 0; i < full.size(); ++i)
                        {
                            if (!std::isfinite(reduced[i].real()) ||
                                !std::isfinite(reduced[i].imag()))
                                status = 1;
                            if (i < 8 * nk)
                            {
                                error = std::max(error, std::abs(full[i] - reduced[i]));
                                magnitude = std::max(magnitude, std::abs(full[i]));
                            }
                            else if (i < 10 * nk)
                                exx_error = std::max(exx_error, std::abs(full[i] - reduced[i]));
                            else
                                qp_error = std::max(qp_error, std::abs(full[i] - reduced[i]));
                        }
                    if (error > 2e-12 || exx_error > 2e-12 || qp_error > 1e-10 || magnitude < 1e-8)
                        status = 1;
                    std::cout << "THERMAL_SIGC_SYMMETRY_API nk=" << nk << " flags=" << flags
                              << " max_sigma_error=" << error << " max_exx_error=" << exx_error
                              << " max_qp_error=" << qp_error << " max_sigma=" << magnitude << '\n';
                }
            }
        }
        const auto full = run(0, 3, false, true);
        const auto ibz = run(7, 3, true, true);
        double sigma_error = 0, exx_error = 0, qp_error = 0;
        if (rank == 0)
        {
            if (full.size() != 24 || ibz.size() != full.size()) status = 1;
            if (!status)
                for (std::size_t i = 0; i < full.size(); ++i)
                {
                    if (!std::isfinite(ibz[i].real()) || !std::isfinite(ibz[i].imag()))
                        status = 1;
                    if (i < 16)
                        sigma_error = std::max(sigma_error, std::abs(full[i] - ibz[i]));
                    else if (i < 20)
                        exx_error = std::max(exx_error, std::abs(full[i] - ibz[i]));
                    else
                        qp_error = std::max(qp_error, std::abs(full[i] - ibz[i]));
                }
            if (sigma_error > 2e-12 || exx_error > 2e-12 || qp_error > 1e-10) status = 1;
            std::cout << "THERMAL_SIGC_IBZ_API max_sigma_error=" << sigma_error
                      << " max_exx_error=" << exx_error << " max_qp_error=" << qp_error << '\n';
            if (status)
                for (std::size_t i = 16; i < full.size(); ++i)
                    std::cout << "THERMAL_SIGC_IBZ_VALUE i=" << i << " full=" << full[i]
                              << " ibz=" << ibz[i] << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
