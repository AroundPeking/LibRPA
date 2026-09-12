#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"

namespace
{
using namespace librpa_int;
using Z = std::complex<double>;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void check_public_route()
{
    librpa::Handler h(MPI_COMM_WORLD);
    const auto ds = api::get_dataset_instance(h);
    const double lattice[]{3, 0, 0, 0.4, 4, 0, 0.2, 0.3, 5};
    double reciprocal[]{1.0 / 3, -1.0 / 30, -0.17 / 15, 0, 0.25, -0.015, 0, 0, 0.2};
    for (double &value : reciprocal) value *= 2 * std::acos(-1.0);
    h.set_latvec_and_G(lattice, reciprocal);
    h.set_atoms({0, 1}, {0, 0, 0, 0.8, 0.7, 0.6});
    h.set_ao_basis_wfc({1, 2});
    h.set_ao_basis_aux({1, 1});
    h.set_kgrids_kvec(1, 1, 1, std::vector<double>{0, 0, 0});
    h.set_scf_dimension(1, 1, 3, 3);
    const double energies[]{-0.4, 0.1, 0.6};
    double occupations[3];
    constexpr double beta = 8, mu = 0.1;
    for (int n = 0; n < 3; ++n) occupations[n] = 2 / (1 + std::exp(beta * (energies[n] - mu)));
    h.set_wg_ekb_efermi(1, 1, 3, occupations, energies, mu);
    h.set_fermi_dirac_reference(1 / beta, mu, 2, 1e-12);
    const double a = 1 / std::sqrt(2.0);
    const Z wfc[]{a, a, 0, -a, a, 0, 0, 0, 1};
    h.set_wfc_packed(0, 0, 3, 3, wfc);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        const int r[]{0, 0, 0};
        const double c0[]{0.2}, c1[]{0.17, 0.03, 0.03, 0.11};
        h.set_lri_coeff(LIBRPA_ROUTING_LIBRI, 0, 0, 1, 1, 1, r, c0);
        h.set_lri_coeff(LIBRPA_ROUTING_LIBRI, 1, 1, 2, 2, 1, r, c1);
        for (int i = 0; i < 2; ++i)
            for (int j = i; j < 2; ++j)
            {
                const Z full = i == j ? 0.7 : 0.08;
                const Z cut = i == j ? 0.5 : 0.04;
                h.set_aux_bare_coulomb_k_atom_pair_packed(0, i, j, 1, 1, &full, 0);
                h.set_aux_cut_coulomb_k_atom_pair_packed(0, i, j, 1, 1, &cut, 0);
            }
    }
    const std::vector<double> times{1, 3, 5, 7}, weights(4, 2);
    const std::vector<int> bosons{0, 1, -1, 2, -2}, fermions{0, -1, 1, -2, 2, -3, 3, -4};
    const auto transform =
        ThermalGWTransform::from_quadrature(beta, times, weights, bosons, fermions);
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
    h.set_external_thermal_gw_grid(beta, 1, 2, 3, 1e-10, 4, 5, 8, times.data(), bosons.data(),
                                   fermions.data(), br.data(), bi.data(), fr.data(), fi.data());
    librpa::Options opts;
    opts.parallel_routing = LIBRPA_ROUTING_LIBRI;
    opts.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    opts.nfreq = 3;
    opts.ntau = 4;
    opts.use_symmetry_exx = opts.use_symmetry_rpa = opts.use_symmetry_gw = LIBRPA_SWITCH_OFF;
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
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto original_size = ds->basis_wfc.nb_total;
    if (rank == size - 1) ds->basis_wfc.nb_total = 99;
    int preflight_caught = 0;
    try
    {
        h.build_g0w0_sigma(opts);
    }
    catch (const std::exception &error)
    {
        preflight_caught =
            std::string(error.what()).find("Finite-temperature GW") != std::string::npos;
    }
    ds->basis_wfc.nb_total = original_size;
    MPI_Allreduce(MPI_IN_PLACE, &preflight_caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    require(preflight_caught == size && !ds->p_g0w0,
            "rank-local basis mismatch did not fail collectively before GW allocation");
    h.build_g0w0_sigma(opts);
    require(ds->p_g0w0 && ds->p_g0w0->is_rspace_built(),
            "generic thermal public API did not produce self-energy");
    require(ds->tfg.get_freq_nodes().size() == 3 && ds->tfg.get_freq_nodes()[0] == 0,
            "self-energy replaced the bosonic response grid");
    ds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(ds->blacs_h);
    double largest = 0;
    std::string message;
    if (rank == 0)
    {
        try
        {
            const auto &frequency_map = ds->p_g0w0->sigc_diag_is_ik_f_KS.at(0).at(0);
            require(frequency_map.size() == 4, "KS output did not use the fermionic grid");
            for (const auto &[frequency, diagonal] : frequency_map)
            {
                require(frequency > 0 && diagonal.size() == 3, "invalid KS self-energy shape");
                for (Z value : diagonal)
                {
                    require(std::isfinite(value.real()) && std::isfinite(value.imag()),
                            "nonfinite generic thermal self-energy");
                    largest = std::max(largest, std::abs(value));
                }
            }
            require(largest > 1e-8, "generic thermal path silently returned zero");
        }
        catch (const std::exception &error)
        {
            message = error.what();
        }
    }
    int failed = !message.empty();
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(!failed, message.empty() ? "root KS self-energy check failed" : message.c_str());
    opts.n_params_anacon = 4;
    opts.option_qpe_solver = 2;
    const auto qp = h.get_g0w0_qpe_kgrid(opts, 1, {0}, 0, 3, std::vector<double>(3, 0),
                                         std::vector<double>(3, 0));
    require(qp.eqp.size() == 3 && qp.sigc.size() == 3, "wrong public thermal QP result size");
    for (int n = 0; n < 3; ++n)
        require(std::isfinite(qp.eqp[n]) && std::isfinite(qp.sigc[n].real()) &&
                    std::isfinite(qp.sigc[n].imag()),
                "nonfinite public thermal QP result");
    const double gamma[]{0, 0, 0};
    h.set_band_kvec(1, gamma);
    h.set_band_occ_eigval(1, 1, 3, occupations, energies);
    h.set_wfc_band_packed(0, 0, 3, 3, wfc);
    const auto band_qp = h.get_g0w0_qpe_band_k(opts, 1, {0}, 0, 3, std::vector<double>(3, 0),
                                               std::vector<double>(3, 0));
    double band_error = 0;
    for (int n = 0; n < 3; ++n)
    {
        require(std::isfinite(band_qp.eqp[n]), "nonfinite public thermal band QP result");
        band_error = std::max(band_error, std::abs(band_qp.eqp[n] - qp.eqp[n]));
        band_error = std::max(band_error, std::abs(band_qp.sigc[n] - qp.sigc[n]));
    }
    require(band_error < 1e-12, "thermal grid and band APIs differ at the same Gamma state");
    opts.anacon_nfreq = 6;
    int resample_caught = 0;
    try
    {
        (void)h.get_g0w0_qpe_band_k(opts, 1, {0}, 0, 3, std::vector<double>(3, 0),
                                    std::vector<double>(3, 0));
    }
    catch (const std::exception &error)
    {
        resample_caught =
            std::string(error.what()).find("legacy resampling is disabled") != std::string::npos;
    }
    require(resample_caught, "thermal band API accepted legacy continuation resampling");
    if (rank == 0) std::cout << "GENERIC_THERMAL_GW_API_PASS max_sigma=" << largest << '\n';
}
}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int status = 0;
    try
    {
        check_public_route();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
