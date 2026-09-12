#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"

namespace
{
using namespace librpa_int;
using Z = std::complex<double>;
using HeadTensor = std::array<Z, 9>;

constexpr int N_STATES = 2;
constexpr int N_FREQ = 3;
constexpr double BETA_1 = 8.0;
constexpr double BETA_2 = 16.0;
constexpr double MU = 0.0;
constexpr double VOLUME = 60.0;
constexpr double TOLERANCE = 1e-12;
constexpr std::array<double, N_STATES> ENERGIES{{-0.25, 0.25}};
// Three real Hermitian velocity matrices, in Cartesian/state/state order.
constexpr double VELOCITY[3][N_STATES][N_STATES] = {
    {{0.3, 0.4}, {0.4, -0.1}}, {{-0.2, 0.2}, {0.2, 0.25}}, {{0.1, -0.3}, {-0.3, 0.35}}};

void require_collective(bool condition, const std::string &message)
{
    int failed = !condition;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (failed) throw std::runtime_error(message);
}

bool finite(Z value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }

void set_reference(librpa::Handler &handler, double beta)
{
    std::array<double, N_STATES> occupations{{2.0, 0.0}};
    if (beta > 0.0)
    {
        handler.set_fermi_dirac_reference(1.0 / beta, MU, 2.0, TOLERANCE);
        for (int n = 0; n < N_STATES; ++n)
            occupations[n] = 2.0 / (1.0 + std::exp(beta * (ENERGIES[n] - MU)));
    }
    else
    {
        handler.clear_fermi_dirac_reference();
    }
    handler.set_wg_ekb_efermi(1, 1, N_STATES, occupations.data(), ENERGIES.data(), MU);
}

void set_bosonic_nodes(Dataset &ds, double beta, int mode_stride)
{
    constexpr int N_TIME = 16;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double dt = beta / N_TIME;
    std::vector<double> times(N_TIME), weights(N_FREQ, 1.0 / beta);
    std::vector<int> modes(N_FREQ);
    ComplexMatrix transform(N_FREQ, N_TIME);
    weights[0] *= 0.5;
    for (int itime = 0; itime < N_TIME; ++itime) times[itime] = (itime + 0.5) * dt;
    for (int ifreq = 0; ifreq < N_FREQ; ++ifreq)
    {
        modes[ifreq] = mode_stride * ifreq;
        for (int itime = 0; itime < N_TIME; ++itime)
            transform(ifreq, itime) =
                dt * std::exp(Z(0.0, two_pi * modes[ifreq] * times[itime] / beta));
    }
    // Deliberately bypass dataset invalidation: equal-size node changes must
    // be handled by initialize_ds_headwing itself. No RPA sum is evaluated.
    ds.tfg.reset(N_FREQ);
    ds.tfg.set_finite_beta_rpa_grid(times, beta, modes, weights, transform);
}

void initialize_fixture(librpa::Handler &handler, double beta, int mode_stride)
{
    const double lattice[]{3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0};
    double reciprocal[]{1.0 / 3.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.2};
    for (double &value : reciprocal) value *= 2.0 * std::acos(-1.0);
    handler.set_latvec_and_G(lattice, reciprocal);
    handler.set_atoms({0}, {0.0, 0.0, 0.0});
    handler.set_ao_basis_wfc({N_STATES});
    handler.set_ao_basis_aux({1});
    handler.set_kgrids_kvec(1, 1, 1, std::vector<double>{0.0, 0.0, 0.0});
    handler.set_scf_dimension(1, 1, N_STATES, N_STATES);
    set_reference(handler, beta);
    std::array<double, 3 * N_STATES * N_STATES> real{}, imag{};
    for (int alpha = 0; alpha < 3; ++alpha)
        for (int i = 0; i < N_STATES; ++i)
            for (int j = 0; j < N_STATES; ++j)
                real[(alpha * N_STATES + i) * N_STATES + j] = VELOCITY[alpha][i][j];
    // The public velocity setter clears p_headwing, so call it only at setup.
    handler.set_velocity_matrix(1, 1, N_STATES, real.data(), imag.data());
    const auto ds = api::get_dataset_instance(handler);
    // The zero-T reference is evaluated at the same frequencies as the last
    // thermal model; this helper test does not invoke zero-T grid selection.
    set_bosonic_nodes(*ds, beta > 0.0 ? beta : BETA_2, mode_stride);
    // Head-only initialization needs no wave functions, RI C, or Coulomb blocks.
}

struct HeadSnapshot
{
    std::array<HeadTensor, N_FREQ> tensors;
    std::vector<double> epsmacs;
    std::vector<double> frequencies;
    bool fd_enabled;
    double kbt;
    bool metallic_static;
};

HeadSnapshot build_head(Dataset &ds, const librpa::Options &options)
{
    initialize_ds_headwing(ds, options, false);
    require_collective(ds.p_headwing != nullptr, "analytic head cache was not created");
    require_collective(!ds.p_headwing->has_wing(), "head-only test unexpectedly built wings");
    require_collective(ds.p_headwing->get_head_vec().size() == N_FREQ &&
                           ds.epsmacs_imagfreq.size() == N_FREQ &&
                           ds.omegas_imagfreq.size() == N_FREQ,
                       "analytic head cache has the wrong frequency count");
    HeadSnapshot snapshot;
    snapshot.epsmacs = ds.epsmacs_imagfreq;
    snapshot.frequencies = ds.omegas_imagfreq;
    const auto &fd = ds.p_headwing->get_meanfield_df().get_fermi_dirac_reference();
    snapshot.fd_enabled = fd.enabled;
    snapshot.kbt = fd.kbt_ha;
    snapshot.metallic_static = ds.p_headwing->is_metallic_static_3d_frequency(0);
    bool all_finite = true;
    for (int ifreq = 0; ifreq < N_FREQ; ++ifreq)
    {
        const auto tensor = ds.p_headwing->get_rpa_chi0v_head(ifreq);
        require_collective(tensor.nr() == 3 && tensor.nc() == 3,
                           "analytic head tensor is not 3 by 3");
        all_finite = all_finite && std::isfinite(snapshot.epsmacs[ifreq]) &&
                     std::isfinite(snapshot.frequencies[ifreq]);
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int beta = 0; beta < 3; ++beta)
            {
                const Z value = tensor(alpha, beta);
                snapshot.tensors[ifreq][3 * alpha + beta] = value;
                all_finite = all_finite && finite(value);
            }
    }
    require_collective(all_finite, "analytic head cache contains nonfinite data");
    return snapshot;
}

HeadTensor expected_head(double beta, double frequency)
{
    std::array<double, N_STATES> f{{1.0, 0.0}}, minus_df{};
    if (beta > 0.0)
        for (int n = 0; n < N_STATES; ++n)
        {
            f[n] = 1.0 / (1.0 + std::exp(beta * (ENERGIES[n] - MU)));
            minus_df[n] = beta * f[n] * (1.0 - f[n]);
        }
    const double gap = ENERGIES[1] - ENERGIES[0];
    // chi0*v = -(epsilon-I), with 4*pi/volume and scalar-spin degeneracy 2.
    const double prefactor = -8.0 * std::acos(-1.0) / VOLUME;
    HeadTensor tensor{};
    for (int alpha = 0; alpha < 3; ++alpha)
        for (int cart = 0; cart < 3; ++cart)
        {
            double response = 2.0 * (f[0] - f[1]) * VELOCITY[alpha][1][0] * VELOCITY[cart][0][1] /
                              (gap * (gap * gap + frequency * frequency));
            if (frequency > 0.0)
                for (int n = 0; n < N_STATES; ++n)
                    response += minus_df[n] * VELOCITY[alpha][n][n] * VELOCITY[cart][n][n] /
                                (frequency * frequency);
            tensor[3 * alpha + cart] = prefactor * response;
        }
    return tensor;
}

void check_reference(const HeadSnapshot &snapshot, double beta, int mode_stride)
{
    bool correct = true;
    for (int ifreq = 0; ifreq < N_FREQ; ++ifreq)
    {
        const double frequency =
            2.0 * std::acos(-1.0) * mode_stride * ifreq / (beta > 0.0 ? beta : BETA_2);
        const auto expected = expected_head(beta, frequency);
        for (int entry = 0; entry < 9; ++entry)
            correct = correct && std::abs(snapshot.tensors[ifreq][entry] - expected[entry]) <=
                                     TOLERANCE * (1.0 + std::abs(expected[entry]));
        const double epsmac = 1.0 - (expected[0] + expected[4] + expected[8]).real() / 3.0;
        correct =
            correct &&
            std::abs(snapshot.epsmacs[ifreq] - epsmac) <= TOLERANCE * (1.0 + std::abs(epsmac)) &&
            std::abs(snapshot.frequencies[ifreq] - frequency) <= TOLERANCE;
    }
    require_collective(correct, "head fixture disagrees with the explicit two-level tensor");
    require_collective(snapshot.fd_enabled == (beta > 0.0) &&
                           snapshot.metallic_static == (beta > 0.0) &&
                           (beta <= 0.0 || std::abs(snapshot.kbt - 1.0 / beta) <= TOLERANCE),
                       "head fixture has inconsistent FD metadata");
}

void check_same(const HeadSnapshot &reused, const HeadSnapshot &fresh, const std::string &stage)
{
    bool same = true;
    double max_error = 0.0;
    for (int ifreq = 0; ifreq < N_FREQ; ++ifreq)
    {
        for (int entry = 0; entry < 9; ++entry)
        {
            const double error =
                std::abs(reused.tensors[ifreq][entry] - fresh.tensors[ifreq][entry]);
            max_error = std::max(max_error, error);
            same = same && error <= TOLERANCE * (1.0 + std::abs(fresh.tensors[ifreq][entry]));
        }
        same = same && std::abs(reused.epsmacs[ifreq] - fresh.epsmacs[ifreq]) <=
                           TOLERANCE * (1.0 + std::abs(fresh.epsmacs[ifreq]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // Check numerical results before metadata so the unfixed archive fails on
    // stale physics, not merely on an outdated beta or node label.
    require_collective(same, "HEADWING_CACHE_STALE: " + stage +
                                 " reused/fresh tensor or epsmac mismatch; max_tensor_error=" +
                                 std::to_string(max_error));
    require_collective(reused.frequencies == fresh.frequencies &&
                           reused.fd_enabled == fresh.fd_enabled &&
                           reused.metallic_static == fresh.metallic_static &&
                           (!fresh.fd_enabled || reused.kbt == fresh.kbt),
                       "head cache metadata differs from fresh handler: " + stage);
}

void check_dynamic_change(const HeadSnapshot &before, const HeadSnapshot &after,
                          const std::string &stage)
{
    double tensor_delta = 0.0, epsmac_delta = 0.0;
    for (int ifreq = 1; ifreq < N_FREQ; ++ifreq)
    {
        for (int entry = 0; entry < 9; ++entry)
            tensor_delta = std::max(
                tensor_delta, std::abs(before.tensors[ifreq][entry] - after.tensors[ifreq][entry]));
        epsmac_delta =
            std::max(epsmac_delta, std::abs(before.epsmacs[ifreq] - after.epsmacs[ifreq]));
    }
    require_collective(tensor_delta > 1e-5 && epsmac_delta > 1e-5,
                       "head fixture cannot expose a numerical stale cache: " + stage);
}

HeadSnapshot fresh_head(double beta, int mode_stride, const librpa::Options &options)
{
    librpa::Handler fresh(MPI_COMM_WORLD);
    initialize_fixture(fresh, beta, mode_stride);
    const auto snapshot = build_head(*api::get_dataset_instance(fresh), options);
    check_reference(snapshot, beta, mode_stride);
    return snapshot;
}

void check_cache_lifetime()
{
    librpa::Options options;
    options.replace_w_head = LIBRPA_SWITCH_ON;
    options.option_dielect_func = 3;
    options.use_2d_dielectric = LIBRPA_SWITCH_OFF;
    options.use_symmetry_exx = options.use_symmetry_rpa = options.use_symmetry_gw =
        LIBRPA_SWITCH_OFF;
    options.use_shrink_abfs = options.use_shrink_chi = LIBRPA_SWITCH_OFF;
    options.use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;

    librpa::Handler reused(MPI_COMM_WORLD);
    initialize_fixture(reused, BETA_1, 1);
    const auto ds = api::get_dataset_instance(reused);
    const auto initial = build_head(*ds, options);
    check_same(initial, fresh_head(BETA_1, 1, options), "initial beta");

    set_reference(reused, BETA_2);
    set_bosonic_nodes(*ds, BETA_2, 1);
    const auto fresh_beta = fresh_head(BETA_2, 1, options);
    check_dynamic_change(initial, fresh_beta, "beta1 -> beta2");
    const auto changed_beta = build_head(*ds, options);
    check_same(changed_beta, fresh_beta, "beta1 -> beta2");

    // No public setters between these builds: the current FD flag alone must
    // invalidate cached head data when the bosonic nodes change at fixed size.
    set_bosonic_nodes(*ds, BETA_2, 2);
    const auto fresh_nodes = fresh_head(BETA_2, 2, options);
    check_dynamic_change(changed_beta, fresh_nodes, "same-count bosonic nodes");
    const auto changed_nodes = build_head(*ds, options);
    check_same(changed_nodes, fresh_nodes, "same-count bosonic nodes");

    set_reference(reused, 0.0);
    const auto fresh_zero = fresh_head(0.0, 2, options);
    check_dynamic_change(changed_nodes, fresh_zero, "FD -> zero T at fixed frequencies");
    const auto zero = build_head(*ds, options);
    check_same(zero, fresh_zero, "FD -> zero T at fixed frequencies");

    const auto *zero_cache = ds->p_headwing.get();
    const auto zero_again = build_head(*ds, options);
    require_collective(ds->p_headwing.get() == zero_cache, "ordinary zero-T cache was not reused");
    check_same(zero_again, zero, "unchanged zero-T cache");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    const librpa_int::MpiCommHandler comm(MPI_COMM_WORLD, true);
    try
    {
        librpa::init_global(LIBRPA_SWITCH_OFF, "stdout", LIBRPA_SWITCH_OFF);
        require_collective(comm.nprocs == 1 || comm.nprocs == 4,
                           "head cache test requires MPI1 or MPI4");
        check_cache_lifetime();
        librpa::finalize_global();
        if (comm.myid == 0)
            std::cout << "THERMAL_HEADWING_CACHE_PASS ranks=" << comm.nprocs << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << "rank " << comm.myid << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
