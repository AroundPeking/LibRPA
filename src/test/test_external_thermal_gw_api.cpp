#include <mpi.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> &operation, const std::string &message = "")
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
    {
        require(std::string(error.what()).find(message) != std::string::npos,
                "rejection did not identify the expected invalid input");
        return;
    }
    throw std::runtime_error("invalid external GW input was accepted");
}

struct GridInput
{
    double beta = 8, gmax = 1, wmax = 2, smax = 3, tolerance = 1e-10;
    int ntau = 2, nb = 3, nf = 4;
    std::vector<double> times{0.5, 7};
    std::vector<int> bosons{2, 0, -2}, fermions{3, -1, 0, -4};
    // Deliberately non-even complex matrices: only static normalization is exact.
    std::vector<double> br{1, 0.125, 2, 3, 0.125, 4}, bi{2, 0, 3, 4, 0, 5};
    std::vector<double> fr{1, 2, 3, 4, 5, 6, 7, 8}, fi{8, 7, 6, 5, 4, 3, 2, 1};

    void set(librpa::Handler &h, int missing = -1) const
    {
        h.set_external_thermal_gw_grid(
            beta, gmax, wmax, smax, tolerance, ntau, nb, nf, missing == 0 ? nullptr : times.data(),
            missing == 1 ? nullptr : bosons.data(), missing == 2 ? nullptr : fermions.data(),
            missing == 3 ? nullptr : br.data(), missing == 4 ? nullptr : bi.data(),
            missing == 5 ? nullptr : fr.data(), missing == 6 ? nullptr : fi.data());
    }
};

void check_api()
{
    librpa::Handler h(MPI_COMM_WORLD);
    const auto ds = librpa_int::api::get_dataset_instance(h);
    GridInput input;
    rejects([&]() { input.set(h); }, "SCF");
    h.set_scf_dimension(1, 1, 2, 2);
    const double energies[]{-0.25, 0.75}, occupations[]{1, 1};
    h.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.5);
    rejects([&]() { input.set(h); }, "FD");
    h.set_fermi_dirac_reference(0.125, 0.5, 2, 1e-10);
    ds->is_band_calc_done = true;
    input.set(h);
    require(!ds->is_band_calc_done, "setting GW input did not invalidate computed data");
    const auto saved = input;
    input.times[0] = 1;
    input.bosons[0] = 100;
    input.fermions[0] = 100;
    input.br[0] = input.bi[0] = input.fr[0] = input.fi[0] = 99;
    auto check_saved = [&]()
    {
        const auto &g = *ds->external_thermal_gw_grid;
        require(g.g_wmax_ha == saved.gmax && g.w_wmax_ha == saved.wmax &&
                    g.sigma_wmax_ha == saved.smax && g.tolerance == saved.tolerance,
                "lost independent spectral bounds");
        const auto &t = g.transform;
        require(t.get_beta_ha_inv() == saved.beta && t.get_times() == saved.times &&
                    t.get_bosonic_indices() == saved.bosons &&
                    t.get_fermionic_indices() == saved.fermions,
                "lost owned nodes or reordered signed indices");
        auto b = t.copy_bosonic_frequency_to_time();
        auto f = t.copy_fermionic_time_to_frequency();
        for (int i = 0; i < b.size; ++i)
            require(b.c[i] == std::complex<double>(saved.br[i], saved.bi[i]), "changed B layout");
        for (int i = 0; i < f.size; ++i)
            require(f.c[i] == std::complex<double>(saved.fr[i], saved.fi[i]), "changed F layout");
        b.c[0] = f.c[0] = 0;
        require(t.copy_bosonic_frequency_to_time().c[0] == std::complex<double>(1, 2) &&
                    t.copy_fermionic_time_to_frequency().c[0] == std::complex<double>(1, 8),
                "matrix getters exposed mutable operator storage");
    };
    check_saved();
    librpa::Options opts;
    opts.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    opts.ntau = 8;
    opts.nfreq = 5;
    auto initialize = [&]() { librpa_int::initialize_ds_tfgrids(*ds, opts); };
    initialize();
    require(ds->tfg.get_n_grids() == 5 && ds->tfg.get_n_time_grids() == 8,
            "GW input replaced the response grid");
    auto bad_edit = [&](const std::function<void(GridInput &)> &edit)
    {
        auto bad = saved;
        edit(bad);
        ds->is_band_calc_done = true;
        rejects([&]() { bad.set(h); });
        require(ds->is_band_calc_done, "failed input invalidated existing computation");
        check_saved();
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    bad_edit([](GridInput &g) { g.beta = 16; });
    bad_edit([&](GridInput &g) { g.beta = nan; });
    bad_edit([](GridInput &g) { g.gmax = 0.7; });
    bad_edit([](GridInput &g) { g.wmax = 0; });
    bad_edit([](GridInput &g) { g.smax = 2.9; });
    bad_edit([&](GridInput &g) { g.smax = inf; });
    bad_edit([](GridInput &g) { g.smax = 1e308; });
    bad_edit([&](GridInput &g) { g.gmax = inf; });
    bad_edit([&](GridInput &g) { g.wmax = nan; });
    bad_edit([](GridInput &g) { g.tolerance = 0; });
    bad_edit([](GridInput &g) { g.tolerance = 1; });
    bad_edit([&](GridInput &g) { g.tolerance = nan; });
    bad_edit([](GridInput &g) { g.ntau = 0; });
    bad_edit([](GridInput &g) { g.nb = -1; });
    bad_edit([](GridInput &g) { g.nf = std::numeric_limits<int>::max(); });
    bad_edit([](GridInput &g) { g.nb = std::numeric_limits<int>::max(); });
    bad_edit([](GridInput &g) { g.times[0] = 0; });
    bad_edit([](GridInput &g) { g.times[1] = g.beta; });
    bad_edit([](GridInput &g) { g.times[1] = g.times[0]; });
    bad_edit([](GridInput &g) { g.bosons[1] = 1; });
    bad_edit([](GridInput &g) { g.bosons[0] = -2; });
    bad_edit([](GridInput &g) { g.bosons[0] = 3; });
    bad_edit([](GridInput &g) { g.bosons[0] = std::numeric_limits<int>::min(); });
    bad_edit([](GridInput &g) { g.fermions[0] = 2; });
    bad_edit([](GridInput &g) { g.fermions[0] = -1; });
    bad_edit([&](GridInput &g) { g.br[0] = nan; });
    bad_edit([&](GridInput &g) { g.bi[0] = inf; });
    bad_edit([&](GridInput &g) { g.fr[0] = nan; });
    bad_edit([&](GridInput &g) { g.fi[0] = inf; });
    bad_edit(
        [](GridInput &g)
        {
            g.br[1] *= 0.5;
            g.tolerance = 0.1;
        });
    bad_edit([](GridInput &g) { g.bi[4] = 1e-4; });
    for (int i = 0; i < 7; ++i) rejects([&]() { saved.set(h, i); });
    check_saved();

    // G's support is abs(epsilon-mu), not the full electron-hole range (1 Ha).
    auto narrower = saved;
    narrower.gmax = 0.8;
    narrower.set(h);
    initialize();
    saved.set(h);
    h.set_fermi_dirac_reference(0.25, 0.5, 2, 1e-10);
    rejects(initialize, "beta");
    h.clear_fermi_dirac_reference();
    rejects(initialize, "FD");
    auto legacy_options = opts;
    legacy_options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    legacy_options.nfreq = 0;
    ds->is_band_calc_done = true;
    rejects([&]() { h.build_g0w0_sigma(legacy_options); },
            "Finite-temperature GW is not yet supported");
    require(ds->is_band_calc_done, "stored GW metadata guard no longer fails before mutation");
    h.set_fermi_dirac_reference(0.125, 0.5, 2, 1e-10);
    const double wider[]{-0.25, 2};
    ds->is_band_calc_done = true;
    h.set_wg_ekb_efermi(1, 1, 2, occupations, wider, 0.5);
    require(!ds->is_band_calc_done, "changed SCF energies retained computed GW data");
    rejects(initialize, "g_wmax");
    h.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.5);
    initialize();
    check_saved();
    rejects([&]() { h.build_g0w0_sigma(opts); }, "Finite-temperature GW is not yet supported");

    const double rt[]{1, 3, 7}, rr[]{1, 2, 5, 1, -3, 2}, ri[]{0, 0, 0, 2, -1, -1};
    h.set_external_thermal_time_grid(8, 2, 1e-10, 2, 3, rt, rr, ri);
    opts.nfreq = 2;
    opts.ntau = 3;
    initialize();
    check_saved();
    h.clear_external_thermal_gw_grid();
    require(!ds->external_thermal_gw_grid && ds->external_thermal_time_grid,
            "clearing GW input cleared the response grid");
    initialize();
    saved.set(h);
    h.clear_external_thermal_time_grid();
    initialize();
    check_saved();
    // Dimensions cannot be reset on an initialized MeanField. Inject stale
    // readiness internally to exercise reuse validation without changing that API.
    ds->is_scf_eigocc_set = false;
    rejects(initialize, "SCF");
    h.clear_external_thermal_gw_grid();
    h.clear_external_thermal_gw_grid();
    require(ds->mf.get_fermi_dirac_reference().enabled, "clear removed FD reference");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check_api();
    MPI_Finalize();
}
