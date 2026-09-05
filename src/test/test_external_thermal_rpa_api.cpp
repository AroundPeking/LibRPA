#include <mpi.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
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

void rejects(const std::function<void()> &operation)
{
    try
    {
        operation();
    }
    catch (const std::runtime_error &)
    {
        return;
    }
    throw std::runtime_error("invalid external thermal RPA grid was accepted");
}

struct GridInput
{
    double beta = 8.0, wmax = 2.0, rpa_wmax = 4.0, tolerance = 1e-10;
    int nfreq = 4, ntau = 3;
    std::vector<double> times{0.5, 3.0, 7.0};
    std::vector<int> indices{0, 2, 9, 100};
    std::vector<double> weights{0.0625, 0.25, -0.125, 0.0};
    // DC-exact synthetic data test transfer and validation, not RPA accuracy.
    std::vector<double> real{1, 2, 5, 1, -3, 2, 0, 0, 0, 0, 0, 0};
    std::vector<double> imag{0, 0, 0, 2, -1, -1, 0, 0, 0, 0, 0, 0};

    void set(librpa::Handler &handler) const
    {
        handler.set_external_thermal_rpa_grid(beta, wmax, rpa_wmax, tolerance, nfreq, ntau,
                                              times.data(), indices.data(), weights.data(),
                                              real.data(), imag.data());
    }
};

void check_api()
{
    librpa::Handler handler(MPI_COMM_WORLD);
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    GridInput input;
    rejects([&]() { input.set(handler); });
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25, 0.75}, occupations[] = {1.0, 1.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    rejects([&]() { input.set(handler); });
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    input.set(handler);
    const auto saved = input;
    input.times[0] = 0.75;
    input.indices[1] = 3;
    input.weights[1] = 123;
    input.real[0] = 123;
    input.imag[3] = 456;

    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = saved.nfreq;
    options.ntau = saved.ntau;
    auto initialize = [&]() { librpa_int::initialize_ds_tfgrids(*ds, options); };
    auto check_saved = [&]()
    {
        require(ds->external_thermal_time_grid->rpa_wmax_ha == saved.rpa_wmax,
                "lost RPA spectral bound");
        require(ds->external_thermal_time_grid->frequency_indices == saved.indices &&
                    ds->external_thermal_time_grid->correlation_weights == saved.weights,
                "API did not copy sparse indices and weights");
        require(ds->tfg.get_time_nodes() == saved.times, "API did not copy sparse times");
        require(ds->tfg.get_time_to_frequency_factor(0, 0).real() == 1.0 &&
                    ds->tfg.get_time_to_frequency_factor(1, 0) == std::complex<double>(1, 2),
                "API did not copy row-major complex transform");
        const auto &frequencies = ds->tfg.get_freq_nodes();
        for (int row = 0; row < saved.nfreq; ++row)
        {
            const double expected = 2.0 * std::acos(-1.0) * saved.indices[row] / saved.beta;
            require(std::abs(frequencies[row] - expected) < 1e-12,
                    "sparse frequency gaps were replaced by consecutive indices");
            require(std::abs(ds->tfg.find_correlation_frequency_weight(frequencies[row]) -
                             saved.weights[row]) < 1e-14,
                    "signed RPA correlation weights changed");
        }
    };
    initialize();
    check_saved();

    auto reject_edit = [&](const std::function<void(GridInput &)> &edit)
    {
        auto bad = saved;
        edit(bad);
        rejects([&]() { bad.set(handler); });
        initialize();
        check_saved();
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    reject_edit([](GridInput &g) { g.beta *= 2; });
    reject_edit([&](GridInput &g) { g.beta = nan; });
    reject_edit([](GridInput &g) { g.wmax = 0.9; });
    reject_edit([](GridInput &g) { g.rpa_wmax = 1.9; });
    reject_edit([&](GridInput &g) { g.rpa_wmax = inf; });
    reject_edit([](GridInput &g) { g.tolerance = 0; });
    reject_edit([](GridInput &g) { g.tolerance = 1; });
    reject_edit([](GridInput &g) { g.nfreq = 0; });
    reject_edit([](GridInput &g) { g.ntau = -1; });
    reject_edit([](GridInput &g) { g.nfreq = std::numeric_limits<int>::max(); });
    reject_edit([](GridInput &g) { g.indices[0] = 1; });
    reject_edit([](GridInput &g) { g.indices[1] = -1; });
    reject_edit([](GridInput &g) { g.indices[2] = 2; });
    reject_edit([](GridInput &g) { g.indices[2] = 1; });
    reject_edit([&](GridInput &g) { g.weights[1] = nan; });
    reject_edit([&](GridInput &g) { g.weights[1] = inf; });
    reject_edit([](GridInput &g) { g.weights[0] *= 1.0 + 2e-10; });
    reject_edit([](GridInput &g) { g.weights[0] = -0.0625; });
    reject_edit([](GridInput &g) { g.times[0] = 0; });
    reject_edit([](GridInput &g) { g.times[1] = g.times[0]; });
    reject_edit([](GridInput &g) { g.times[2] = g.beta; });
    reject_edit([&](GridInput &g) { g.real[3] = nan; });
    reject_edit([&](GridInput &g) { g.imag[3] = inf; });
    reject_edit([](GridInput &g) { g.real[0] += 1; });
    reject_edit([](GridInput &g) { g.real[3] += 1; });
    reject_edit([](GridInput &g) { g.imag[3] += 1; });
    reject_edit(
        [](GridInput &g)
        {
            g.imag[0] = 1;
            g.imag[1] = -1;
        });
    for (int missing = 0; missing < 5; ++missing)
        rejects(
            [&]()
            {
                handler.set_external_thermal_rpa_grid(saved.beta, saved.wmax, saved.rpa_wmax,
                                                      saved.tolerance, saved.nfreq, saved.ntau,
                                                      missing == 0 ? nullptr : saved.times.data(),
                                                      missing == 1 ? nullptr : saved.indices.data(),
                                                      missing == 2 ? nullptr : saved.weights.data(),
                                                      missing == 3 ? nullptr : saved.real.data(),
                                                      missing == 4 ? nullptr : saved.imag.data());
            });
    initialize();
    check_saved();

    auto boundary = saved;
    boundary.rpa_wmax = boundary.wmax;
    boundary.weights[0] *= 1.0 + 5e-11;
    boundary.set(handler);
    initialize();
    saved.set(handler);
    initialize();
    for (int mismatch = 0; mismatch < 3; ++mismatch)
    {
        auto bad_options = options;
        if (mismatch == 0) ++bad_options.nfreq;
        if (mismatch == 1) ++bad_options.ntau;
        if (mismatch == 2) bad_options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
        rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, bad_options); });
        check_saved();
    }
    handler.set_fermi_dirac_reference(0.25, 0.0, 2.0, 1e-10);
    rejects(initialize);
    handler.clear_fermi_dirac_reference();
    rejects(initialize);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    const double wider_energies[] = {-0.25, 3.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, wider_energies, 0.0);
    rejects(initialize);
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    initialize();
    check_saved();

    handler.set_external_thermal_time_grid(saved.beta, saved.wmax, saved.tolerance, saved.nfreq,
                                           saved.ntau, saved.times.data(), saved.real.data(),
                                           saved.imag.data());
    initialize();
    require(ds->external_thermal_time_grid->frequency_indices.empty() &&
                ds->external_thermal_time_grid->correlation_weights.empty(),
            "legacy setter retained sparse RPA input");
    require(std::abs(ds->tfg.get_freq_nodes()[1] - 2.0 * std::acos(-1.0) / saved.beta) < 1e-14 &&
                std::abs(ds->tfg.find_correlation_frequency_weight(ds->tfg.get_freq_nodes()[1]) -
                         1.0 / saved.beta) < 1e-14,
            "legacy consecutive Matsubara path changed");
    saved.set(handler);
    handler.clear_external_thermal_time_grid();
    options.ntau = 32;
    initialize();
    require(!ds->external_thermal_time_grid &&
                std::abs(ds->tfg.get_time_nodes()[0] - saved.beta / 64.0) < 1e-14,
            "clearing sparse input did not restore uniform fallback");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check_api();
    MPI_Finalize();
}
