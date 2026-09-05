#include <mpi.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"
#include "librpa_input.h"

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> &operation)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    require(rejected, "invalid external thermal grid input was accepted");
}

void check_api()
{
    librpa::Handler handler(MPI_COMM_WORLD);
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25, 0.75};
    const double occupations[] = {1.0, 1.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    constexpr double beta = 8.0;
    const std::vector<double> times{0.2, 1.0, 3.0, 5.0, 7.8};
    // A DC-exact synthetic operator exercises data transfer, not physical accuracy.
    std::vector<double> real(8 * times.size(), 0.0), imag(real.size(), 0.0);
    for (int j = 0; j < 5; ++j) real[j] = beta / 5.0;
    real[5] = 0.25;
    real[6] = -0.25;
    imag[5] = 0.5;
    imag[6] = -0.5;
    auto set = [&](double inverse_temperature, double wmax, double tolerance, int nfreq, int ntau,
                   const double *tau, const double *re, const double *im)
    {
        handler.set_external_thermal_time_grid(inverse_temperature, wmax, tolerance, nfreq, ntau,
                                               tau, re, im);
    };
    auto set_valid = [&]() { set(beta, 2.0, 1e-10, 8, 5, times.data(), real.data(), imag.data()); };
    rejects(set_valid);
    handler.set_fermi_dirac_reference(1.0 / beta, 0.0, 2.0, 1e-10);
    set_valid();
    const auto real_saved = real;
    const auto imag_saved = imag;
    real[0] = 123.0;
    imag[5] = 456.0;

    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = 8;
    options.ntau = 5;
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->tfg.get_time_nodes() == times, "API fell back to uniform thermal times");
    require(ds->tfg.get_time_to_frequency_factor(1, 0) == std::complex<double>(0.25, 0.5),
            "API did not copy row-major complex operator");
    require(std::abs(ds->tfg.find_correlation_frequency_weight(0.0) - 1.0 / (2.0 * beta)) < 1e-14,
            "external grid changed RPA Matsubara weights");
    real = real_saved;
    imag = imag_saved;

    rejects([&]() { set(2 * beta, 2.0, 1e-10, 8, 5, times.data(), real.data(), imag.data()); });
    rejects([&]() { set(beta, 0.9, 1e-10, 8, 5, times.data(), real.data(), imag.data()); });
    rejects([&]() { set(beta, 2.0, 0.0, 8, 5, times.data(), real.data(), imag.data()); });
    rejects([&]() { set(beta, 2.0, 1e-10, 8, 5, nullptr, real.data(), imag.data()); });
    rejects([&]() { set(beta, 2.0, 1e-10, 8, 5, times.data(), nullptr, imag.data()); });
    rejects([&]() { set(beta, 2.0, 1e-10, 0, 5, times.data(), real.data(), imag.data()); });
    rejects([&]() { set(beta, 2.0, 1e-10, 8, 5, times.data(), real.data(), nullptr); });
    rejects(
        [&]()
        {
            set(beta, 2.0, 1e-10, std::numeric_limits<int>::max(), 5, times.data(), real.data(),
                imag.data());
        });
    auto bad_real = real;
    bad_real[0] *= 2.0;
    rejects([&]() { set(beta, 2.0, 1e-10, 8, 5, times.data(), bad_real.data(), imag.data()); });
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->tfg.get_time_to_frequency_factor(0, 0).real() == beta / 5.0,
            "rejected setter changed stored external input");

    auto bad_options = options;
    bad_options.ntau = 6;
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, bad_options); });
    require(ds->tfg.get_time_nodes() == times, "invalid options changed active grid");
    bad_options = options;
    bad_options.nfreq = 7;
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, bad_options); });
    bad_options = options;
    bad_options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, bad_options); });
    handler.set_fermi_dirac_reference(2.0 / beta, 0.0, 2.0, 1e-10);
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, options); });
    handler.set_fermi_dirac_reference(1.0 / beta, 0.0, 2.0, 1e-10);
    const double wider_energies[] = {-0.25, 3.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, wider_energies, 0.0);
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, options); });
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);

    handler.clear_external_thermal_time_grid();
    options.ntau = 32;
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(std::abs(ds->tfg.get_time_nodes()[0] - beta / 64.0) < 1e-14,
            "clearing external grid did not restore uniform fallback");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check_api();
    MPI_Finalize();
}
