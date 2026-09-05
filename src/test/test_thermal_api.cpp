#include <mpi.h>

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"
#include "librpa_options.h"

namespace
{
bool near(const double lhs, const double rhs) { return std::abs(lhs - rhs) < 1.0e-15; }

void require_thermal_gw_rejection(const bool thermal_reference, const bool thermal_grid)
{
    librpa::Handler handler(MPI_COMM_WORLD);
    if (thermal_reference) handler.set_fermi_dirac_reference(0.0025, 0.125, 2.0, 1.0e-12);
    const auto dataset = librpa_int::api::get_dataset_instance(handler);
    dataset->is_band_calc_done = true;
    librpa::Options options;
    options.tfgrids_type = thermal_grid ? LIBRPA_TFGRID_FD_MATSUBARA : LIBRPA_TFGRID_MINIMAX;
    // The unsupported task must fail before grid validation or dataset mutation.
    options.nfreq = 0;
    bool rejected = false;
    try
    {
        handler.build_g0w0_sigma(options);
    }
    catch (const std::runtime_error &error)
    {
        const std::string message = error.what();
        rejected = message.find("Finite-temperature GW is not yet supported") != std::string::npos;
    }
    if (!rejected || !dataset->is_band_calc_done)
        throw std::runtime_error("missing fail-fast thermal GW protection");
}

void require_legacy_gw_grid_validation()
{
    librpa::Handler handler(MPI_COMM_WORLD);
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    options.nfreq = 0;
    bool reached_grid_validation = false;
    try
    {
        handler.build_g0w0_sigma(options);
    }
    catch (const std::runtime_error &error)
    {
        const std::string message = error.what();
        reached_grid_validation =
            message.find("number of frequency points must be positive") != std::string::npos;
    }
    if (!reached_grid_validation)
        throw std::runtime_error("legacy GW no longer reaches normal grid validation");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    {
        librpa::Handler handler(MPI_COMM_WORLD);
        const auto dataset = librpa_int::api::get_dataset_instance(handler);
        assert(!dataset->mf.get_fermi_dirac_reference().enabled);

        handler.set_fermi_dirac_reference(0.0025, 0.125, 2.0, 1.0e-12);
        const auto &reference = dataset->mf.get_fermi_dirac_reference();
        assert(reference.enabled);
        assert(near(reference.kbt_ha, 0.0025));
        assert(near(reference.chemical_potential_ha, 0.125));
        assert(near(reference.max_occupation_per_band, 2.0));
        assert(near(reference.occupation_tolerance, 1.0e-12));

        handler.clear_fermi_dirac_reference();
        assert(!dataset->mf.get_fermi_dirac_reference().enabled);

        bool rejected = false;
        try
        {
            handler.set_fermi_dirac_reference(0.0, 0.125, 2.0, 1.0e-12);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);

        LibrpaOptions options;
        librpa_init_options(&options);
        options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
        options.nfreq = 3;
        options.ntau = 4;
        rejected = false;
        try
        {
            librpa_int::initialize_ds_tfgrids(*dataset, options);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        assert(rejected);

        handler.set_fermi_dirac_reference(0.0025, 0.125, 2.0, 1.0e-12);
        librpa_int::initialize_ds_tfgrids(*dataset, options);
        assert(dataset->tfg.get_grid_type() == LIBRPA_TFGRID_FD_MATSUBARA);
        assert(dataset->tfg.get_n_grids() == 3);
        assert(dataset->tfg.get_n_time_grids() == 4);

        options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
        rejected = false;
        try
        {
            librpa_int::initialize_ds_tfgrids(*dataset, options);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        assert(rejected);

        handler.set_fermi_dirac_reference(0.0025, 0.125, 2.0, 1.0e-12);
        handler.set_scf_dimension(1, 1, 1, 1);
        const double weight[] = {2.0};
        const double energy[] = {0.125};
        rejected = false;
        try
        {
            handler.set_wg_ekb_efermi(1, 1, 1, weight, energy, 0.25);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);

        handler.clear_fermi_dirac_reference();
        handler.set_wg_ekb_efermi(1, 1, 1, weight, energy, 0.125);
        rejected = false;
        try
        {
            handler.set_fermi_dirac_reference(0.0025, 0.25, 2.0, 1.0e-12);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        assert(rejected);
    }

    require_thermal_gw_rejection(true, true);
    require_thermal_gw_rejection(true, false);
    require_thermal_gw_rejection(false, true);
    require_legacy_gw_grid_validation();

    MPI_Finalize();
    return 0;
}
