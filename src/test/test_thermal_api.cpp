#include "librpa.hpp"

#include "../api/instance_manager.h"
#include "../api/dataset_helper.h"
#include "librpa_options.h"

#include <cassert>
#include <cmath>
#include <mpi.h>
#include <stdexcept>

namespace
{
bool near(const double lhs, const double rhs)
{
    return std::abs(lhs - rhs) < 1.0e-15;
}
}

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

    MPI_Finalize();
    return 0;
}
