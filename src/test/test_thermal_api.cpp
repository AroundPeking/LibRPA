#include "librpa.hpp"

#include "../api/instance_manager.h"

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
        assert(!dataset->fermi_dirac_reference.enabled);

        handler.set_fermi_dirac_reference(0.0025, 0.125, 2.0, 1.0e-12);
        assert(dataset->fermi_dirac_reference.enabled);
        assert(near(dataset->fermi_dirac_reference.kbt_ha, 0.0025));
        assert(near(dataset->fermi_dirac_reference.chemical_potential_ha, 0.125));
        assert(near(dataset->fermi_dirac_reference.max_occupation_per_band, 2.0));
        assert(near(dataset->fermi_dirac_reference.occupation_tolerance, 1.0e-12));

        handler.clear_fermi_dirac_reference();
        assert(!dataset->fermi_dirac_reference.enabled);

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
