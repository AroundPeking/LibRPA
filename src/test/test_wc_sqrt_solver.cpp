#include "../gpu/la_connector.h"
#include "../mpi/global_mpi.h"
#include "../mpi/utils_blacs.h"

#include <cassert>
#include <cmath>
#include <mpi.h>

int main(int argc, char *argv[])
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    init_global_mpi(MPI_COMM_WORLD);

    const int grid_size = static_cast<int>(std::sqrt(size_global));
    assert(grid_size * grid_size == size_global);

    BlacsCtxtHandler blacs_h(mpi_comm_global);
    blacs_h.init();
    blacs_h.set_square_grid();

    const auto result = run_wc_sqrt_benchmark(
        blacs_h, env_positive_int("LIBRPA_TEST_WC_DIM", 32),
        env_positive_int("LIBRPA_TEST_WC_BLOCK", 8),
        env_flag("LIBRPA_TEST_WC_USE_ELPA", false));
    assert(result.finite);
    assert(result.filtered == 0);
    assert(result.relative_residual <= 1.0e-9);
    assert(result.hermiticity_residual <= 1.0e-12);

    blacs_h.exit();
    finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
