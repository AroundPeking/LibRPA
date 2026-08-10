#include "../mpi/global_mpi.h"
#include "../mpi/shrink_scalapack_layout.h"

#include <cassert>
#include <mpi.h>
#include <stdexcept>

int main(int argc, char *argv[])
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    init_global_mpi(MPI_COMM_WORLD);

    if (size_global != 4)
        throw std::runtime_error("test imposes 4 MPI processes");

    BlacsCtxtHandler blacs_h(mpi_comm_global);
    blacs_h.init();
    blacs_h.set_square_grid();

    const auto layout = make_shrink_scalapack_layout(blacs_h, 1884, 1078, 128);
    assert(layout.large_large.mb() == 128);
    assert(layout.large_large.nb() == 128);
    assert(layout.small_large.mb() == 128);
    assert(layout.small_large.nb() == 128);
    assert(layout.small_small.mb() == 128);
    assert(layout.small_small.nb() == 128);

    blacs_h.exit();
    finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
