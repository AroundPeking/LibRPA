#include "../mpi/global_mpi.h"
#include "../mpi/shrink_scalapack_layout.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>

namespace {

using Complex = std::complex<double>;

struct ShrinkScalapackBenchmarkResult
{
    double gemm1_seconds = 0.0;
    double gemm2_seconds = 0.0;
    double relative_error = 0.0;
    double hermiticity_residual = 0.0;
    int large_large_block = 0;
    int small_large_block = 0;
    int small_small_block = 0;
};

int env_positive_int(const char *name, const int fallback, const bool allow_zero = false)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallback;

    const int parsed = std::stoi(value);
    if (parsed < 0 || (!allow_zero && parsed == 0))
        throw std::invalid_argument(std::string(name) + " must be positive");
    return parsed;
}

double row_scale(const int i)
{
    return 1.0 + 1.0e-3 * static_cast<double>(i % 17);
}

Complex hermitian_element(const int i, const int j, const int n)
{
    if (i == j)
        return Complex(2.0 + static_cast<double>(i + 1) / static_cast<double>(n), 0.0);

    const double real_part = 1.0e-2 / (1.0 + std::abs(i - j));
    const double imag_part = 1.0e-3 * static_cast<double>(i - j) / static_cast<double>(n);
    return Complex(real_part, imag_part);
}

librpa_int::ShrinkScalapackLayout make_benchmark_layout(
    const librpa_int::BlacsCtxtHandler &blacs_h, const int n_large,
    const int n_small, const int block_size)
{
    if (block_size > 0)
        return librpa_int::make_shrink_scalapack_layout(
            blacs_h, n_large, n_small, block_size);

    librpa_int::ShrinkScalapackLayout layout(blacs_h);
    layout.large_large.init_square_blk_capped(n_large, n_large, 2048, 0, 0);
    layout.small_large.init_square_blk_capped(n_small, n_large, 2048, 0, 0);
    layout.small_small.init_square_blk_capped(n_small, n_small, 2048, 0, 0);
    return layout;
}

ShrinkScalapackBenchmarkResult run_shrink_scalapack_benchmark(
    const librpa_int::BlacsCtxtHandler &blacs_h, const int n_large,
    const int n_small, const int block_size, const int repeats)
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    if (n_large <= 0 || n_small <= 0 || n_small > n_large || repeats <= 0)
        throw std::invalid_argument("invalid shrink ScaLAPACK benchmark dimensions");

    auto layout = make_benchmark_layout(blacs_h, n_large, n_small, block_size);
    ArrayDesc chi0_source_desc(blacs_h);
    ArrayDesc u_source_desc(blacs_h);
    ArrayDesc result_source_desc(blacs_h);
    chi0_source_desc.init(n_large, n_large, n_large, n_large, 0, 0);
    u_source_desc.init(n_small, n_large, n_small, n_large, 0, 0);
    result_source_desc.init(n_small, n_small, n_small, n_small, 0, 0);

    matrix_m<Complex> chi0_source(1, 1, MAJOR::COL);
    matrix_m<Complex> u_source(1, 1, MAJOR::COL);
    matrix_m<Complex> result_source(1, 1, MAJOR::COL);
    if (myid_global == 0)
    {
        chi0_source.resize(n_large, n_large);
        u_source.resize(n_small, n_large);
        result_source.resize(n_small, n_small);
        for (int j = 0; j < n_large; ++j)
            for (int i = 0; i < n_large; ++i)
                chi0_source(i, j) = hermitian_element(i, j, n_large);
        u_source.zero_out();
        for (int i = 0; i < n_small; ++i)
            u_source(i, i) = row_scale(i);
    }

    auto chi0_local = init_local_mat<Complex>(layout.large_large, MAJOR::COL);
    auto u_local = init_local_mat<Complex>(layout.small_large, MAJOR::COL);
    auto u_chi0_local = init_local_mat<Complex>(layout.small_large, MAJOR::COL);
    auto result_local = init_local_mat<Complex>(layout.small_small, MAJOR::COL);

    ScalapackConnector::pgemr2d_f(
        n_large, n_large, chi0_source.ptr(), 1, 1, chi0_source_desc.desc,
        chi0_local.ptr(), 1, 1, layout.large_large.desc, blacs_h.ictxt);
    ScalapackConnector::pgemr2d_f(
        n_small, n_large, u_source.ptr(), 1, 1, u_source_desc.desc,
        u_local.ptr(), 1, 1, layout.small_large.desc, blacs_h.ictxt);

    double gemm1_seconds = 0.0;
    double gemm2_seconds = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        u_chi0_local.zero_out();
        result_local.zero_out();
        blacs_h.barrier();
        const double gemm1_start = MPI_Wtime();
        ScalapackConnector::pgemm_f(
            'N', 'N', n_small, n_large, n_large, Complex(1.0, 0.0),
            u_local.ptr(), 1, 1, layout.small_large.desc,
            chi0_local.ptr(), 1, 1, layout.large_large.desc,
            Complex(0.0, 0.0), u_chi0_local.ptr(), 1, 1,
            layout.small_large.desc);
        const double gemm1_local = MPI_Wtime() - gemm1_start;
        double gemm1_max = 0.0;
        MPI_Allreduce(&gemm1_local, &gemm1_max, 1, MPI_DOUBLE, MPI_MAX,
                      blacs_h.comm());
        gemm1_seconds += gemm1_max;

        blacs_h.barrier();
        const double gemm2_start = MPI_Wtime();
        ScalapackConnector::pgemm_f(
            'N', 'C', n_small, n_small, n_large, Complex(1.0, 0.0),
            u_chi0_local.ptr(), 1, 1, layout.small_large.desc,
            u_local.ptr(), 1, 1, layout.small_large.desc,
            Complex(0.0, 0.0), result_local.ptr(), 1, 1,
            layout.small_small.desc);
        const double gemm2_local = MPI_Wtime() - gemm2_start;
        double gemm2_max = 0.0;
        MPI_Allreduce(&gemm2_local, &gemm2_max, 1, MPI_DOUBLE, MPI_MAX,
                      blacs_h.comm());
        gemm2_seconds += gemm2_max;
    }

    ScalapackConnector::pgemr2d_f(
        n_small, n_small, result_local.ptr(), 1, 1, layout.small_small.desc,
        result_source.ptr(), 1, 1, result_source_desc.desc, blacs_h.ictxt);

    ShrinkScalapackBenchmarkResult benchmark;
    benchmark.gemm1_seconds = gemm1_seconds / repeats;
    benchmark.gemm2_seconds = gemm2_seconds / repeats;
    benchmark.large_large_block = layout.large_large.mb();
    benchmark.small_large_block = layout.small_large.mb();
    benchmark.small_small_block = layout.small_small.mb();
    if (myid_global == 0)
    {
        double reference_norm2 = 0.0;
        double error_norm2 = 0.0;
        double hermitian_error_norm2 = 0.0;
        for (int j = 0; j < n_small; ++j)
        {
            for (int i = 0; i < n_small; ++i)
            {
                const Complex reference =
                    row_scale(i) * hermitian_element(i, j, n_large) * row_scale(j);
                reference_norm2 += std::norm(reference);
                error_norm2 += std::norm(result_source(i, j) - reference);
                hermitian_error_norm2 += std::norm(
                    result_source(i, j) - std::conj(result_source(j, i)));
            }
        }
        benchmark.relative_error = std::sqrt(error_norm2 / reference_norm2);
        benchmark.hermiticity_residual =
            std::sqrt(hermitian_error_norm2 / reference_norm2);
    }
    MPI_Bcast(&benchmark.relative_error, 2, MPI_DOUBLE, 0, blacs_h.comm());
    return benchmark;
}

} // namespace

int main(int argc, char *argv[])
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    init_global_mpi(MPI_COMM_WORLD);

    const int grid_size = static_cast<int>(std::sqrt(size_global));
    if (grid_size * grid_size != size_global)
        throw std::runtime_error("test requires a square number of MPI processes");

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

    bool invalid_layout_rejected = false;
    try
    {
        make_shrink_scalapack_layout(blacs_h, 1884, 1078, 0);
    }
    catch (const std::invalid_argument &)
    {
        invalid_layout_rejected = true;
    }
    assert(invalid_layout_rejected);

    const int n_large = env_positive_int("LIBRPA_TEST_SHRINK_LARGE", 32);
    const int n_small = env_positive_int("LIBRPA_TEST_SHRINK_SMALL", 20);
    const int block_size = env_positive_int("LIBRPA_TEST_SHRINK_BLOCK", 8, true);
    const int repeats = env_positive_int("LIBRPA_TEST_SHRINK_REPEATS", 1);
    const auto benchmark = run_shrink_scalapack_benchmark(
        blacs_h, n_large, n_small, block_size, repeats);
    assert(benchmark.relative_error <= 1.0e-10);
    assert(benchmark.hermiticity_residual <= 1.0e-10);
    if (myid_global == 0)
    {
        std::cout << std::setprecision(12)
                  << "SHRINK_SCALAPACK_BENCH"
                  << " large=" << n_large
                  << " small=" << n_small
                  << " block=" << block_size
                  << " ranks=" << size_global
                  << " grid=" << blacs_h.nprows << "x" << blacs_h.npcols
                  << " ll_mb=" << benchmark.large_large_block
                  << " sl_mb=" << benchmark.small_large_block
                  << " ss_mb=" << benchmark.small_small_block
                  << " gemm1_s=" << benchmark.gemm1_seconds
                  << " gemm2_s=" << benchmark.gemm2_seconds
                  << " relerr=" << benchmark.relative_error
                  << " herm=" << benchmark.hermiticity_residual
                  << " status=PASS" << std::endl;
    }

    blacs_h.exit();
    finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
