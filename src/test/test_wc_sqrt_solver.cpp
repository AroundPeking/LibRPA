#include "../math/utils_matrix_m_mpi.h"
#include "../gpu/la_connector.h"
#include "../mpi/global_mpi.h"
#include "../mpi/utils_blacs.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;

struct WcSqrtBenchmarkResult
{
    double sqrt_seconds = 0.0;
    double residual_gemm_seconds = 0.0;
    double relative_residual = 0.0;
    double hermiticity_residual = 0.0;
    int filtered = 0;
    bool finite = false;
};

int env_positive_int(const char *name, const int fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallback;

    const int parsed = std::stoi(value);
    if (parsed <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive");
    return parsed;
}

bool env_flag(const char *name, const bool fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallback;

    std::string parsed(value);
    std::transform(parsed.begin(), parsed.end(), parsed.begin(),
                   [](const unsigned char c) { return std::tolower(c); });
    if (parsed == "1" || parsed == "true")
        return true;
    if (parsed == "0" || parsed == "false")
        return false;
    throw std::invalid_argument(
        std::string(name) + " must be one of 0, 1, false, or true");
}

const char *env_text(const char *name)
{
    const char *value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? "unset" : value;
}

double test_element(const int i, const int j, const int n)
{
    if (i == j)
        return 2.0 + static_cast<double>(i + 1) / static_cast<double>(n);
    return 1.0e-2 / (1.0 + std::abs(i - j));
}

WcSqrtBenchmarkResult run_wc_sqrt_benchmark(
    librpa_int::BlacsCtxtHandler &blacs_h, const int n, const int block,
    const bool use_elpa)
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    if (n <= 0 || block <= 0 || block > n)
        throw std::invalid_argument("invalid Wc square-root test dimensions");

#ifndef LIBRPA_USE_ELPA
    if (use_elpa)
        throw std::runtime_error(
            "LIBRPA_TEST_WC_USE_ELPA requested but LIBRPA_USE_ELPA is not compiled");
#endif

    ArrayDesc source_desc(blacs_h);
    ArrayDesc distributed_desc(blacs_h);
    source_desc.init(n, n, n, n, 0, 0);
    distributed_desc.init(n, n, block, block, 0, 0);
#ifdef LIBRPA_USE_ELPA
    if (use_elpa)
        distributed_desc.set_elpa_handle(false);
#endif

    matrix_m<Complex> source(1, 1, MAJOR::COL);
    matrix_m<Complex> gathered_sqrt(1, 1, MAJOR::COL);
    if (myid_global == 0)
    {
        source.resize(n, n);
        gathered_sqrt.resize(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                source(i, j) = Complex(test_element(i, j, n), 0.0);
    }

    auto a_local = init_local_mat<Complex>(distributed_desc, MAJOR::COL);
    auto z_local = init_local_mat<Complex>(distributed_desc, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(
        n, n, source.ptr(), 1, 1, source_desc.desc, a_local.ptr(), 1, 1,
        distributed_desc.desc, blacs_h.ictxt);
    const auto original_local = a_local.copy();

    std::vector<double> eigenvalues(static_cast<std::size_t>(n));
    std::size_t filtered = 0;
    blacs_h.barrier();
    const double sqrt_start = MPI_Wtime();
    LaConnector::power_hemat_la_real<double>(
        a_local, distributed_desc, z_local, distributed_desc, filtered,
        eigenvalues.data(), 0.5, -1.0e5, false, use_elpa);
    blacs_h.barrier();
    const double sqrt_local_seconds = MPI_Wtime() - sqrt_start;

    WcSqrtBenchmarkResult result;
    MPI_Allreduce(&sqrt_local_seconds, &result.sqrt_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, blacs_h.comm());
    result.filtered = static_cast<int>(filtered);

    auto squared_local = init_local_mat<Complex>(distributed_desc, MAJOR::COL);
    blacs_h.barrier();
    const double gemm_start = MPI_Wtime();
    ScalapackConnector::pgemm_f(
        'N', 'N', n, n, n, Complex(1.0, 0.0), a_local.ptr(), 1, 1,
        distributed_desc.desc, a_local.ptr(), 1, 1, distributed_desc.desc,
        Complex(0.0, 0.0), squared_local.ptr(), 1, 1,
        distributed_desc.desc);
    blacs_h.barrier();
    const double gemm_local_seconds = MPI_Wtime() - gemm_start;
    MPI_Allreduce(&gemm_local_seconds, &result.residual_gemm_seconds, 1,
                  MPI_DOUBLE, MPI_MAX, blacs_h.comm());

    double local_error_norm2 = 0.0;
    double local_reference_norm2 = 0.0;
    int local_finite = 1;
    for (std::size_t i = 0; i < original_local.size(); ++i)
    {
        const Complex value = a_local.ptr()[i];
        const Complex squared = squared_local.ptr()[i];
        const Complex reference = original_local.ptr()[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            !std::isfinite(squared.real()) || !std::isfinite(squared.imag()))
            local_finite = 0;
        local_error_norm2 += std::norm(squared - reference);
        local_reference_norm2 += std::norm(reference);
    }

    double global_error_norm2 = 0.0;
    double global_reference_norm2 = 0.0;
    int global_finite = 0;
    MPI_Allreduce(&local_error_norm2, &global_error_norm2, 1, MPI_DOUBLE,
                  MPI_SUM, blacs_h.comm());
    MPI_Allreduce(&local_reference_norm2, &global_reference_norm2, 1,
                  MPI_DOUBLE, MPI_SUM, blacs_h.comm());
    MPI_Allreduce(&local_finite, &global_finite, 1, MPI_INT, MPI_MIN,
                  blacs_h.comm());
    result.relative_residual =
        std::sqrt(global_error_norm2 / global_reference_norm2);
    result.finite = global_finite == 1;

    ScalapackConnector::pgemr2d_f(
        n, n, a_local.ptr(), 1, 1, distributed_desc.desc,
        gathered_sqrt.ptr(), 1, 1, source_desc.desc, blacs_h.ictxt);
    if (myid_global == 0)
    {
        double hermitian_error_norm2 = 0.0;
        double sqrt_norm2 = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const Complex value = gathered_sqrt(i, j);
                hermitian_error_norm2 +=
                    std::norm(value - std::conj(gathered_sqrt(j, i)));
                sqrt_norm2 += std::norm(value);
            }
        result.hermiticity_residual =
            std::sqrt(hermitian_error_norm2 / sqrt_norm2);
    }
    MPI_Bcast(&result.hermiticity_residual, 1, MPI_DOUBLE, 0, blacs_h.comm());
    return result;
}

} // namespace

int main(int argc, char *argv[])
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    init_global_mpi(MPI_COMM_WORLD);
#ifdef LIBRPA_USE_ELPA
    if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        throw std::runtime_error("elpa_init failure");
#endif

    const int grid_size = static_cast<int>(std::sqrt(size_global));
    if (grid_size * grid_size != size_global)
        throw std::runtime_error("test requires a square number of MPI processes");

    BlacsCtxtHandler blacs_h(mpi_comm_global);
    blacs_h.init();
    blacs_h.set_square_grid();

    const int n = env_positive_int("LIBRPA_TEST_WC_DIM", 32);
    const int block = env_positive_int("LIBRPA_TEST_WC_BLOCK", 8);
    const bool use_elpa = env_flag("LIBRPA_TEST_WC_USE_ELPA", false);
    const auto result = run_wc_sqrt_benchmark(blacs_h, n, block, use_elpa);
    assert(result.finite);
    assert(result.filtered == 0);
    assert(result.relative_residual <= 1.0e-9);
    assert(result.hermiticity_residual <= 1.0e-12);

    if (myid_global == 0)
    {
        std::cout << std::setprecision(12)
                  << "WC_SQRT_BENCH"
                  << " solver=" << (use_elpa ? "elpa" : "scalapack")
                  << " n=" << n
                  << " block=" << block
                  << " ranks=" << size_global
                  << " grid=" << blacs_h.nprows << "x" << blacs_h.npcols
                  << " mpi_thread_provided=" << provided
                  << " fi_provider=" << env_text("FI_PROVIDER")
                  << " i_mpi_ofi_provider=" << env_text("I_MPI_OFI_PROVIDER")
                  << " ucx_tls=" << env_text("UCX_TLS")
                  << " sqrt_s=" << result.sqrt_seconds
                  << " residual_gemm_s=" << result.residual_gemm_seconds
                  << " filtered=" << result.filtered
                  << " relres=" << result.relative_residual
                  << " herm=" << result.hermiticity_residual
                  << " finite=" << (result.finite ? 1 : 0)
                  << " status=PASS" << std::endl;
    }

#ifdef LIBRPA_USE_ELPA
    int elpa_error = ELPA_OK;
    elpa_uninit(&elpa_error);
    if (elpa_error != ELPA_OK)
        throw std::runtime_error("elpa_uninit failure");
#endif
    blacs_h.exit();
    finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
