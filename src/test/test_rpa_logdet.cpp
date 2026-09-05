#include <mpi.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;
using Complex = std::complex<double>;

Complex evaluate(const std::vector<std::vector<Complex>> &dense, const BlacsCtxtHandler &blacs,
                 int block = 1)
{
    const int n = dense.size();
    ArrayDesc desc(blacs);
    desc.init(n, n, block, block, 0, 0);
    auto local = init_local_mat<Complex>(desc, MAJOR::COL);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int row = desc.indx_g2l_r(i), col = desc.indx_g2l_c(j);
            if (row >= 0 && col >= 0) local(row, col) = dense[i][j];
        }
    std::vector<int> pivots(desc.m_loc() + desc.mb());
    int info = 0;
    return compute_pi_det_blacs_2d(local, desc, pivots.data(), info);
}

void check(const std::vector<std::vector<Complex>> &dense, Complex expected,
           const BlacsCtxtHandler &blacs, int block = 1)
{
    const auto actual = evaluate(dense, blacs, block);
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual.real() - expected.real()) > 2e-12 ||
        std::abs(std::polar(1.0, actual.imag()) - std::polar(1.0, expected.imag())) > 2e-12 ||
        std::abs(actual.imag()) > std::acos(-1.0) + 1e-12)
    {
        std::cerr << "logdet actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("LU determinant phase or log magnitude is incorrect");
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    {
        BlacsCtxtHandler blacs(MPI_COMM_WORLD);
        blacs.init();
        blacs.set_square_grid();
        for (int block : {1, 2, 4})
        {
            check({{2.0}}, std::log(2.0), blacs, block);
            check({{2.0, {0.0, -4.0}}, {{0.0, 4.0}, 18.0}}, std::log(20.0), blacs, block);
        }
        // Positive eigenvalues, determinant 20, but LU swaps nearly imaginary pivots.
        for (double shift : {0.0, 1e-12, -1e-12})
        {
            check({{2.0, {shift, -4.0}}, {{shift, 4.0}, 18.0}}, std::log(20.0 - shift * shift),
                  blacs);
            check({{18.0, {shift, 4.0}}, {{shift, -4.0}, 2.0}}, std::log(20.0 - shift * shift),
                  blacs);
        }
        check({{0.0, 1.0}, {1.0, 0.0}}, std::log(Complex(-1, 0)), blacs);
        check({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}}, 0.0, blacs);
        check({{{1.0, 0.3}, 2.0}, {{0.0, -3.0}, 0.2}}, std::log(Complex(0.2, 6.06)), blacs);
        const Complex z = std::polar(2.0, 2.0);
        check({{z, 0.0}, {0.0, z}}, std::log(z * z), blacs);
        check({{2.0, {0.0, -4.0}, 0.0, 0.0},
               {{0.0, 4.0}, 18.0, 0.0, 0.0},
               {0.0, 0.0, 2.0, {0.0, -4.0}},
               {0.0, 0.0, {0.0, 4.0}, 18.0}},
              std::log(400.0), blacs);
        for (const auto &bad : std::vector<std::vector<std::vector<Complex>>>{
                 {{1.0, 1.0}, {1.0, 1.0}},
                 {{std::numeric_limits<double>::quiet_NaN(), 0.0}, {0.0, 1.0}}})
        {
            bool rejected = false;
            try
            {
                evaluate(bad, blacs);
            }
            catch (const std::runtime_error &)
            {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("singular/nonfinite logdet accepted");
        }
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
}
