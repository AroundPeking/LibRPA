#include "../core/timefreq.h"

#include "../mpi/global_mpi.h"
#include "../io/global_io.h"
#include "../io/stl_io_helper.h"
#include "../utils/constants.h"

#include "testutils.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace std;
using namespace librpa_int;

namespace
{
void require_near(const double actual, const double expected, const double tolerance)
{
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("values differ beyond tolerance");
}
}

void check_initialize()
{
    cout << "Available time-frequency grids: " << LIBRPA_TFGRID_COUNT << endl;
    // cout << source_dir << endl;
    // cout << minimax_grid_path << endl;
    TFGrids tfg(6);
}

void check_gauss_grids()
{
    TFGrids gl(4);
    gl.generate(LIBRPA_TFGRID_GAUSS_LEGENDRE);
    assert(gl.get_grid_type() == LIBRPA_TFGRID_GAUSS_LEGENDRE);
    assert(!gl.has_time_grids());

    const auto gl_freq = gl.get_freq_nodes();
    const auto gl_weight = gl.get_freq_weights();
    const vector<double> gl_ref_freq = {
        0.037306157410633582,
        0.24627921401387612,
        1.0151079984602933,
        6.7013066301151962,
    };
    const vector<double> gl_ref_weight = {
        0.10042496565842267,
        0.36320093923997282,
        1.4970332756138096,
        18.039340819487776,
    };
    const double tol = 1e-13;
    for (size_t i = 0; i != gl.size(); ++i)
    {
        assert(std::isfinite(gl_freq[i]) && gl_freq[i] > 0.0);
        assert(std::isfinite(gl_weight[i]) && gl_weight[i] > 0.0);
        assert(std::fabs(gl_freq[i] - gl_ref_freq[i]) < tol);
        assert(std::fabs(gl_weight[i] - gl_ref_weight[i]) < tol);
        require_near(gl.find_correlation_frequency_weight(gl_freq[i]), gl_ref_weight[i] / TWO_PI,
                     tol);
        if (i > 0) assert(gl_freq[i - 1] < gl_freq[i]);
    }

    TFGrids split_gl(4);
    split_gl.generate(LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE, -1, -1, 2.0);
    assert(split_gl.get_grid_type() == LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE);
    assert(!split_gl.has_time_grids());

    const auto split_freq = split_gl.get_freq_nodes();
    const auto split_weight = split_gl.get_freq_weights();
    const vector<double> split_ref_freq = {
        0.42264973081037427,
        2.2540333075851664,
        4.0,
        17.745966692414836,
    };
    const vector<double> split_ref_weight = {
        1.0,
        0.70564807662546158,
        3.5555555555555558,
        43.738796367818985,
    };
    for (size_t i = 0; i != split_gl.size(); ++i)
    {
        assert(std::isfinite(split_freq[i]) && split_freq[i] > 0.0);
        assert(std::isfinite(split_weight[i]) && split_weight[i] > 0.0);
        assert(std::fabs(split_freq[i] - split_ref_freq[i]) < tol);
        assert(std::fabs(split_weight[i] - split_ref_weight[i]) < tol);
        if (i > 0) assert(split_freq[i - 1] < split_freq[i]);
    }
}

void check_finite_beta_matsubara_grid()
{
    constexpr std::size_t nfreq = 3;
    constexpr std::size_t ntau = 4;
    constexpr double beta = 8.0;
    constexpr double tol = 1.0e-13;

    TFGrids tfg(nfreq);
    tfg.generate_finite_beta_matsubara(ntau, beta);
    assert(tfg.get_grid_type() == LIBRPA_TFGRID_FD_MATSUBARA);
    assert(tfg.get_n_grids() == nfreq);
    assert(tfg.get_n_time_grids() == ntau);

    const auto frequencies = tfg.get_freq_nodes();
    const auto frequency_weights = tfg.get_freq_weights();
    const auto times = tfg.get_time_nodes();
    const auto time_weights = tfg.get_time_weights();
    for (std::size_t itau = 0; itau != ntau; ++itau)
    {
        require_near(times[itau], (itau + 0.5) * beta / ntau, tol);
        require_near(time_weights[itau], beta / ntau, tol);
    }
    for (std::size_t ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        require_near(frequencies[ifreq], TWO_PI * ifreq / beta, tol);
        require_near(frequency_weights[ifreq],
                     (ifreq == 0 ? 0.5 * TWO_PI : TWO_PI) / beta, tol);
        require_near(tfg.find_correlation_frequency_weight(frequencies[ifreq]),
                     (ifreq == 0 ? 0.5 : 1.0) / beta, tol);
    }

    const auto &fourier = tfg.get_fourier_t2f();
    assert(fourier.nr == static_cast<int>(nfreq));
    assert(fourier.nc == static_cast<int>(ntau));
    for (std::size_t ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        std::complex<double> transformed_constant = 0.0;
        for (std::size_t itau = 0; itau != ntau; ++itau)
        {
            const auto expected = (beta / ntau)
                                  * std::exp(std::complex<double>(0.0, 1.0)
                                             * frequencies[ifreq] * times[itau]);
            require_near(std::abs(fourier(ifreq, itau) - expected), 0.0, tol);
            require_near(
                std::abs(tfg.get_time_to_frequency_factor(ifreq, itau) - expected), 0.0, tol);
            transformed_constant += fourier(ifreq, itau);
        }
        require_near(std::abs(transformed_constant - (ifreq == 0 ? beta : 0.0)), 0.0, tol);
    }

    bool rejected = false;
    try
    {
        tfg.generate_finite_beta_matsubara(0, beta);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

double finite_beta_rpa_model_sum(const std::size_t nfreq, const double beta, const double pole,
                                 const bool add_power4_tail)
{
    TFGrids tfg(nfreq);
    tfg.generate_finite_beta_matsubara(8, beta);
    double sum = 0.0;
    for (const double frequency : tfg.get_freq_nodes())
    {
        const double denominator = frequency * frequency + pole * pole;
        const double integrand = std::pow(pole, 4) / (denominator * denominator);
        sum += tfg.find_correlation_frequency_weight(frequency) * integrand;
    }
    if (add_power4_tail) sum += std::pow(pole, 4) * tfg.finite_beta_power4_tail_weight(nfreq);
    return sum;
}

void check_finite_beta_rpa_sum_and_tail()
{
    constexpr double beta = 8.0;
    constexpr double pole = 0.7;
    const double x = 0.5 * beta * pole;
    const double exact =
        pole / (8.0 * std::tanh(x)) + beta * pole * pole / (16.0 * std::pow(std::sinh(x), 2));

    const double raw_16_error = std::abs(finite_beta_rpa_model_sum(16, beta, pole, false) - exact);
    const double raw_32_error = std::abs(finite_beta_rpa_model_sum(32, beta, pole, false) - exact);
    const double corrected_32_error =
        std::abs(finite_beta_rpa_model_sum(32, beta, pole, true) - exact);

    if (!(raw_16_error > 5.0 * raw_32_error && raw_16_error < 12.0 * raw_32_error))
        throw std::runtime_error(
            "finite-beta RPA model tail does not show the expected N^-3 decay");
    if (!(corrected_32_error < 0.02 * raw_32_error))
        throw std::runtime_error("leading power-four Matsubara tail did not improve the model sum");
}

void check_minimax_ng16_diamond_k222()
{
    TFGrids tfg(16);
    // Check minimax grids
    // emin and emax (in Hartree) from diamond PBE k2c
    double emin = 0.173388;
    double emax = 23.7738;
    double residual = tfg.generate_minimax(emin, emax);
    cout << tfg.get_freq_nodes() << endl;
    // TODO(minye): CI failed after this part. I am completely ignorant of why it should.
    cout << tfg.get_freq_weights() << endl;
    cout << tfg.get_time_nodes() << endl;
    cout << tfg.get_time_weights() << endl;
    cout << "Residual: " << residual << endl;
    /* if (tfg.get_grid_type() != LIBRPA_TFGRID_MINIMAX) */
    /*     throw logic_error("internal type should be minimax grid"); */
    /* print_matrix("cos: t2f * f2t, ideally close to identity", tfg.get_costrans_t2f() * tfg.get_costrans_f2t() ); */
    /* print_matrix("sin: t2f * f2t, ideally close to identity", tfg.get_sintrans_t2f() * tfg.get_sintrans_f2t() ); */
    // tfg.write_cos_sin_trans_matrices("test.dat");
}

void check_minimax_ng6_HF_123()
{
    // TODO(minyez): the correct data vary due to the version of GreenX data.
    // The following is correct for the local grid data, but not for the latest GreenX library
    // Need to adapt later.
    TFGrids tfg(6);
    double emin = 0.657768, emax = 30.1366;
    tfg.generate_minimax(emin, emax);
    assert ( tfg.size() == 6 );
    /* printf("%20.15f, %20.15f\n", tfg.get_freq_nodes()[0], tfg.get_freq_weights()[0]); */
    vector<double> freq_node = {0.233556, 0.844872, 2.029850, 4.815547, 12.239097, 36.979336};
    vector<double> freq_weight = {0.489089, 0.798829, 1.724256, 4.282121, 12.092283, 48.530744};
    vector<double> time_node = {0.021614, 0.129568, 0.408121, 1.072294, 2.593619, 6.074920};
    vector<double> time_weight = {0.057049, 0.171324, 0.419863, 0.982601, 2.222569, 5.258784};
    /* cout << tfg.find_freq_weight(tfg.get_freq_nodes()[0]) << endl; */
    assert( 1 == tfg.get_freq_index(tfg.get_freq_nodes()[1]));
    assert( tfg.find_freq_weight(tfg.get_freq_nodes()[0]) == tfg.get_freq_weights()[0]);
    assert( 2 == tfg.get_time_index(tfg.get_time_nodes()[2]));
    auto i_tf = tfg.get_tf_index({tfg.get_time_nodes()[2], tfg.get_freq_nodes()[1]});
    assert(i_tf.first  == tfg.get_time_index(tfg.get_time_nodes()[2]));
    assert(i_tf.second  == tfg.get_time_index(tfg.get_time_nodes()[1]));
    for ( size_t i = 0; i != tfg.size(); i++ )
    {
        assert( fabs(freq_node[i] - tfg.get_freq_nodes()[i]) < 1e-5);
        assert( fabs(freq_weight[i] - tfg.get_freq_weights()[i]) < 1e-5);
        // assert( fabs(time_node[i] - tfg.get_time_nodes()[i]) < 1e-5);
        // assert( fabs(time_weight[i] - tfg.get_time_weights()[i]) < 1e-5);
    }
    // matrix costrans_t2f(6, 6);
    // costrans_t2f(0, 0) = 0.11418;
    // costrans_t2f(0, 1) = 0.34210;
    // costrans_t2f(0, 2) = 0.83715;
    // costrans_t2f(0, 3) = 1.90047;
    // costrans_t2f(0, 4) = 3.66221;
    // costrans_t2f(0, 5) = 1.66271;
    // costrans_t2f(1, 0) = 0.11384;
    // costrans_t2f(1, 1) = 0.34204;
    // costrans_t2f(1, 2) = 0.78564;
    // costrans_t2f(1, 3) = 1.21529;
    // costrans_t2f(1, 4) = -2.37093;
    // costrans_t2f(1, 5) = -2.80551;
    // costrans_t2f(2, 0) = 0.11532;
    // costrans_t2f(2, 1) = 0.32304;
    // costrans_t2f(2, 2) = 0.59313;
    // costrans_t2f(2, 3) = -1.09755;
    // costrans_t2f(2, 4) = -0.39995;
    // costrans_t2f(2, 5) = 2.31135;
    // costrans_t2f(3, 0) = 0.11016;
    // costrans_t2f(3, 1) = 0.29865;
    // costrans_t2f(3, 2) =-0.37431;
    // costrans_t2f(3, 3) =-0.21496;
    // costrans_t2f(3, 4) = 0.52021;
    // costrans_t2f(3, 5) =-1.71560;
    // costrans_t2f(4, 0) = 0.11853;
    // costrans_t2f(4, 1) =-0.05220;
    // costrans_t2f(4, 2) =-0.16181;
    // costrans_t2f(4, 3) = 0.21842;
    // costrans_t2f(4, 4) =-0.36024;
    // costrans_t2f(4, 5) = 1.24194;
    // costrans_t2f(5, 0) = 0.04570;
    // costrans_t2f(5, 1) =-0.08751;
    // costrans_t2f(5, 2) = 0.08781;
    // costrans_t2f(5, 3) =-0.10941;
    // costrans_t2f(5, 4) = 0.19074;
    // costrans_t2f(5, 5) =-0.67673;
    // assert ( is_mat_A_equal_B(6, 6, costrans_t2f.c, tfg.get_costrans_t2f().c, false, false, 1e-5) );
    /* print_matrix("cos: t2f * f2t, ideally close to identity", tfg.get_costrans_t2f() * tfg.get_costrans_f2t() ); */
    /* print_matrix("sin: t2f * f2t, ideally close to identity", tfg.get_sintrans_t2f() * tfg.get_sintrans_f2t() ); */

/* Sine transform matrix */
/*  */
/*   */
/*    0.00072   0.00950   0.08336   0.47439   2.58887   9.87802 */
/*    0.00074   0.04589   0.24892   1.68667   2.90489  -6.70134 */
/*    0.00845   0.06669   0.72122   1.11313  -1.91638   3.95426 */
/*    0.00614   0.24116   0.53245  -0.64767   0.70235  -2.01430 */
/*    0.03966   0.25076  -0.20340   0.15712  -0.22819   0.75918 */
/*    0.06496  -0.00675  -0.01183   0.01955  -0.03648   0.13263 */
}

void check_minimax_ng32_H2O()
{
    // For large minimax grids, high precision is required to reproduce the actual results
    const double emin = 0.302066195818780992e-2 + 0.260399583086844855e0;
    const double emax = 0.322960860707412234e1 + 0.187886173733204913e2;
    // cout << "emin/erange: " << emin << " " << emax/emin << endl;
    double residual;

    TFGrids tfg(32);
    residual = tfg.generate_minimax(emin, emax);
    cout << "Residual: " << residual << endl;
    // tfg.write_cos_sin_trans_matrices("test.dat");
    residual = tfg.generate_minimax(emin, emax, 1e-12);
    cout << "Residual (regulation): " << residual << endl;
    // tfg.write_cos_sin_trans_matrices("test_regulated.dat");
}

int main (int argc, char **argv)
{
    using namespace librpa_int::global;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    init_global_mpi(MPI_COMM_WORLD);
    init_global_io();

    check_initialize();
    check_gauss_grids();
    check_finite_beta_matsubara_grid();
    check_finite_beta_rpa_sum_and_tail();
    check_minimax_ng16_diamond_k222();
    check_minimax_ng6_HF_123();
    check_minimax_ng32_H2O();

    finalize_global_io();
    finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
