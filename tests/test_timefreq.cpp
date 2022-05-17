#include "timefreq.h"
#include <iostream>
#include <stdexcept>

void check_minimax_ng16_diamond_k222()
{
    TFGrids tfg(16);
    // Check minimax grids
    // emin and emax (in Hartree) from diamond PBE k2c
    double emin = 0.173388;
    double emax = 23.7738;
    tfg.generate_minimax(emin, emax);
    /* if (tfg.get_grid_type() != TFGrids::GRID_TYPES::Minimax) */
    /*     throw logic_error("internal type should be minimax grid"); */
}

void check_minimax_ng6_HF_123()
{
    TFGrids tfg(6);
    double emin = 0.657768, emax = 30.1366;
    tfg.generate_minimax(emin, emax);
    assert ( tfg.size() == 6 );
    vector<double> freq_node = {0.233556, 0.844872, 2.029850, 4.815547, 12.239097, 36.979336};
    vector<double> freq_weight = {0.489089, 0.798829, 1.724256, 4.282121, 12.092283, 48.530744};
    vector<double> time_node = {0.021614, 0.129568, 0.408121, 1.072294, 2.593619, 6.074920};
    vector<double> time_weight = {0.057049, 0.171324, 0.419863, 0.982601, 2.222569, 5.258784};
    for ( int i = 0; i != tfg.size(); i++ )
    {
        assert( fabs(freq_node[i] - tfg.get_freq_nodes()[i]) < 1e-5);
        assert( fabs(freq_weight[i] - tfg.get_freq_weights()[i]) < 1e-5);
        assert( fabs(time_node[i] - tfg.get_time_nodes()[i]) < 1e-5);
        assert( fabs(time_weight[i] - tfg.get_time_weights()[i]) < 1e-5);
    }
    matrix costrans_t2f(6, 6);
    costrans_t2f(0, 0) = 0.11418;
    costrans_t2f(0, 1) = 0.34210;
    costrans_t2f(0, 2) = 0.83715;
    costrans_t2f(0, 3) = 1.90047;
    costrans_t2f(0, 4) = 3.66221;
    costrans_t2f(0, 5) = 1.66271;
    costrans_t2f(1, 0) = 0.11384;
    costrans_t2f(1, 1) = 0.34204;
    costrans_t2f(1, 2) = 0.78564;
    costrans_t2f(1, 3) = 1.21529;
    costrans_t2f(1, 4) = -2.37093;
    costrans_t2f(1, 5) = -2.80551;
    costrans_t2f(2, 0) = 0.11532;
    costrans_t2f(2, 1) = 0.32304;
    costrans_t2f(2, 2) = 0.59313;
    costrans_t2f(2, 3) = -1.09755;
    costrans_t2f(2, 4) = -0.39995;
    costrans_t2f(2, 5) = 2.31135;
    costrans_t2f(3, 0) = 0.11016;
    costrans_t2f(3, 1) = 0.29865;
    costrans_t2f(3, 2) =-0.37431;
    costrans_t2f(3, 3) =-0.21496;
    costrans_t2f(3, 4) = 0.52021;
    costrans_t2f(3, 5) =-1.71560;
    costrans_t2f(4, 0) = 0.11853;
    costrans_t2f(4, 1) =-0.05220;
    costrans_t2f(4, 2) =-0.16181;
    costrans_t2f(4, 3) = 0.21842;
    costrans_t2f(4, 4) =-0.36024;
    costrans_t2f(4, 5) = 1.24194;
    costrans_t2f(5, 0) = 0.04570;
    costrans_t2f(5, 1) =-0.08751;
    costrans_t2f(5, 2) = 0.08781;
    costrans_t2f(5, 3) =-0.10941;
    costrans_t2f(5, 4) = 0.19074;
    costrans_t2f(5, 5) =-0.67673;
    print_matrix("", costrans_t2f);
    print_matrix("", tfg.get_costrans_t2f());
    for ( int i = 0; i != costrans_t2f.size; i++ )
    {
        cout << costrans_t2f.c[i] << " " << tfg.get_costrans_t2f().c[i] << endl;
        assert ( fabs( costrans_t2f.c[i] - tfg.get_costrans_t2f().c[i]) < 2e-5);
    }

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

int main ()
{
    check_minimax_ng16_diamond_k222();
    check_minimax_ng6_HF_123();
    return 0;
}
