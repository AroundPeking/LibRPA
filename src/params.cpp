#include "params.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils_io.h"

// default setting

std::string Params::task = "rpa";
std::string Params::output_file = "stdout";
std::string Params::output_dir = "librpa.d/";
std::string Params::tfgrids_type = "minimax";
std::string Params::DFT_software = "auto";
std::string Params::parallel_routing = "auto";

int Params::nfreq = 0;
int Params::n_params_anacon = -1;
int Params::option_dielect_func = 2;
double Params::absorption_omega_max_ev = -1.0;
double Params::absorption_domega_ev = -1.0;
double Params::absorption_eta_ev = 0.0;
std::string Params::absorption_imag_axis_input = "";
std::string Params::absorption_continuation_method = "pade";
int Params::absorption_n_poles = 3;

double Params::gf_R_threshold = 1e-4;
double Params::cs_threshold = 1e-4;
double Params::vq_threshold = 0;
double Params::sqrt_coulomb_threshold = 1e-8;
double Params::libri_chi0_threshold_C = 0.0;
double Params::libri_chi0_threshold_G = 0.0;
double Params::libri_exx_threshold_C = 0.0;
double Params::libri_exx_threshold_D = 0.0;
double Params::libri_exx_threshold_V = 0.0;
double Params::libri_g0w0_threshold_C = 0.0;
double Params::libri_g0w0_threshold_G = 0.0;
double Params::libri_g0w0_threshold_Wc = 0.0;
double Params::minimax_min_gap = -1.0;
double Params::minimax_max_transition = -1.0;
bool Params::use_fullcoul_exx = false;
bool Params::use_abacus_exx_symmetry = true;
bool Params::use_abacus_gw_symmetry = true;
bool Params::output_abacus_gw_gf = false;
bool Params::use_fullcoul_wc = false;

bool Params::use_scalapack_ecrpa = true;
bool Params::use_scalapack_gw_wc = false;
bool Params::debug = false;
bool Params::replace_w_head = true;
bool Params::use_shrink_chi = true;
bool Params::use_shrink_abfs = false;
bool Params::use_soc = false;
bool Params::use_2d_dielectric = false;
bool Params::use_pyatb = true;

bool Params::band_continue = false;

/* ==========================================================
 * output options begin
 * ========================================================== */
int Params::output_Wc_Rf_mat = 0;
bool Params::output_energy_qp = false;
bool Params::output_gw_sigc_mat = false;
bool Params::output_gw_sigc_mat_rt = false;
bool Params::output_gw_sigc_mat_rf = false;
bool Params::output_hamgnn = false;
int Params::nbands_G = -1;
/* ==========================================================
 * output options end
 * ========================================================== */

void Params::check_consistency()
{
    std::transform(absorption_continuation_method.begin(), absorption_continuation_method.end(),
                   absorption_continuation_method.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (n_params_anacon < 0)
    {
        n_params_anacon = nfreq;
    }
    if (absorption_eta_ev < 0.0)
    {
        throw std::runtime_error("absorption_eta_ev must be non-negative");
    }
    if (absorption_n_poles <= 0)
    {
        throw std::runtime_error("absorption_n_poles must be positive");
    }
    if (absorption_continuation_method != "pade"
        && absorption_continuation_method != "multipole"
        && absorption_continuation_method != "both")
    {
        throw std::runtime_error(
            "absorption_continuation_method must be one of: pade, multipole, both");
    }
}

void Params::print()
{
    const std::vector<std::pair<std::string, double>> double_params{
        {"gf_R_threshold", gf_R_threshold},
        {"absorption_omega_max_ev", absorption_omega_max_ev},
        {"absorption_domega_ev", absorption_domega_ev},
        {"absorption_eta_ev", absorption_eta_ev},
        {"cs_R_threshold", cs_threshold},
        {"vq_threshold", vq_threshold},
        {"sqrt_coulomb_threshold", sqrt_coulomb_threshold},
        {"libri_chi0_threshold_C", libri_chi0_threshold_C},
        {"libri_chi0_threshold_G", libri_chi0_threshold_G},
        {"libri_exx_threshold_C", libri_exx_threshold_C},
        {"libri_exx_threshold_D", libri_exx_threshold_D},
        {"libri_exx_threshold_V", libri_exx_threshold_V},
        {"libri_g0w0_threshold_C", libri_g0w0_threshold_C},
        {"libri_g0w0_threshold_G", libri_g0w0_threshold_G},
        {"libri_g0w0_threshold_Wc", libri_g0w0_threshold_Wc},
        {"minimax_min_gap", minimax_min_gap},
        {"minimax_max_transition", minimax_max_transition},
    };

    const std::vector<std::pair<std::string, int>> int_params{
        {"nfreq", nfreq},
        {"n_params_anacon", n_params_anacon},
        {"absorption_n_poles", absorption_n_poles},
        {"option_dielect_func", option_dielect_func},
        {"output_Wc_Rf_mat", output_Wc_Rf_mat},
        {"nbands_G", nbands_G},
    };

    const std::vector<std::pair<std::string, std::string>> str_params{
        {"task", task},
        {"output_dir", output_dir},
        {"output_file", output_file},
        {"tfgrids_type", tfgrids_type},
        {"absorption_imag_axis_input", absorption_imag_axis_input},
        {"absorption_continuation_method", absorption_continuation_method},
        {"parallel_routing", parallel_routing},
    };

    const std::vector<std::pair<std::string, bool>> bool_params{
        {"debug", debug},
        {"band_continue", band_continue},
        {"use_scalapack_ecrpa", use_scalapack_ecrpa},
        {"use_scalapack_gw_wc", use_scalapack_gw_wc},
        {"output_energy_qp", output_energy_qp},
        {"output_gw_sigc_mat", output_gw_sigc_mat},
        {"output_gw_sigc_mat_rt", output_gw_sigc_mat_rt},
        {"output_gw_sigc_mat_rf", output_gw_sigc_mat_rf},
        {"replace_w_head", replace_w_head},
        {"use_shrink_abfs", use_shrink_abfs},
        {"use_shrink_chi", use_shrink_chi},
        {"use_soc", use_soc},
        {"use_fullcoul_exx", use_fullcoul_exx},
        {"use_abacus_exx_symmetry", use_abacus_exx_symmetry},
        {"use_abacus_gw_symmetry", use_abacus_gw_symmetry},
        {"output_abacus_gw_gf", output_abacus_gw_gf},
        {"use_fullcoul_wc", use_fullcoul_wc},
        {"output_hamgnn", output_hamgnn},
        {"use_2d_dielectric", use_2d_dielectric},
        {"use_pyatb", use_pyatb},
    };

    for (const auto &param : str_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second.c_str());

    for (const auto &param : int_params)
        LIBRPA::utils::lib_printf("%s = %d\n", param.first.c_str(), param.second);

    for (const auto &param : double_params)
        LIBRPA::utils::lib_printf("%s = %f\n", param.first.c_str(), param.second);

    for (const auto &param : bool_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second ? "T" : "F");
}

Params params;
