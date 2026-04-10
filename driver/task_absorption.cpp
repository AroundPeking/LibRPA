#include "task_absorption.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "chi0.h"
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_mpi.h"
#include "epsilon.h"
#include "optics.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "ri.h"
#include "utils_timefreq.h"

namespace
{

bool reuse_saved_imag_axis()
{
    return !Params::absorption_imag_axis_input.empty();
}

std::vector<double> build_absorption_real_mesh(const LIBRPA::optics::ImagAxisOpticsData& imag_data)
{
    if (Params::absorption_omega_max_ev > 0.0 && Params::absorption_domega_ev > 0.0)
    {
        return LIBRPA::optics::build_real_axis_mesh(
            Params::absorption_omega_max_ev / HA2EV, Params::absorption_domega_ev / HA2EV);
    }

    const int n_real_freq = std::max(200, 10 * Params::n_params_anacon);
    return LIBRPA::optics::build_real_axis_mesh(imag_data.imag_freqs, n_real_freq);
}

void write_continued_outputs(const LIBRPA::optics::ImagAxisOpticsData& imag_data,
                             const LIBRPA::optics::RealAxisOpticsData& real_data,
                             const std::string& method, const int continuation_order,
                             const double eta_ha, const bool write_primary,
                             const bool write_named_copy)
{
    const std::string method_suffix = "_" + method;
    const auto write_bundle = [&](const std::string& suffix) {
        LIBRPA::optics::write_real_axis_data(
            Params::output_dir + "/absorption_real_axis" + suffix + ".dat", real_data);
        LIBRPA::optics::write_absorption_spectrum(
            Params::output_dir + "/absorption_spectrum" + suffix + ".dat", real_data);
        LIBRPA::optics::write_summary(
            Params::output_dir + "/absorption_summary" + suffix + ".txt", imag_data, real_data,
            method, continuation_order, eta_ha);
    };

    if (write_primary)
    {
        write_bundle("");
    }
    if (write_named_copy)
    {
        write_bundle(method_suffix);
    }
}

} // namespace

void task_absorption(std::map<Vector3_Order<double>, ComplexMatrix>& sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::utils::lib_printf;

    Profiler::start("absorption", "Build optical absorption response");

    if (!Params::replace_w_head || Params::option_dielect_func != 3)
    {
        throw std::runtime_error(
            "task=absorption currently requires `replace_w_head = true` and"
            " `option_dielect_func = 3`");
    }

    LIBRPA::optics::ImagAxisOpticsData imag_data;
    if (reuse_saved_imag_axis())
    {
        imag_data = LIBRPA::optics::read_imag_axis_data(Params::absorption_imag_axis_input);
    }
    else
    {
        Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
        auto Rlist = construct_R_grid(period);

        auto gamma_q_iter = std::find_if(klist.cbegin(), klist.cend(), [](const auto& q) {
            return is_gamma_point(q);
        });
        if (gamma_q_iter == klist.cend())
        {
            throw std::runtime_error("task=absorption failed to locate Gamma in the loaded q-grid");
        }
        std::vector<Vector3_Order<double>> qlist{*gamma_q_iter};

        auto tfg =
            LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);

        Chi0 chi0(meanfield, klist, tfg);
        chi0.gf_R_threshold = Params::gf_R_threshold;
        chi0.set_input_dir(driver_params.input_dir);

        // Reuse the existing head/wing preparation path to initialize df_headwing on the target grid.
        interpolate_dielec_func(Params::option_dielect_func, {}, {}, chi0.tfg.get_freq_nodes());

        Profiler::start("chi0_build", "Build response function chi0 for Gamma");
        chi0.build(Cs_data, Rlist, period, local_atpair, qlist, sinvS);
        Profiler::stop("chi0_build");

        imag_data = compute_absorption_imag_axis_blacs(chi0, Vq);
    }

    LIBRPA::optics::write_imag_axis_data(Params::output_dir + "/absorption_imag_axis.dat",
                                         imag_data);

    const auto real_mesh = build_absorption_real_mesh(imag_data);
    const double eta_ha = Params::absorption_eta_ev / HA2EV;

    LIBRPA::optics::RealAxisOpticsData primary_real_data;
    std::string primary_method = Params::absorption_continuation_method;
    int primary_order = Params::n_params_anacon;

    if (Params::absorption_continuation_method == "pade"
        || Params::absorption_continuation_method == "both")
    {
        const auto real_data =
            LIBRPA::optics::continue_to_real_axis(imag_data, real_mesh, Params::n_params_anacon,
                                                  eta_ha);
        const bool write_primary = (Params::absorption_continuation_method == "pade");
        write_continued_outputs(imag_data, real_data, "pade", Params::n_params_anacon, eta_ha,
                                write_primary,
                                Params::absorption_continuation_method == "both");
        if (write_primary)
        {
            primary_real_data = real_data;
        }
    }

    if (Params::absorption_continuation_method == "multipole"
        || Params::absorption_continuation_method == "both")
    {
        const auto real_data = LIBRPA::optics::continue_to_real_axis_multipole(
            imag_data, real_mesh, Params::absorption_n_poles, eta_ha);
        write_continued_outputs(imag_data, real_data, "multipole", Params::absorption_n_poles,
                                eta_ha, true,
                                Params::absorption_continuation_method == "both");
        primary_real_data = real_data;
        primary_method = "multipole";
        primary_order = Params::absorption_n_poles;
    }

    if (mpi_comm_global_h.is_root())
    {
        if (reuse_saved_imag_axis())
        {
            lib_printf("Absorption reused imaginary-axis data from %s\n",
                       Params::absorption_imag_axis_input.c_str());
        }
        lib_printf("Absorption outputs written to %s using %s continuation (order=%d)\n",
                   Params::output_dir.c_str(), primary_method.c_str(), primary_order);
        if (!real_mesh.empty())
        {
            const double real_step_ev =
                real_mesh.size() > 1 ? (real_mesh[1] - real_mesh[0]) * HA2EV : 0.0;
            lib_printf("absorption real-axis mesh: n=%zu, omega_max=%12.6f eV, domega=%12.6f eV, eta=%12.6f eV\n",
                       real_mesh.size(), real_mesh.back() * HA2EV, real_step_ev,
                       Params::absorption_eta_ev);
        }
        if (!imag_data.epsm_avg.empty())
        {
            lib_printf("eps_M(iw=0) estimate: (%18.10f, %18.10f)\n",
                       imag_data.epsm_avg.front().real(), imag_data.epsm_avg.front().imag());
        }
        if (Params::absorption_continuation_method == "both")
        {
            lib_printf("Additional comparison outputs written with suffixes _pade and _multipole\n");
        }
    }

    Profiler::stop("absorption");
}
