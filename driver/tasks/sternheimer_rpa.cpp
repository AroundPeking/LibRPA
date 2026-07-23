#include "../../src/core/sternheimer_rpa.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../src/api/dataset_helper.h"
#include "../../src/api/instance_manager.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../reader_sternheimer.h"
#include "../reader_sternheimer_partial.h"
#include "../reader_sternheimer_qpoints.h"
#include "../rpa_qsum.h"
#include "../sternheimer_partial_reconstruction.h"
#include "../task.h"

void driver::task_sternheimer_rpa()
{
    using librpa_int::compute_sternheimer_rpa_frequency;
    using librpa_int::global::lib_printf;
    using librpa_int::global::lib_printf_root;
    using librpa_int::global::mpi_comm_global_h;

    struct QResult
    {
        struct SymmetryAudit
        {
            int ifreq = 0;
            int full_k_count = 0;
            int representative_k_count = 0;
            int little_group_order = 0;
            int qstar_size = 0;
            double hermiticity_relative_residual = 0.0;
            double max_integrand_difference = 0.0;
            std::complex<double> response_trace = {0.0, 0.0};
        };

        SternheimerQPoint point;
        std::vector<librpa_int::SternheimerRpaFrequencyResult> frequencies;
        std::vector<SymmetryAudit> symmetry_audits;
        std::complex<double> energy = {0.0, 0.0};
    };

    const bool partial_mode = !driver_params.fn_sternheimer_partial_manifest.empty();
    const bool write_reconstructed =
        partial_mode && !driver_params.prefix_sternheimer_reconstructed.empty();
    validate_sternheimer_partial_task_contract(driver_params.fn_sternheimer_qpoints,
                                               driver_params.fn_sternheimer_partial_manifest);

    std::string manifest_path;
    std::string partial_manifest_path;
    std::vector<SternheimerQPoint> qpoints;
    if (driver_params.fn_sternheimer_qpoints.empty())
    {
        if (driver_params.sternheimer_iq <= 0)
        {
            throw std::runtime_error("sternheimer_iq must be a positive one-based q index");
        }
        if (!std::isfinite(driver_params.sternheimer_qweight) ||
            driver_params.sternheimer_qweight <= 0.0)
        {
            throw std::runtime_error("sternheimer_qweight must be positive");
        }
        qpoints.push_back(
            {driver_params.sternheimer_iq, {0.0, 0.0, 0.0}, driver_params.sternheimer_qweight});
    }
    else
    {
        manifest_path = librpa_int::is_absolute_path(driver_params.fn_sternheimer_qpoints)
                            ? driver_params.fn_sternheimer_qpoints
                            : librpa_int::join_path(driver_params.input_dir,
                                                    driver_params.fn_sternheimer_qpoints);
        qpoints = read_sternheimer_qpoint_manifest(manifest_path);
    }
    if (!driver_params.use_rpa_gamma && manifest_path.empty())
    {
        throw std::runtime_error(
            "use_rpa_gamma=false requires fn_sternheimer_qpoints with the full q mesh");
    }
    if (partial_mode)
    {
        validate_sternheimer_partial_qpoint_input_files(qpoints, driver_params.input_dir,
                                                        driver_params.prefix_coul_full,
                                                        driver_params.use_rpa_gamma);
        partial_manifest_path =
            librpa_int::is_absolute_path(driver_params.fn_sternheimer_partial_manifest)
                ? driver_params.fn_sternheimer_partial_manifest
                : librpa_int::join_path(driver_params.input_dir,
                                        driver_params.fn_sternheimer_partial_manifest);
    }
    else
    {
        validate_sternheimer_qpoint_input_files(
            qpoints, driver_params.input_dir, driver_params.prefix_coul_full,
            driver_params.prefix_sternheimer_chi0, driver::opts.nfreq, driver_params.use_rpa_gamma);
    }

    std::vector<QResult> qresults;
    qresults.reserve(qpoints.size());
    if (partial_mode)
    {
        const auto records = read_sternheimer_partial_manifest(partial_manifest_path);
        const auto groups = read_sternheimer_partial_response_groups(records);
        auto pds = librpa_int::api::get_dataset_instance(driver::h);
        librpa_int::initialize_symmetry_context(*pds, true);
        const auto &symmetry = pds->symmetry_context;
        const auto layouts = pds->basis_aux.build_species_basis_layouts(symmetry.atom_to_type);
        const auto atom_nabf = pds->basis_aux.get_atom_nb_map();
        const auto &full_kpoints = pds->pbc.kfrac_list_full;
        const int lmax = pds->basis_aux.get_max_l();
        const auto reconstructed = reconstruct_sternheimer_partial_responses(
            symmetry, layouts, atom_nabf, full_kpoints, qpoints, groups, driver::opts.nfreq,
            driver_params.use_rpa_gamma, lmax);

        for (const auto &point : qpoints)
        {
            if (!driver_params.use_rpa_gamma && is_rpa_gamma_point(point.q))
            {
                continue;
            }

            const librpa_int::Vector3_Order<double> q{point.q[0], point.q[1], point.q[2]};
            const auto coulomb = read_coulomb_v1_full_matrix(
                driver_params.input_dir, driver_params.prefix_coul_full, point.iq);
            const auto coulomb_qstar = librpa_int::reconstruct_sternheimer_qstar_responses(
                symmetry, layouts, atom_nabf, q, coulomb, lmax);

            QResult qresult;
            qresult.point = point;
            qresult.frequencies.reserve(static_cast<std::size_t>(driver::opts.nfreq));
            qresult.symmetry_audits.reserve(static_cast<std::size_t>(driver::opts.nfreq));
            for (const auto &response : reconstructed)
            {
                if (response.iq != point.iq)
                {
                    continue;
                }
                const auto audit = compute_sternheimer_qstar_rpa_frequency(
                    response, coulomb_qstar, driver::opts.sqrt_coulomb_threshold);
                if (write_reconstructed && mpi_comm_global_h.is_root())
                {
                    const auto &metadata = groups.at({point.iq, response.ifreq});
                    for (const auto &member : response.qstar_responses)
                    {
                        const auto folded = librpa_int::fold_fractional_kpoint_to_targets(
                            member.q, full_kpoints, 1.0e-8);
                        SternheimerChi0V1Matrix output;
                        output.iq = folded.target_k_index + 1;
                        output.ifreq = response.ifreq;
                        output.omega = response.omega;
                        output.weight = response.weight;
                        output.atom_naux = metadata.atom_naux;
                        output.matrix = member.matrix;
                        const std::string path =
                            driver_params.prefix_sternheimer_reconstructed
                            + std::to_string(output.iq) + "_ifreq_"
                            + std::to_string(output.ifreq) + ".dat";
                        write_sternheimer_chi0_v1_matrix_file(path, output);
                    }
                }
                double matrix_scale = 0.0;
                double hermiticity_residual = 0.0;
                std::complex<double> response_trace(0.0, 0.0);
                for (int row = 0; row != response.matrix.nr; ++row)
                {
                    response_trace += response.matrix(row, row);
                    for (int column = 0; column != response.matrix.nc; ++column)
                    {
                        matrix_scale =
                            std::max(matrix_scale, std::abs(response.matrix(row, column)));
                        hermiticity_residual =
                            std::max(hermiticity_residual,
                                     std::abs(response.matrix(row, column) -
                                              std::conj(response.matrix(column, row))));
                    }
                }
                qresult.frequencies.push_back(audit.result);
                qresult.symmetry_audits.push_back(
                    {response.ifreq, response.full_k_count, response.representative_k_count,
                     response.little_group_order, audit.qstar_size,
                     hermiticity_residual / std::max(1.0, matrix_scale),
                     audit.max_integrand_difference, response_trace});
            }
            if (qresult.frequencies.size() != static_cast<std::size_t>(driver::opts.nfreq))
            {
                throw std::runtime_error(
                    "Sternheimer partial task did not reconstruct every frequency for iq=" +
                    std::to_string(point.iq));
            }
            qresult.energy = librpa_int::sum_sternheimer_rpa_energies(qresult.frequencies);
            qresults.push_back(std::move(qresult));
        }
    }
    else
    {
        for (const auto &point : qpoints)
        {
            if (!driver_params.use_rpa_gamma && is_rpa_gamma_point(point.q))
            {
                continue;
            }
            auto coulomb = read_coulomb_v1_full_matrix(driver_params.input_dir,
                                                       driver_params.prefix_coul_full, point.iq);
            auto responses = read_sternheimer_chi0_v1_matrices(
                driver_params.input_dir, driver_params.prefix_sternheimer_chi0, point.iq);
            QResult qresult;
            qresult.point = point;
            qresult.frequencies.reserve(responses.size());
            for (const auto &response : responses)
            {
                qresult.frequencies.push_back(compute_sternheimer_rpa_frequency(
                    coulomb, response.matrix, response.ifreq, response.omega, response.weight,
                    point.weight, driver::opts.sqrt_coulomb_threshold));
            }
            qresult.energy = librpa_int::sum_sternheimer_rpa_energies(qresult.frequencies);
            qresults.push_back(std::move(qresult));
        }
    }
    std::vector<librpa_int::SternheimerRpaFrequencyResult> all_frequencies;
    for (const auto &qresult : qresults)
    {
        all_frequencies.insert(all_frequencies.end(), qresult.frequencies.begin(),
                               qresult.frequencies.end());
    }
    const auto total_energy = librpa_int::sum_sternheimer_rpa_energies(all_frequencies);

    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        lib_printf("Sternheimer RPA correlation energy (Hartree)\n");
        lib_printf("| q mode = %s\n", partial_mode
                                          ? "partial-symmetry"
                                          : (manifest_path.empty() ? "single" : "manifest"));
        if (!manifest_path.empty())
        {
            lib_printf("| q manifest = %s\n", manifest_path.c_str());
        }
        if (partial_mode)
        {
            lib_printf("| partial-response manifest = %s\n", partial_manifest_path.c_str());
            if (write_reconstructed)
            {
                lib_printf("| reconstructed Sternheimer prefix = %s\n",
                           driver_params.prefix_sternheimer_reconstructed.c_str());
            }
        }
        lib_printf("| Coulomb prefix = %s\n", driver_params.prefix_coul_full.c_str());
        if (!partial_mode)
        {
            lib_printf("| Sternheimer chi0 prefix = %s\n",
                       driver_params.prefix_sternheimer_chi0.c_str());
        }
        lib_printf("| use_rpa_gamma = %s\n", driver_params.use_rpa_gamma ? "true" : "false");
        if (!driver_params.use_rpa_gamma)
        {
            for (const auto &point : qpoints)
            {
                if (is_rpa_gamma_point(point.q))
                {
                    lib_printf("| excluded Gamma iq = %d, q weight = %.16e\n", point.iq,
                               point.weight);
                }
            }
        }
        for (const auto &qresult : qresults)
        {
            lib_printf("| iq = %d\n", qresult.point.iq);
            if (!manifest_path.empty())
            {
                lib_printf("| q = %.16e %.16e %.16e\n", qresult.point.q[0], qresult.point.q[1],
                           qresult.point.q[2]);
            }
            const double q_weight = qresult.frequencies.empty()
                                        ? qresult.point.weight
                                        : qresult.frequencies.front().qweight;
            lib_printf("| q weight = %.16e\n", q_weight);
            if (partial_mode)
            {
                lib_printf("| q manifest weight = %.16e\n", qresult.point.weight);
            }
            lib_printf(
                "| ifreq omega_Ha weight_Ha integrand_real integrand_imag Ec_real Ec_imag\n");
            for (const auto &result : qresult.frequencies)
            {
                lib_printf("| %5d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n", result.ifreq,
                           result.omega, result.weight, result.integrand.real(),
                           result.integrand.imag(), result.energy.real(), result.energy.imag());
            }
            for (const auto &audit : qresult.symmetry_audits)
            {
                lib_printf(
                    "| symmetry audit ifreq=%d full_k=%d representative_k=%d "
                    "little_group=%d qstar=%d hermiticity_relative_residual=%.12e "
                    "response_trace_real=%.12e response_trace_imag=%.12e "
                    "max_integrand_difference=%.12e\n",
                    audit.ifreq, audit.full_k_count, audit.representative_k_count,
                    audit.little_group_order, audit.qstar_size, audit.hermiticity_relative_residual,
                    audit.response_trace.real(), audit.response_trace.imag(),
                    audit.max_integrand_difference);
            }
            lib_printf("| q Sternheimer EcRPA: %20.12e %20.12e\n", qresult.energy.real(),
                       qresult.energy.imag());
        }
        if (!driver_params.use_rpa_gamma)
        {
            lib_printf("| Total Sternheimer EcRPA excluding q=0: %20.12e %20.12e\n",
                       total_energy.real(), total_energy.imag());
        }
        lib_printf("| Total Sternheimer EcRPA: %20.12e %20.12e\n", total_energy.real(),
                   total_energy.imag());
        if (std::abs(total_energy.imag()) > 1.0e-6)
        {
            lib_printf_root(LIBRPA_VERBOSE_WARN,
                            "Warning: sizable imaginary part in Sternheimer EcRPA = %.12e\n",
                            total_energy.imag());
        }
    }
    mpi_comm_global_h.barrier();
}
