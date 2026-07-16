#include "../../src/core/sternheimer_rpa.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../reader_sternheimer.h"
#include "../reader_sternheimer_qpoints.h"
#include "../task.h"

void driver::task_sternheimer_rpa()
{
    using librpa_int::compute_sternheimer_rpa_frequency;
    using librpa_int::global::lib_printf;
    using librpa_int::global::lib_printf_root;
    using librpa_int::global::mpi_comm_global_h;

    struct QResult
    {
        SternheimerQPoint point;
        std::vector<librpa_int::SternheimerRpaFrequencyResult> frequencies;
        std::complex<double> energy = {0.0, 0.0};
    };

    std::string manifest_path;
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
    validate_sternheimer_qpoint_input_files(
        qpoints, driver_params.input_dir, driver_params.prefix_coul_full,
        driver_params.prefix_sternheimer_chi0, driver::opts.nfreq);

    std::vector<QResult> qresults;
    qresults.reserve(qpoints.size());
    for (const auto &point : qpoints)
    {
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
        lib_printf("| q mode = %s\n", manifest_path.empty() ? "single" : "manifest");
        if (!manifest_path.empty())
        {
            lib_printf("| q manifest = %s\n", manifest_path.c_str());
        }
        lib_printf("| Coulomb prefix = %s\n", driver_params.prefix_coul_full.c_str());
        lib_printf("| Sternheimer chi0 prefix = %s\n",
                   driver_params.prefix_sternheimer_chi0.c_str());
        for (const auto &qresult : qresults)
        {
            lib_printf("| iq = %d\n", qresult.point.iq);
            if (!manifest_path.empty())
            {
                lib_printf("| q = %.16e %.16e %.16e\n", qresult.point.q[0], qresult.point.q[1],
                           qresult.point.q[2]);
            }
            lib_printf("| q weight = %.16e\n", qresult.point.weight);
            lib_printf(
                "| ifreq omega_Ha weight_Ha integrand_real integrand_imag Ec_real Ec_imag\n");
            for (const auto &result : qresult.frequencies)
            {
                lib_printf("| %5d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n", result.ifreq,
                           result.omega, result.weight, result.integrand.real(),
                           result.integrand.imag(), result.energy.real(), result.energy.imag());
            }
            lib_printf("| q Sternheimer EcRPA: %20.12e %20.12e\n", qresult.energy.real(),
                       qresult.energy.imag());
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
