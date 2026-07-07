#include "../../src/core/sternheimer_rpa.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../reader_sternheimer.h"
#include "../task.h"

void driver::task_sternheimer_rpa()
{
    using librpa_int::compute_sternheimer_rpa_frequency;
    using librpa_int::global::lib_printf;
    using librpa_int::global::lib_printf_root;
    using librpa_int::global::mpi_comm_global_h;

    const int iq = driver_params.sternheimer_iq;
    if (iq <= 0)
    {
        throw std::runtime_error("sternheimer_iq must be a positive one-based q index");
    }

    auto coulomb =
        read_coulomb_v1_full_matrix(driver_params.input_dir, driver_params.prefix_coul_full, iq);
    auto responses = read_sternheimer_chi0_v1_matrices(driver_params.input_dir,
                                                       driver_params.prefix_sternheimer_chi0, iq);
    if (responses.empty())
    {
        throw std::runtime_error("No Sternheimer chi0 response matrices were read");
    }
    if (static_cast<int>(responses.size()) != driver::opts.nfreq)
    {
        throw std::runtime_error("Sternheimer chi0 frequency count (" +
                                 std::to_string(responses.size()) + ") does not match nfreq (" +
                                 std::to_string(driver::opts.nfreq) + ")");
    }

    std::complex<double> total_energy(0.0, 0.0);
    std::vector<librpa_int::SternheimerRpaFrequencyResult> results;
    results.reserve(responses.size());
    for (const auto &response : responses)
    {
        const auto result = compute_sternheimer_rpa_frequency(
            coulomb, response.matrix, response.ifreq, response.omega, response.weight,
            driver_params.sternheimer_qweight, driver::opts.sqrt_coulomb_threshold);
        total_energy += result.energy;
        results.push_back(result);
    }

    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        lib_printf("Sternheimer RPA correlation energy (Hartree)\n");
        lib_printf("| iq = %d\n", iq);
        lib_printf("| q weight = %.16e\n", driver_params.sternheimer_qweight);
        lib_printf("| Coulomb prefix = %s\n", driver_params.prefix_coul_full.c_str());
        lib_printf("| Sternheimer chi0 prefix = %s\n",
                   driver_params.prefix_sternheimer_chi0.c_str());
        lib_printf("| ifreq omega_Ha weight_Ha integrand_real integrand_imag Ec_real Ec_imag\n");
        for (const auto &result : results)
        {
            lib_printf("| %5d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n", result.ifreq,
                       result.omega, result.weight, result.integrand.real(),
                       result.integrand.imag(), result.energy.real(), result.energy.imag());
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
