#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/collective_stage.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../reader/localized_operator_io.h"
#include "../task.h"

namespace
{
using Complex = std::complex<double>;

std::vector<Complex> reference_at_k(const librpa_int::LocalizedMatrix& blocks,
                                    const std::vector<double>& fractional_k, const int ik,
                                    const int n)
{
    std::vector<Complex> result(static_cast<std::size_t>(n) * n);
    const double two_pi = 2.0 * std::acos(-1.0);
    for (const auto& [r, matrix] : blocks)
    {
        if (matrix.nr() != n || matrix.nc() != n)
            throw std::runtime_error("Hartree reference spatial AO dimension differs from input");
        const double angle = two_pi * (fractional_k[3 * ik] * r.x + fractional_k[3 * ik + 1] * r.y +
                                       fractional_k[3 * ik + 2] * r.z);
        const auto phase = std::polar(1.0, angle);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result[static_cast<std::size_t>(i) * n + j] += phase * matrix(i, j);
    }
    return result;
}
}  // namespace

void driver::task_hartree()
{
    using namespace librpa_int;
    using namespace librpa_int::global;
    const auto& params = driver_params;
    if (params.prefix_coul_hartree.empty())
        throw std::runtime_error("task=hartree requires explicit prefix_coul_hartree");
    int nspins, nkpoints, nstates, nao, nspinor;
    librpa_get_meanfield_dimensions(h.get_c_handler(), 0, &nspins, &nkpoints, &nstates, &nao,
                                    &nspinor);
    const auto matrix_size = static_cast<std::size_t>(nao) * nao;
    std::vector<int> selected;
    if (mpi_comm_global_h.is_root())
    {
        selected.resize(nkpoints);
        std::iota(selected.begin(), selected.end(), 0);
    }
    // All ranks participate; only root requests matrices for driver output.
    const auto values = h.compute_hartree_ao(opts, selected, nao);
    collective_all_stage(
        mpi_comm_global_h, "Hartree AO output and reference comparison",
        [&]
        {
            if (!mpi_comm_global_h.is_root()) return;
            if (nao <= 0 || values.size() != selected.size() * matrix_size)
                throw std::runtime_error("Hartree AO output size differs from the SCF basis/grid");
            std::ofstream output("hartree_ao.dat");
            if (!output) throw std::runtime_error("Cannot open hartree_ao.dat");
            output << std::scientific << std::setprecision(17)
                   << "LIBRPA_HARTREE_AO 1\nenergy_unit Hartree\nindex_base 0\n"
                   << "n_kpoints " << selected.size() << "\nn_aos " << nao
                   << "\n# Complete complex spatial AO matrices; k i j real imag\n";
            for (std::size_t k = 0; k < selected.size(); ++k)
                for (int i = 0; i < nao; ++i)
                    for (int j = 0; j < nao; ++j)
                    {
                        const auto value =
                            values[k * matrix_size + static_cast<std::size_t>(i) * nao + j];
                        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                            throw std::runtime_error(
                                "Hartree AO output contains non-finite entries");
                        output << selected[k] << ' ' << i << ' ' << j << ' ' << value.real() << ' '
                               << value.imag() << '\n';
                    }
            output.close();
            if (!output) throw std::runtime_error("Failed to write hartree_ao.dat");
            lib_printf(LIBRPA_VERBOSE_CRITICAL,
                       "Hartree AO matrices: %zu k points, %d spatial AOs, Hartree units\n",
                       selected.size(), nao);
            if (params.hartree_fn_reference.empty()) return;

            const auto path = is_absolute_path(params.hartree_fn_reference)
                                  ? params.hartree_fn_reference
                                  : join_path(params.input_dir, params.hartree_fn_reference);
            const auto blocks =
                librpa::reader::read_headwing_hamiltonian_csr(path, params.hartree_reference_scale);
            if (blocks.empty())
                throw std::runtime_error("Hartree reference has no real-space blocks");
            std::vector<double> fractional_k(static_cast<std::size_t>(nkpoints) * 3);
            librpa_get_crpa_kgrid(h.get_c_handler(), nkpoints, fractional_k.data());
            std::ofstream comparison("hartree_comparison.dat");
            if (!comparison) throw std::runtime_error("Cannot open hartree_comparison.dat");
            comparison
                << std::scientific << std::setprecision(17)
                << "# Complete complex AO comparison; no gauge fitting. All errors in Hartree.\n"
                << "# reference_scale_to_Ha " << params.hartree_reference_scale << '\n'
                << "# k max_abs_Ha frobenius_Ha\n";
            double maximum = 0.0;
            long double squared_norm = 0.0L;
            for (std::size_t k = 0; k < selected.size(); ++k)
            {
                const auto reference = reference_at_k(blocks, fractional_k, selected[k], nao);
                double local_maximum = 0.0;
                long double local_squared_norm = 0.0L;
                for (std::size_t i = 0; i < matrix_size; ++i)
                {
                    const auto difference = values[k * matrix_size + i] - reference[i];
                    if (!std::isfinite(difference.real()) || !std::isfinite(difference.imag()))
                        throw std::runtime_error(
                            "Hartree reference comparison contains non-finite entries");
                    local_maximum = std::max(local_maximum, std::abs(difference));
                    local_squared_norm += std::norm(difference);
                }
                maximum = std::max(maximum, local_maximum);
                squared_norm += local_squared_norm;
                comparison << selected[k] << ' ' << local_maximum << ' '
                           << std::sqrt(local_squared_norm) << '\n';
            }
            const double frobenius = std::sqrt(squared_norm);
            comparison << "max_abs_Ha " << maximum << "\nfrobenius_Ha " << frobenius
                       << "\ncheck_tolerance_Ha " << params.hartree_check_tolerance << '\n';
            comparison.close();
            if (!comparison) throw std::runtime_error("Failed to write hartree_comparison.dat");
            lib_printf(LIBRPA_VERBOSE_CRITICAL,
                       "Hartree comparison: max_abs_Ha = %.17e frobenius_Ha = %.17e\n", maximum,
                       frobenius);
            if (params.hartree_check_tolerance > 0.0)
            {
                if (maximum > params.hartree_check_tolerance)
                    throw std::runtime_error(
                        "Hartree complete AO reference comparison exceeds hartree_check_tolerance");
                lib_printf(LIBRPA_VERBOSE_CRITICAL,
                           "Hartree explicit reference comparison gate passed\n");
            }
            else
                lib_printf(
                    LIBRPA_VERBOSE_CRITICAL,
                    "Hartree reference comparison is diagnostic; no acceptance gate supplied\n");
        });
}
