#include <algorithm>
#include <stdexcept>

#include "../core/hartree.h"
#include "../mpi/collective_stage.h"
#include "hartree_input.h"
#include "instance_manager.h"
#include "librpa_compute.h"

void librpa_compute_hartree_ao(LibrpaHandler* handler, const LibrpaOptions* options,
                               const int n_kpoints, const int* kpoints, const int n_aos,
                               double* values_re, double* values_im)
{
    auto dataset = librpa_int::api::get_dataset_instance(handler);
    librpa_int::collective_all_stage(
        dataset->comm_h, "Hartree API input",
        [&]()
        {
            if (!options || n_kpoints < 0 ||
                (n_kpoints && (!kpoints || !values_re || !values_im)) || n_aos <= 0 ||
                n_aos != static_cast<int>(dataset->basis_wfc.nb_total))
                throw std::invalid_argument("Invalid Hartree AO output dimensions or buffers");
            if (options->use_2d_dielectric == LIBRPA_SWITCH_ON)
                throw std::invalid_argument(
                    "Periodic Hartree currently requires a bare three-dimensional Coulomb metric");
            if (dataset->basis_convention.bloch_ratom != 0)
                throw std::invalid_argument(
                    "Periodic Hartree requires the native RI translation convention and "
                    "atom-position phase 0");
            const bool has_gamma = std::any_of(dataset->pbc.klist.begin(), dataset->pbc.klist.end(),
                                               [](const auto& k) { return k.norm() < 1.0e-10; });
            if (!has_gamma)
                throw std::invalid_argument(
                    "The current Hartree Coulomb input API requires a Gamma-centred SCF grid");
            for (int i = 0; i < n_kpoints; ++i)
                if (kpoints[i] < 0 || kpoints[i] >= dataset->mf.get_n_kpoints())
                    throw std::invalid_argument(
                        "Hartree output k-point index is outside the loaded full grid");
        });
    if (options->use_kpara_scf_eigvec == LIBRPA_SWITCH_ON) dataset->redistribute_eigvecs_kpara_2d();
    const bool shrink = options->use_shrink_abfs == LIBRPA_SWITCH_ON;
    const auto density = librpa_int::build_hartree_density_k(*dataset);
    const auto metric = librpa_int::collect_hartree_coulomb_q0(*dataset, shrink);
    const auto& basis = shrink ? dataset->basis_aux_shrink : dataset->basis_aux;
    const auto& coefficients = shrink ? dataset->cs_data_shrink : dataset->cs_data;
    const auto result = librpa_int::compute_hartree_ao_k(
        dataset->basis_wfc, basis, coefficients, metric, dataset->pbc, density, dataset->comm_h);
    for (int k = 0; k < n_kpoints; ++k)
        for (int i = 0; i < n_aos; ++i)
            for (int j = 0; j < n_aos; ++j)
            {
                const auto value = result.at(kpoints[k])(i, j);
                const std::size_t index = (static_cast<std::size_t>(k) * n_aos + i) * n_aos + j;
                values_re[index] = value.real();
                values_im[index] = value.imag();
            }
}
