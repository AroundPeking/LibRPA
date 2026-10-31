#include "hartree_input.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../math/utils_matrix_mpi.h"
#include "../mpi/collective_stage.h"

namespace librpa_int
{
namespace
{
bool finite(const cplxdb& value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void allreduce_matrix_in_place(Matz& matrix, const MpiCommHandler& comm)
{
    const std::size_t chunk = 1u << 23;
    for (std::size_t begin = 0; begin < matrix.size(); begin += chunk)
        comm.allreduce(MPI_IN_PLACE, matrix.ptr() + begin,
                       static_cast<int>(std::min(chunk, matrix.size() - begin)), MPI_SUM);
}
}  // namespace

std::vector<Matz> build_hartree_density_k(Dataset& dataset)
{
    const auto& mf = dataset.mf;
    const auto& comm = dataset.comm_h;
    const int nk = mf.get_n_kpoints(), nao = mf.get_n_aos(), nb = mf.get_n_bands();
    const bool distributed = dataset.eigvecs_kpara_2d_ready();
    collective_all_stage(
        comm, "Hartree density preparation",
        [&]
        {
            if (!mf.initialized() || nao != static_cast<int>(dataset.basis_wfc.nb_total))
                throw std::invalid_argument(
                    "Hartree density requires a matching initialized mean field");
            if (distributed &&
                (!dataset.scfk_blacs_ctxt.is_initialized() ||
                 !dataset.desc_wfc_kb.is_initialized() ||
                 dataset.scfk_blacs_ctxt.n_kpoints() != nk ||
                 dataset.desc_wfc_kb.ictxt() != dataset.scfk_blacs_ctxt.blacs_h.ictxt ||
                 dataset.desc_wfc_kb.m() != nao || dataset.desc_wfc_kb.n() != nb))
                throw std::invalid_argument("Hartree density has an incompatible k-BLACS layout");
            for (int spin = 0; spin != mf.get_n_spins(); ++spin)
                for (int k = 0; k != nk; ++k)
                    for (int band = 0; band != nb; ++band)
                        if (!std::isfinite(mf.get_weight()[spin](k, band)) ||
                            mf.get_weight()[spin](k, band) < 0.0)
                            throw std::invalid_argument(
                                "Hartree occupations must be finite and nonnegative");
        });

    std::vector<Matz> result;
    result.reserve(nk);
    for (int k = 0; k != nk; ++k)
    {
        Matz density(nao, nao, MAJOR::COL);
        for (int spin = 0; spin != mf.get_n_spins(); ++spin)
            for (int spinor = 0; spinor != mf.get_n_spinor(); ++spinor)
            {
                const auto* local = mf.find_wfc(spin, spinor, k);
                bool owns_k = false;
                if (distributed)
                {
                    const auto& ks = dataset.scfk_blacs_ctxt.kpoints_local();
                    owns_k = std::find(ks.begin(), ks.end(), k) != ks.end();
                }
                collective_all_stage(
                    comm, "Hartree wavefunction validation",
                    [&]
                    {
                        if (distributed)
                        {
                            if (owns_k &&
                                dataset.desc_wfc_kb.m_loc() * dataset.desc_wfc_kb.n_loc() != 0 &&
                                (!local || local->nr != dataset.desc_wfc_kb.n_loc() ||
                                 local->nc != dataset.desc_wfc_kb.m_loc()))
                                throw std::invalid_argument(
                                    "Hartree density is missing a k-BLACS block");
                        }
                        else if (local && (local->nr != nb || local->nc != nao))
                            throw std::invalid_argument(
                                "Hartree density requires complete rows or prepared k-BLACS "
                                "blocks");
                        if (local)
                            for (int i = 0; i != local->size; ++i)
                                if (!finite(local->c[i]))
                                    throw std::invalid_argument(
                                        "Hartree wavefunction contains nonfinite data");
                    });

                Matz coefficients;
                if (distributed)
                {
                    if (owns_k)
                    {
                        Matz block(dataset.desc_wfc_kb.m_loc(), dataset.desc_wfc_kb.n_loc(),
                                   MAJOR::COL);
                        if (block.size()) std::copy(local->c, local->c + block.size(), block.ptr());
                        coefficients = collect_blacs_matrix_root(block, dataset.desc_wfc_kb);
                    }
                }
                else
                {
                    const int candidate = local ? comm.myid : comm.nprocs;
                    int owner = comm.nprocs;
                    comm.allreduce(&candidate, &owner, 1, MPI_MIN);
                    if (owner == comm.nprocs)
                        throw std::invalid_argument(
                            "Hartree density is missing a wavefunction channel");
                    // Replicated dense input contributes once, just like k-distributed input.
                    if (comm.myid == owner) coefficients = Matz(nao, nb, local->c, MAJOR::COL);
                }
                if (coefficients.size())
                {
                    // All stored occupations are retained, including small or noncontiguous ones.
                    Matz weighted = coefficients.copy();
                    for (int band = 0; band != nb; ++band)
                    {
                        const double weight = mf.get_weight()[spin](k, band);
                        for (int i = 0; i != nao; ++i) weighted(i, band) *= weight;
                    }
                    LapackConnector::gemm_f('N', 'C', nao, nao, nb, cplxdb(1.0), weighted.ptr(),
                                            nao, coefficients.ptr(), nao, cplxdb(1.0),
                                            density.ptr(), nao);
                }
            }
        allreduce_matrix_in_place(density, comm);
        result.push_back(std::move(density));
    }
    return result;
}

Matz collect_hartree_coulomb_q0(Dataset& dataset, const bool use_shrink)
{
    dataset.redistribute_coulomb_blacs2ap();
    const auto& basis = use_shrink ? dataset.basis_aux_shrink : dataset.basis_aux;
    const auto& comm = dataset.comm_h;
    const int nat = static_cast<int>(basis.n_atoms);
    const int naux = static_cast<int>(basis.nb_total);
    std::vector<int> owners(static_cast<std::size_t>(nat) * nat, 0);
    std::map<std::pair<int, int>, const ComplexMatrix*> blocks;
    collective_all_stage(
        comm, "Hartree q=0 metric input",
        [&]()
        {
            if (nat <= 0 || naux <= 0 ||
                static_cast<std::size_t>(naux) * naux > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "Hartree auxiliary basis is empty or exceeds dense MPI limits");
            for (const auto& row : dataset.vq)
                for (const auto& column : row.second)
                    for (const auto& qblock : column.second)
                    {
                        if (qblock.first.norm() > 1.0e-10) continue;
                        const int i = row.first, j = column.first;
                        if (i < 0 || i >= nat || j < 0 || j >= nat || !qblock.second ||
                            qblock.second->nr != static_cast<int>(basis[i]) ||
                            qblock.second->nc != static_cast<int>(basis[j]))
                            throw std::invalid_argument(
                                "Hartree q=0 block does not match the selected auxiliary basis");
                        if (!blocks.emplace(std::make_pair(i, j), qblock.second.get()).second)
                            throw std::invalid_argument("Repeated local Hartree q=0 metric block");
                        owners[i * nat + j] = 1;
                    }
        });
    comm.allreduce(MPI_IN_PLACE, owners.data(), static_cast<int>(owners.size()), MPI_SUM);
    if (std::none_of(owners.begin(), owners.end(), [](int n) { return n != 0; }))
        throw std::invalid_argument(
            "Hartree requires an explicit bare q=0 Coulomb metric with a specified zero mode");
    Matz metric(naux, naux, MAJOR::ROW);
    collective_all_stage(
        comm, "Hartree q=0 metric assembly",
        [&]()
        {
            for (const auto& entry : blocks)
            {
                const int i = entry.first.first, j = entry.first.second;
                const auto ii = basis.get_global_indices(i), jj = basis.get_global_indices(j);
                const double count = owners[i * nat + j];
                for (std::size_t a = 0; a < ii.size(); ++a)
                    for (std::size_t b = 0; b < jj.size(); ++b)
                    {
                        const auto value = (*entry.second)(a, b);
                        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                            throw std::invalid_argument("Non-finite Hartree q=0 Coulomb metric");
                        metric(ii[a], jj[b]) = value / count;
                    }
            }
        });
    comm.allreduce(MPI_IN_PLACE, metric.ptr(), static_cast<int>(metric.size()), MPI_SUM);
    // A complete atom-pair block may be replicated at an input boundary. Check
    // agreement before accepting the average; never sum replicated matrices.
    double mismatch = 0.0, scale = 1.0;
    for (const auto& entry : blocks)
    {
        const auto ii = basis.get_global_indices(entry.first.first);
        const auto jj = basis.get_global_indices(entry.first.second);
        for (std::size_t a = 0; a < ii.size(); ++a)
            for (std::size_t b = 0; b < jj.size(); ++b)
            {
                scale = std::max(scale, std::abs(metric(ii[a], jj[b])));
                mismatch =
                    std::max(mismatch, std::abs((*entry.second)(a, b) - metric(ii[a], jj[b])));
            }
    }
    comm.allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_MAX);
    comm.allreduce(MPI_IN_PLACE, &scale, 1, MPI_MAX);
    if (mismatch > 1.0e-11 * scale)
        throw std::invalid_argument("Replicated Hartree q=0 blocks disagree between MPI ranks");
    // Standard Coulomb input may contain only one triangle of atom pairs.
    for (int i = 0; i < nat; ++i)
        for (int j = 0; j < nat; ++j)
            if (!owners[i * nat + j] && owners[j * nat + i])
            {
                const auto ii = basis.get_global_indices(i), jj = basis.get_global_indices(j);
                for (const auto a : ii)
                    for (const auto b : jj) metric(a, b) = std::conj(metric(b, a));
            }
    double hermitian_error = 0.0;
    for (int a = 0; a < naux; ++a)
        for (int b = 0; b < naux; ++b)
            hermitian_error =
                std::max(hermitian_error, std::abs(metric(a, b) - std::conj(metric(b, a))));
    if (hermitian_error > 1.0e-10 * scale)
        throw std::invalid_argument("Hartree q=0 Coulomb metric is not Hermitian");
    return metric;
}
}  // namespace librpa_int
