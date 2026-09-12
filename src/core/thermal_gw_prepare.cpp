#include "thermal_gw_prepare.h"

#include <cmath>
#include <set>
#include <stdexcept>

#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "epsilon.h"

namespace librpa_int
{
namespace
{
bool finite(cplxdb value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }

auto find_frequency(const ThermalWcQMap &wc, double target)
{
    auto found = wc.lower_bound(target);
    const double tolerance = 1e-12 * std::max(1.0, std::abs(target));
    if (found != wc.end() && std::abs(found->first - target) <= tolerance) return found;
    if (found != wc.begin())
    {
        --found;
        if (std::abs(found->first - target) <= tolerance) return found;
    }
    return wc.end();
}
}  // namespace

ThermalWcQMap prepare_thermal_wc(const MpiCommHandler &comm_h, const ThermalWcQMap &positive_wc,
                                 const PeriodicBoundaryData &pbc,
                                 const ThermalGWTransform &transform, const ArrayDesc &input_desc,
                                 const ArrayDesc &output_desc,
                                 const std::map<Vector3_Order<double>, ComplexMatrix> *sinvS,
                                 const SymmetryQPointView *qpoints, const SymmetryContext *symmetry,
                                 const AtomicBasis *full_basis)
{
    comm_h.check_initialized();
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    const bool restore_q =
        qpoints && qpoints->restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL;
    const auto &input_q = restore_q ? qpoints->representatives : pbc.klist_full;
    std::string error;
    try
    {
        if (!input_desc.initialized() || !output_desc.initialized())
            throw std::invalid_argument("thermal Wc requires initialized BLACS descriptors");
        int relation = MPI_UNEQUAL;
        MPI_Comm_compare(input_desc.comm(), comm_h.comm, &relation);
        if ((relation != MPI_IDENT && relation != MPI_CONGRUENT) ||
            input_desc.ictxt() != output_desc.ictxt() || input_desc.m() != input_desc.n() ||
            output_desc.m() != output_desc.n() || input_desc.mb() != input_desc.nb() ||
            output_desc.mb() != output_desc.nb() || input_desc.irsrc() != 0 ||
            input_desc.icsrc() != 0 || output_desc.irsrc() != 0 || output_desc.icsrc() != 0)
            throw std::invalid_argument("thermal Wc requires compatible square BLACS descriptors");
        if (!sinvS && (input_desc.m() != output_desc.m() || input_desc.mb() != output_desc.mb()))
            throw std::invalid_argument("thermal Wc basis change requires sinvS");
        if (pbc.klist_full.empty()) throw std::invalid_argument("thermal Wc has no full q grid");
        if (restore_q && (!symmetry || !symmetry->available || !full_basis ||
                          !full_basis->has_l_shells() || full_basis->nb_total != output_desc.m()))
            throw std::invalid_argument(
                "thermal Wc q symmetry requires the original ABF shell basis");
        for (double frequency : frequencies)
        {
            const auto row = find_frequency(positive_wc, std::abs(frequency));
            if (row == positive_wc.end())
                throw std::invalid_argument("thermal Wc missing required positive bosonic sample");
            for (const auto &q : input_q)
            {
                const auto block = row->second.find(q);
                if (block == row->second.end() || block->second.nr() != input_desc.m_loc() ||
                    block->second.nc() != input_desc.n_loc())
                    throw std::invalid_argument(
                        "thermal Wc missing full-q block or wrong local shape");
                const auto &w = block->second;
                if ((w.major() != MAJOR::ROW && w.major() != MAJOR::COL) || !w.sptr() ||
                    w.sptr()->size() < std::max<std::size_t>(1, w.size()))
                    throw std::invalid_argument("invalid thermal Wc block storage");
                for (std::size_t i = 0; i < w.size(); ++i)
                    if (!finite(w.ptr()[i])) throw std::invalid_argument("nonfinite thermal Wc");
                if (sinvS)
                {
                    const auto u = sinvS->find(q);
                    if (u == sinvS->end() || u->second.nr != input_desc.m() ||
                        u->second.nc != output_desc.m() || !u->second.c)
                        throw std::invalid_argument(
                            "thermal Wc missing sinvS or wrong unfold shape");
                    for (int i = 0; i < u->second.size; ++i)
                        if (!finite(u->second.c[i]))
                            throw std::invalid_argument("nonfinite thermal Wc unfold coefficient");
                }
            }
        }
    }
    catch (const std::exception &caught)
    {
        error = caught.what();
    }
    int failed = !error.empty();
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm_h.comm);
    if (failed)
        throw std::invalid_argument(error.empty() ? "invalid thermal Wc on another MPI rank"
                                                  : error);

    ArrayDesc u_desc(input_desc.ictxt());
    if (sinvS)
        u_desc.init(input_desc.m(), output_desc.m(), input_desc.mb(), output_desc.nb(), 0, 0);
    ThermalWcQMap result;
    std::set<double> positive;
    for (double frequency : frequencies) positive.insert(std::abs(frequency));
    for (double frequency : positive)
    {
        const auto &row = find_frequency(positive_wc, frequency)->second;
        std::map<Vector3_Order<double>, Matz> unfolded_row;
        for (const auto &q : input_q)
        {
            auto w = row.at(q).copy();
            if (w.is_row_major()) w.swap_to_col_major();
            Matz unfolded;
            if (sinvS)
            {
                auto u = init_local_mat<cplxdb>(u_desc, MAJOR::COL);
                const auto &global_u = sinvS->at(q);
                for (int i = 0; i < u_desc.m_loc(); ++i)
                    for (int j = 0; j < u_desc.n_loc(); ++j)
                        u(i, j) = global_u(u_desc.indx_l2g_r(i), u_desc.indx_l2g_c(j));
                auto wu = init_local_mat<cplxdb>(u_desc, MAJOR::COL);
                unfolded = init_local_mat<cplxdb>(output_desc, MAJOR::COL);
                ScalapackConnector::pgemm_f(
                    'N', 'N', input_desc.m(), output_desc.m(), input_desc.m(), 1.0, w.ptr(), 1, 1,
                    input_desc.desc, u.ptr(), 1, 1, u_desc.desc, 0.0, wu.ptr(), 1, 1, u_desc.desc);
                ScalapackConnector::pgemm_f('C', 'N', output_desc.m(), output_desc.m(),
                                            input_desc.m(), 1.0, u.ptr(), 1, 1, u_desc.desc,
                                            wu.ptr(), 1, 1, u_desc.desc, 0.0, unfolded.ptr(), 1, 1,
                                            output_desc.desc);
            }
            else
                unfolded = std::move(w);
            unfolded_row[q] = std::move(unfolded);
        }
        if (restore_q)
            unfolded_row = restore_symmetry_dense_wq_map(unfolded_row, pbc, *qpoints, *symmetry,
                                                         *full_basis, output_desc);
        for (const auto &q : pbc.klist_full)
        {
            auto block = unfolded_row.find(q);
            if (block == unfolded_row.end())
                block = std::find_if(unfolded_row.begin(), unfolded_row.end(),
                                     [&](const auto &item)
                                     {
                                         return same_fractional_kpoint(
                                             Vector3_Order<double>{pbc.latvec * item.first},
                                             Vector3_Order<double>{pbc.latvec * q}, 1e-5);
                                     });
            if (block == unfolded_row.end())
                throw std::invalid_argument("thermal Wc q symmetry did not cover the full grid");
            result[frequency][q] = std::move(block->second);
            if (frequency > 0)
            {
                auto adjoint = init_local_mat<cplxdb>(output_desc, MAJOR::COL);
                const auto &value = result.at(frequency).at(q);
                ScalapackConnector::pgeadd_f('C', output_desc.m(), output_desc.n(), 1.0,
                                             value.ptr(), 1, 1, output_desc.desc, 0.0,
                                             adjoint.ptr(), 1, 1, output_desc.desc);
                result[-frequency][q] = std::move(adjoint);
            }
        }
    }
    return result;
}
}  // namespace librpa_int
