#include "hartree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

#include "../math/utils_matrix_mpi.h"
#include "../mpi/collective_stage.h"

#ifdef LIBRPA_LIBRI_HAS_HARTREE
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/physics/Hartree.h>
#endif

namespace librpa_int
{
namespace
{
constexpr double two_pi = 6.283185307179586476925286766559;

bool finite(const cplxdb& z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

void check_matrix(const Matz& a, int n, const char* label)
{
    if (a.nr() != n || a.nc() != n)
        throw std::invalid_argument(std::string(label) + " has an incompatible basis size");
    double scale = 1.0, residual = 0.0;
    for (int i = 0; i != n; ++i)
        for (int j = 0; j != n; ++j)
        {
            if (!finite(a(i, j)))
                throw std::invalid_argument(std::string(label) + " contains nonfinite data");
            scale = std::max(scale, std::abs(a(i, j)));
            residual = std::max(residual, std::abs(a(i, j) - std::conj(a(j, i))));
        }
    if (residual > 1e-10 * scale)
        throw std::invalid_argument(std::string(label) + " is not Hermitian");
}

void allreduce_matrix_in_place(Matz& a, const MpiCommHandler& comm)
{
    // Keep MPI counts bounded independently of the global AO dimension.
    const std::size_t chunk = 1u << 23;
    for (std::size_t begin = 0; begin < a.size(); begin += chunk)
        comm.allreduce(MPI_IN_PLACE, a.ptr() + begin,
                       static_cast<int>(std::min(chunk, a.size() - begin)), MPI_SUM);
}
}  // namespace

std::map<int, Matz> compute_hartree_ao_k(
    const AtomicBasis& basis_wfc, const AtomicBasis& basis_aux, const Cs_LRI& cs,
    const Matz& coulomb_q0, const PeriodicBoundaryData& pbc,
    const std::vector<Matz>& weighted_density_k, const MpiCommHandler& comm_h,
    const std::vector<Vector3_Order<double>>* evaluation_kpoints,
    const AtomPairBvKRemap<atom_t>& bvk_remap)
{
#ifndef LIBRPA_LIBRI_HAS_HARTREE
    (void)basis_wfc;
    (void)basis_aux;
    (void)cs;
    (void)coulomb_q0;
    (void)pbc;
    (void)weighted_density_k;
    (void)comm_h;
    (void)evaluation_kpoints;
    (void)bvk_remap;
    throw std::runtime_error("Hartree requires LibRI with RI/physics/Hartree.h");
#else
    using Cell = std::array<int, 3>;
    using Tensor = RI::Tensor<cplxdb>;
    using TensorMap = std::map<int, std::map<std::pair<int, Cell>, Tensor>>;
    const int nat = static_cast<int>(basis_wfc.n_atoms);
    const int nao = static_cast<int>(basis_wfc.nb_total);
    const int nk = static_cast<int>(pbc.kfrac_list.size());
    std::vector<std::array<double, 3>> reduced_k;
    std::array<double, 3> shift{};
    TensorMap local_cs, ownership;
    std::set<int> atoms;
    collective_all_stage(
        comm_h, "Hartree input validation",
        [&]
        {
            if (nat <= 0 || basis_aux.n_atoms != basis_wfc.n_atoms || nao <= 0 ||
                basis_aux.nb_total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(
                    "Hartree requires matching positive AO/auxiliary atom bases");
            for (int atom = 0; atom != nat; ++atom)
            {
                if (!basis_wfc.get_atom_nb(atom) || !basis_aux.get_atom_nb(atom))
                    throw std::invalid_argument(
                        "Hartree requires nonempty AO and auxiliary bases on each atom");
                atoms.insert(atom);
            }
            std::size_t count = 1;
            for (int d = 0; d != 3; ++d)
            {
                if (pbc.period_array[d] <= 0)
                    throw std::invalid_argument("Hartree BvK periods must be positive");
                count *= pbc.period_array[d];
            }
            if (count != static_cast<std::size_t>(nk) || nk == 0 ||
                weighted_density_k.size() != static_cast<std::size_t>(nk))
                throw std::invalid_argument("Hartree requires the complete regular full-BZ mesh");
            shift = {pbc.kfrac_list[0].x, pbc.kfrac_list[0].y, pbc.kfrac_list[0].z};
            std::set<Cell> residues;
            for (int k = 0; k != nk; ++k)
            {
                const std::array<double, 3> original{pbc.kfrac_list[k].x, pbc.kfrac_list[k].y,
                                                     pbc.kfrac_list[k].z};
                Cell residue{};
                std::array<double, 3> reduced{};
                for (int d = 0; d != 3; ++d)
                {
                    const double index = (original[d] - shift[d]) * pbc.period_array[d];
                    if (!std::isfinite(index) || std::abs(index - std::round(index)) > 1e-8)
                        throw std::invalid_argument(
                            "Hartree k points do not form a regular BvK mesh");
                    const int index_int = static_cast<int>(std::round(index));
                    residue[d] = (index_int % pbc.period_array[d] + pbc.period_array[d]) %
                                 pbc.period_array[d];
                    reduced[d] = static_cast<double>(residue[d]) / pbc.period_array[d];
                }
                if (!residues.insert(residue).second)
                    throw std::invalid_argument(
                        "Hartree full-BZ mesh contains equivalent duplicate k points");
                reduced_k.push_back(reduced);
                check_matrix(weighted_density_k[k], nao, "Hartree weighted density");
            }
            check_matrix(coulomb_q0, static_cast<int>(basis_aux.nb_total), "Hartree q0 metric");
            if (evaluation_kpoints)
            {
                if (evaluation_kpoints->empty())
                    throw std::invalid_argument("Hartree evaluation k-point list is empty");
                for (const auto& k : *evaluation_kpoints)
                    if (!std::isfinite(k.x) || !std::isfinite(k.y) || !std::isfinite(k.z))
                        throw std::invalid_argument("Hartree evaluation k points must be finite");
            }
            // Real localized AO/ABF input has a real symmetric q=0 metric.
            for (int i = 0; i != coulomb_q0.nr(); ++i)
                for (int j = 0; j != coulomb_q0.nc(); ++j)
                    if (std::abs(coulomb_q0(i, j).imag()) >
                        1e-10 * std::max(1.0, std::abs(coulomb_q0(i, j))))
                        throw std::invalid_argument(
                            "Hartree q0 metric must be real in the supplied real auxiliary basis");
            for (const auto& i_entry : cs.data_libri)
            {
                const int i = i_entry.first;
                if (i < 0 || i >= nat)
                    throw std::invalid_argument("Hartree Cs has an invalid first atom");
                for (const auto& jr_entry : i_entry.second)
                {
                    const int j = jr_entry.first.first;
                    const auto& r = jr_entry.first.second;
                    const auto& block = jr_entry.second;
                    if (j < 0 || j >= nat || block.shape.size() != 3 ||
                        block.shape[0] != basis_aux.get_atom_nb(i) ||
                        block.shape[1] != basis_wfc.get_atom_nb(i) ||
                        block.shape[2] != basis_wfc.get_atom_nb(j) || !block.data ||
                        block.data->size() != block.shape[0] * block.shape[1] * block.shape[2])
                        throw std::invalid_argument(
                            "Hartree Cs does not match the supplied AO/auxiliary bases");
                    Tensor converted(block.shape);
                    const cplxdb phase = std::polar(
                        1.0, two_pi * (shift[0] * r[0] + shift[1] * r[1] + shift[2] * r[2]));
                    for (std::size_t n = 0; n != block.data->size(); ++n)
                    {
                        if (!std::isfinite((*block.data)[n]))
                            throw std::invalid_argument("Hartree Cs contains nonfinite data");
                        converted.ptr()[n] = phase * block.ptr()[n];
                    }
                    local_cs[i][jr_entry.first] = std::move(converted);
                    ownership[i][jr_entry.first] = Tensor({1});
                    ownership[i][jr_entry.first].ptr()[0] = 1.0;
                }
            }
        });
    // LibRI's map communication adds contributions: duplicate input ownership
    // would therefore change the physics.  Verify ownership before assembly.
    ownership =
        RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, ownership, atoms, atoms);
    collective_all_stage(
        comm_h, "Hartree Cs ownership",
        [&]
        {
            if (ownership.empty())
                throw std::invalid_argument("Hartree requires localized RI coefficients");
            for (const auto& row : ownership)
                for (const auto& entry : row.second)
                    if (entry.second.ptr()[0] != cplxdb(1.0))
                        throw std::invalid_argument(
                            "Hartree Cs blocks must have unique MPI ownership");
        });
    auto assembled_cs =
        RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, local_cs, atoms, atoms);
    std::map<int, Matz> output;
    for (int k = 0; k != nk; ++k) output.emplace(k, Matz(nao, nao, MAJOR::COL));

    // The upstream I/J/k schedule applies V before globally reducing charge.
    // Keep all atom pairs and split only k, so its reduction of V*M is exact.
    // Only nonempty k groups participate, avoiding upstream empty-charge .at().
    const int active_ranks = std::min(nk, comm_h.nprocs);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm_h.comm, comm_h.myid < active_ranks ? 0 : MPI_UNDEFINED, comm_h.myid,
                   &active_comm);
    std::unique_ptr<RI::Hartree<int, int, 3, cplxdb>> prepared;
    std::map<int, std::map<int, std::map<int, Tensor>>> density;
    try
    {
        collective_all_stage(
            comm_h, "LibRI Hartree preparation",
            [&]
            {
                if (active_comm == MPI_COMM_NULL) return;
                prepared.reset(new RI::Hartree<int, int, 3, cplxdb>);
                auto& hartree = *prepared;
                hartree.init(reduced_k);
                hartree.lrik.mpi_comm = active_comm;
                hartree.lrik.period = pbc.period_array;
                hartree.list_I.assign(atoms.begin(), atoms.end());
                hartree.list_J = hartree.list_I;
                hartree.list_IJ = hartree.list_I;
                for (int k = comm_h.myid; k < nk; k += active_ranks) hartree.k_indices.push_back(k);
                const Cell origin{0, 0, 0};
                TensorMap metric;
                for (int i = 0; i != nat; ++i)
                {
                    // Preserve an explicit zero onsite block even if a supplied
                    // atom has no surviving sparse Cs, as required by upstream.
                    if (assembled_cs[i].find({i, origin}) == assembled_cs[i].end())
                        assembled_cs[i][{i, origin}] =
                            Tensor({basis_aux.get_atom_nb(i), basis_wfc.get_atom_nb(i),
                                    basis_wfc.get_atom_nb(i)});
                    for (int j = 0; j != nat; ++j)
                    {
                        Tensor v({basis_aux.get_atom_nb(i), basis_aux.get_atom_nb(j)});
                        for (std::size_t a = 0; a != v.shape[0]; ++a)
                            for (std::size_t b = 0; b != v.shape[1]; ++b)
                                v(a, b) = coulomb_q0(basis_aux.get_global_index(i, a),
                                                     basis_aux.get_global_index(j, b))
                                              .real();
                        metric[i][{j, origin}] = std::move(v);
                    }
                }
                // At q=0 only sum_R V(R) matters; one R=0 block is exactly V(q=0).
                const std::map<std::string, double> flags{{"flag_comm", false},
                                                          {"flag_filter", false}};
                hartree.lrik.set_tensors_map2(assembled_cs, {RI::Label::ab::a, RI::Label::ab::b},
                                              flags, "Cs_");
                hartree.lrik.set_tensors_map2(metric, {RI::Label::ab::a0b0}, flags, "Vs_");
                for (int i = 0; i != nat; ++i)
                    for (int j = 0; j != nat; ++j)
                        for (const int k : hartree.k_indices)
                        {
                            Tensor block({basis_wfc.get_atom_nb(i), basis_wfc.get_atom_nb(j)});
                            for (std::size_t a = 0; a != block.shape[0]; ++a)
                                for (std::size_t b = 0; b != block.shape[1]; ++b)
                                    // The official kernel divides by Nk once;
                                    // physical stored wg already includes k weights.
                                    block(a, b) =
                                        static_cast<double>(nk) *
                                        weighted_density_k[k](basis_wfc.get_global_index(i, a),
                                                              basis_wfc.get_global_index(j, b));
                            density[i][j][k] = std::move(block);
                        }
            });
        collective_all_stage(
            comm_h, "LibRI Hartree contraction",
            [&]
            {
                if (!prepared) return;
                const auto blocks = prepared->cal_hartree(density);
                for (const auto& i : blocks)
                    for (const auto& j : i.second)
                        for (const auto& k : j.second)
                            for (std::size_t a = 0; a != k.second.shape[0]; ++a)
                                for (std::size_t b = 0; b != k.second.shape[1]; ++b)
                                    output.at(k.first)(basis_wfc.get_global_index(i.first, a),
                                                       basis_wfc.get_global_index(j.first, b)) =
                                        k.second(a, b);
            });
    }
    catch (...)
    {
        if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
        throw;
    }
    if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
    for (auto& k : output) allreduce_matrix_in_place(k.second, comm_h);
    if (evaluation_kpoints)
    {
        std::map<int, Matz> interpolated;
        for (std::size_t k = 0; k != evaluation_kpoints->size(); ++k)
            interpolated.emplace(static_cast<int>(k), Matz(nao, nao, MAJOR::COL));
        for (const auto& r : pbc.Rlist)
        {
            Matz hr(nao, nao, MAJOR::COL);
            for (int k = 0; k != nk; ++k)
            {
                const auto& q = reduced_k[k];
                const auto phase =
                    std::polar(1.0 / nk, -two_pi * (q[0] * r.x + q[1] * r.y + q[2] * r.z));
                for (int a = 0; a != nao; ++a)
                    for (int b = 0; b != nao; ++b) hr(a, b) += phase * output.at(k)(a, b);
            }
            for (int i = 0; i != nat; ++i)
                for (int j = 0; j != nat; ++j)
                {
                    const auto* images = bvk_remap.find_R_bvk({i, j}, r);
                    for (std::size_t k = 0; k != evaluation_kpoints->size(); ++k)
                    {
                        const auto& q = evaluation_kpoints->at(k);
                        const auto phase_at = [&](const Vector3_Order<int>& image)
                        {
                            return std::polar(1.0, two_pi * ((q.x - shift[0]) * image.x +
                                                             (q.y - shift[1]) * image.y +
                                                             (q.z - shift[2]) * image.z));
                        };
                        cplxdb phase{};
                        if (images && !images->empty())
                        {
                            for (const auto& image : *images) phase += phase_at(image);
                            phase /= static_cast<double>(images->size());
                        }
                        else
                            phase = phase_at(r);
                        for (std::size_t a = 0; a != basis_wfc.get_atom_nb(i); ++a)
                            for (std::size_t b = 0; b != basis_wfc.get_atom_nb(j); ++b)
                            {
                                const auto ia = basis_wfc.get_global_index(i, a);
                                const auto jb = basis_wfc.get_global_index(j, b);
                                interpolated.at(static_cast<int>(k))(ia, jb) += phase * hr(ia, jb);
                            }
                    }
                }
        }
        output = std::move(interpolated);
    }
    collective_all_stage(comm_h, "Hartree result validation",
                         [&]
                         {
                             for (const auto& k : output)
                                 check_matrix(k.second, nao, "Hartree AO operator");
                         });
    return output;
#endif
}

}  // namespace librpa_int
