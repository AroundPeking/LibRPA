#include "epsilon.h"
#define OPEN_TEST_FOR_LU_DECOMPOSITION
#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <valarray>

#include "abacus_symmetry.h"
#include "atoms.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "geometry.h"
#include "lapack_connector.h"
#include "libri_utils.h"
#include "matrix_m_parallel_utils.h"
#include "parallel_mpi.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "scalapack_connector.h"
#include "stl_io_helper.h"
#include "utils_blacs.h"
#include "utils_io.h"
#include "utils_mem.h"
#include "utils_mpi_io.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif

using LIBRPA::Array_Desc;
using LIBRPA::envs::blacs_ctxt_global_h;
using LIBRPA::envs::mpi_comm_global_h;
using LIBRPA::envs::ofs_myid;
using LIBRPA::utils::lib_printf;

namespace
{

using abf_qspace_complex_block_map_t =
    std::map<atom_t,
             std::map<atom_t, std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>>;
using abf_rspace_complex_block_map_t =
    atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old;
using abf_rspace_dense_block_map_t =
    std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, ComplexMatrix>>>;

bool are_equivalent_abacus_qpoints(const Vector3_Order<double>& lhs,
                                   const Vector3_Order<double>& rhs,
                                   const double tol = 1e-5)
{
    const auto same_component = [tol](const double lhs_component, const double rhs_component) {
        return std::abs((lhs_component - rhs_component) - std::round(lhs_component - rhs_component))
               < tol;
    };
    return same_component(lhs.x, rhs.x) && same_component(lhs.y, rhs.y)
           && same_component(lhs.z, rhs.z);
}

template <typename QMap>
typename QMap::const_iterator find_matching_abacus_qpoint(const QMap& q_map,
                                                          const Vector3_Order<double>& q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }

    return std::find_if(q_map.begin(), q_map.end(), [&q_target](const auto& entry) {
        return are_equivalent_abacus_qpoints(entry.first, q_target);
    });
}

std::vector<Vector3_Order<double>>::const_iterator find_matching_abacus_qpoint(
    const std::vector<Vector3_Order<double>>& q_points, const Vector3_Order<double>& q_target)
{
    const auto exact_iter = std::find(q_points.begin(), q_points.end(), q_target);
    if (exact_iter != q_points.end())
    {
        return exact_iter;
    }

    return std::find_if(q_points.begin(), q_points.end(), [&q_target](const auto& q_point) {
        return are_equivalent_abacus_qpoints(q_point, q_target);
    });
}

bool can_restore_abacus_abf_full_qspace_operator(const std::map<atom_t, size_t>& atom_nabf)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return Params::use_abacus_gw_symmetry && ctx.available && ctx.has_abf_shell_layout()
           && !ctx.kstars.empty() && ctx.kstars.size() == kfrac_list.size()
           && ctx.atom_to_type.size() == atom_nabf.size() && coord_frac.size() == atom_nabf.size();
}

ComplexMatrix to_complex_matrix(const matrix_m<std::complex<double>>& mat)
{
    ComplexMatrix complex_mat(mat.nr(), mat.nc());
    for (int row = 0; row < mat.nr(); ++row)
    {
        for (int col = 0; col < mat.nc(); ++col)
        {
            complex_mat(row, col) = mat(row, col);
        }
    }
    return complex_mat;
}

bool matrix_has_nonfinite_abf(const matrix_m<std::complex<double>>& matrix)
{
    const auto* data = matrix.ptr();
    for (int i = 0; i < static_cast<int>(matrix.size()); ++i)
    {
        const auto value = data[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            return true;
        }
    }
    return false;
}

double matrix_max_abs_abf(const matrix_m<std::complex<double>>& matrix)
{
    const auto* data = matrix.ptr();
    double max_abs = 0.0;
    for (int i = 0; i < static_cast<int>(matrix.size()); ++i)
    {
        const auto value = data[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            continue;
        }
        max_abs = std::max(max_abs, static_cast<double>(std::abs(value)));
    }
    return max_abs;
}

template <typename QSpaceMapT>
void log_abf_qspace_summary(const char* label, const double debug_tau, const QSpaceMapT& blocks_by_q)
{
    if (!Params::debug || !mpi_comm_global_h.is_root())
    {
        return;
    }

    std::size_t n_blocks = 0;
    std::size_t n_nonfinite_blocks = 0;
    double max_abs = 0.0;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            for (const auto& q_block : atom_j_pair.second)
            {
                ++n_blocks;
                const auto& matrix = q_block.second;
                if (matrix_has_nonfinite_abf(matrix))
                {
                    ++n_nonfinite_blocks;
                }
                max_abs = std::max(max_abs, matrix_max_abs_abf(matrix));
            }
        }
    }

    LIBRPA::utils::lib_printf_root(
        "GW debug %s tau = %12.6f, blocks = %zu, nonfinite_blocks = %zu, maxabs = %20.12e\n",
        label, debug_tau, n_blocks, n_nonfinite_blocks, max_abs);
}

bool complex_matrix_has_nonfinite(const ComplexMatrix& matrix)
{
    for (int i = 0; i < matrix.size; ++i)
    {
        const auto value = matrix.c[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            return true;
        }
    }
    return false;
}

double complex_matrix_max_abs_finite(const ComplexMatrix& matrix)
{
    double max_abs = 0.0;
    for (int i = 0; i < matrix.size; ++i)
    {
        const auto value = matrix.c[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            continue;
        }
        max_abs = std::max(max_abs, static_cast<double>(std::abs(value)));
    }
    return max_abs;
}

void log_abacus_atom_block_summary_once(const char* label,
                                        const int iq_ibz,
                                        const int star_index,
                                        const int member_index,
                                        const LIBRPA::abacus_atom_block_matrix_map_t& blocks)
{
    if (!Params::debug || !mpi_comm_global_h.is_root())
    {
        return;
    }

    std::size_t n_blocks = 0;
    std::size_t n_nonfinite_blocks = 0;
    double max_abs = 0.0;
    for (const auto& atom_i_pair : blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            ++n_blocks;
            if (complex_matrix_has_nonfinite(atom_j_pair.second))
            {
                ++n_nonfinite_blocks;
            }
            max_abs = std::max(max_abs, complex_matrix_max_abs_finite(atom_j_pair.second));
        }
    }

    static bool reported_finite = false;
    static bool reported_nonfinite = false;
    if (n_nonfinite_blocks == 0)
    {
        if (reported_finite)
        {
            return;
        }
        reported_finite = true;
    }
    else
    {
        if (reported_nonfinite)
        {
            return;
        }
        reported_nonfinite = true;
    }

    LIBRPA::utils::lib_printf_root(
        "ABACUS GW restore debug %s iq_ibz=%d star=%d member=%d blocks=%zu nonfinite_blocks=%zu maxabs=%20.12e\n",
        label, iq_ibz, star_index, member_index, n_blocks, n_nonfinite_blocks, max_abs);
}

ComplexMatrix allreduce_dense_complex_matrix_via_real_imag(const ComplexMatrix& local_matrix)
{
    constexpr int dense_sum_tag = 90123;
    const int packed_size = local_matrix.size * 2;
    std::vector<double> packed_local(static_cast<std::size_t>(packed_size), 0.0);
    for (int idx = 0; idx < local_matrix.size; ++idx)
    {
        packed_local[static_cast<std::size_t>(2 * idx)] = local_matrix.c[idx].real();
        packed_local[static_cast<std::size_t>(2 * idx + 1)] = local_matrix.c[idx].imag();
    }

    std::vector<double> packed_global(static_cast<std::size_t>(packed_size), 0.0);
    if (mpi_comm_global_h.myid == 0)
    {
        packed_global = packed_local;
        std::vector<double> packed_recv(static_cast<std::size_t>(packed_size), 0.0);
        for (int src = 1; src < mpi_comm_global_h.nprocs; ++src)
        {
            MPI_Status status;
            MPI_Recv(packed_recv.data(), packed_size, MPI_DOUBLE, src, dense_sum_tag,
                     mpi_comm_global_h.comm, &status);
            for (int idx = 0; idx < packed_size; ++idx)
            {
                packed_global[static_cast<std::size_t>(idx)] +=
                    packed_recv[static_cast<std::size_t>(idx)];
            }
        }
    }
    else
    {
        MPI_Send(
            packed_local.data(), packed_size, MPI_DOUBLE, 0, dense_sum_tag, mpi_comm_global_h.comm);
    }
    MPI_Bcast(packed_global.data(), packed_size, MPI_DOUBLE, 0, mpi_comm_global_h.comm);

    ComplexMatrix global_matrix(local_matrix.nr, local_matrix.nc);
    for (int idx = 0; idx < global_matrix.size; ++idx)
    {
        global_matrix.c[idx] =
            std::complex<double>(packed_global[static_cast<std::size_t>(2 * idx)],
                                 packed_global[static_cast<std::size_t>(2 * idx + 1)]);
    }
    return global_matrix;
}

matrix_m<std::complex<double>> to_row_major_matrix_m(const ComplexMatrix& mat)
{
    matrix_m<std::complex<double>> matrix_out(mat.nr, mat.nc, MAJOR::ROW);
    for (int row = 0; row < mat.nr; ++row)
    {
        for (int col = 0; col < mat.nc; ++col)
        {
            matrix_out(row, col) = mat(row, col);
        }
    }
    return matrix_out;
}

Vector3_Order<double> canonicalize_mp_fractional_qpoint(const Vector3_Order<double>& q_frac)
{
    return {q_frac.x - std::round(q_frac.x), q_frac.y - std::round(q_frac.y),
            q_frac.z - std::round(q_frac.z)};
}

Vector3_Order<double> resolve_ft_q_fractional(
    const Vector3_Order<double>& q_internal,
    const std::map<Vector3_Order<double>, Vector3_Order<double>>* qfrac_lookup = nullptr)
{
    if (qfrac_lookup != nullptr)
    {
        const auto exact_iter = qfrac_lookup->find(q_internal);
        if (exact_iter != qfrac_lookup->end())
        {
            return canonicalize_mp_fractional_qpoint(exact_iter->second);
        }
        const auto matched_iter =
            std::find_if(qfrac_lookup->begin(), qfrac_lookup->end(),
                         [&q_internal](const auto& entry) {
                             return are_equivalent_abacus_qpoints(entry.first, q_internal);
                         });
        if (matched_iter != qfrac_lookup->end())
        {
            return canonicalize_mp_fractional_qpoint(matched_iter->second);
        }
    }

    for (std::size_t ik = 0; ik < klist.size(); ++ik)
    {
        if (are_equivalent_abacus_qpoints(klist[ik], q_internal))
        {
            return canonicalize_mp_fractional_qpoint(kfrac_list[ik]);
        }
    }

    return canonicalize_mp_fractional_qpoint(latvec * q_internal);
}

std::complex<double> build_ft_wq_phase(const Vector3_Order<double>& q_internal,
                                       const Vector3_Order<int>& R,
                                       const int n_k_points,
                                       const std::map<Vector3_Order<double>, Vector3_Order<double>>*
                                           qfrac_lookup = nullptr)
{
    const auto q_frac = resolve_ft_q_fractional(q_internal, qfrac_lookup);
    const double ang = -(q_frac * R) * TWO_PI;
    return std::complex<double>(std::cos(ang), std::sin(ang)) / double(n_k_points);
}

void add_scaled_complex_matrix(ComplexMatrix& matrix_dst,
                               const ComplexMatrix& matrix_src,
                               const std::complex<double> scale)
{
    if (matrix_dst.nr != matrix_src.nr || matrix_dst.nc != matrix_src.nc)
    {
        throw std::runtime_error("Cannot accumulate ABACUS W(R) blocks with incompatible dimensions");
    }
    for (int row = 0; row < matrix_dst.nr; ++row)
    {
        for (int col = 0; col < matrix_dst.nc; ++col)
        {
            matrix_dst(row, col) += matrix_src(row, col) * scale;
        }
    }
}

void dump_blacs_debug_matrix(const std::string& file_name,
                             const matrix_m<std::complex<double>>& matrix_local,
                             const LIBRPA::Array_Desc& matrix_desc,
                             const double threshold = 1e-15)
{
    if (!Params::debug)
    {
        return;
    }

    print_matrix_mm_file_parallel(
        (Params::output_dir + "/" + file_name).c_str(), matrix_local, matrix_desc, threshold);
}

void dump_abacus_abf_qspace_blocks(
    const std::string& prefix,
    const abf_qspace_complex_block_map_t& blocks_by_q,
    const double threshold = 1e-15,
    const bool force_explicit_qcoords = false)
{
    if (!Params::debug)
    {
        return;
    }

    for (const auto& atom_i_pair : blocks_by_q)
    {
        const auto atom_i = atom_i_pair.first;
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const auto atom_j = atom_j_pair.first;
            for (const auto& q_iter : atom_j_pair.second)
            {
                std::ostringstream file_name;
                const auto q_index =
                    std::distance(klist.cbegin(), find_matching_abacus_qpoint(klist, q_iter.first));
                if (!force_explicit_qcoords
                    && q_index < static_cast<std::ptrdiff_t>(klist.size()))
                {
                    file_name << prefix << "_iq_" << q_index;
                }
                else
                {
                    // Keep dumping even when the restored q vector differs from the stored grid
                    // by small floating-point noise. The explicit q-components make the mapping
                    // back to the symmetry-off reference straightforward during debugging.
                    file_name.setf(std::ios::fixed);
                    file_name.precision(10);
                    file_name << prefix << "_qx_" << q_iter.first.x << "_qy_" << q_iter.first.y
                              << "_qz_" << q_iter.first.z;
                }
                file_name << "_I_" << atom_i << "_J_" << atom_j << "_id_"
                          << mpi_comm_global_h.myid << ".mtx";
                print_matrix_mm_file(q_iter.second, Params::output_dir + "/" + file_name.str(),
                                     threshold);
            }
        }
    }
}

LIBRPA::abacus_atom_block_matrix_map_t collect_abacus_abf_ibz_blocks_for_q(
    const atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old&
        blocks_by_q,
    const Vector3_Order<double>& q_ibz_key)
{
    LIBRPA::abacus_atom_block_matrix_map_t blocks_ibz;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        const auto atom_i = atom_i_pair.first;
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const auto atom_j = atom_j_pair.first;
            const auto q_iter = find_matching_abacus_qpoint(atom_j_pair.second, q_ibz_key);
            if (q_iter != atom_j_pair.second.end())
            {
                blocks_ibz[atom_i][atom_j] = to_complex_matrix(q_iter->second);
            }
        }
    }
    return blocks_ibz;
}

const LIBRPA::AbacusKStarMember& find_matching_abf_kstar_member(
    const LIBRPA::AbacusKStar& abf_star,
    const LIBRPA::AbacusKStarMember& ao_member)
{
    const auto matched = std::find_if(
        abf_star.members.begin(), abf_star.members.end(),
        [&ao_member](const LIBRPA::AbacusKStarMember& candidate) {
            return candidate.isym == ao_member.isym
                   && are_equivalent_abacus_qpoints(candidate.k_bz, ao_member.k_bz);
        });
    if (matched == abf_star.members.end())
    {
        throw std::runtime_error(
            "Failed to match an ABF k-star member with the AO-side symmetry member");
    }
    return *matched;
}

std::set<std::pair<atom_t, atom_t>> collect_abacus_atom_pairs(
    const LIBRPA::abacus_atom_block_matrix_map_t& atom_blocks);

std::set<std::pair<atom_t, atom_t>> collect_all_upper_atom_pairs(
    const std::map<atom_t, size_t>& atom_nabf)
{
    std::set<std::pair<atom_t, atom_t>> atom_pairs;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        for (std::size_t atom_j = atom_i; atom_j < atom_nabf.size(); ++atom_j)
        {
            atom_pairs.insert(
                {static_cast<atom_t>(atom_i), static_cast<atom_t>(atom_j)});
        }
    }
    return atom_pairs;
}

std::vector<int> build_abacus_atom_offsets(const std::map<atom_t, size_t>& atom_nabf)
{
    std::vector<int> offsets(atom_nabf.size() + 1, 0);
    for (std::size_t atom = 0; atom < atom_nabf.size(); ++atom)
    {
        offsets[atom + 1] = offsets[atom] + static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom)));
    }
    return offsets;
}

ComplexMatrix build_dense_abacus_hermitian_matrix_from_local_blocks(
    const LIBRPA::abacus_atom_block_matrix_map_t& local_blocks,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto offsets = build_abacus_atom_offsets(atom_nabf);
    ComplexMatrix dense(offsets.back(), offsets.back());
    for (const auto& atom_i_pair : local_blocks)
    {
        const int row_offset = offsets[static_cast<std::size_t>(atom_i_pair.first)];
        const int expected_nrows =
            static_cast<int>(atom_nabf.at(atom_i_pair.first));
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const int col_offset = offsets[static_cast<std::size_t>(atom_j_pair.first)];
            const int expected_ncols =
                static_cast<int>(atom_nabf.at(atom_j_pair.first));
            const auto& block = atom_j_pair.second;
            if (block.nr != expected_nrows || block.nc != expected_ncols)
            {
                std::ostringstream oss;
                oss << "ABACUS dense chi0 restore block dimension mismatch for atom pair ("
                    << atom_i_pair.first << ", " << atom_j_pair.first << "): block="
                    << block.nr << "x" << block.nc << ", expected=" << expected_nrows
                    << "x" << expected_ncols;
                throw std::runtime_error(oss.str());
            }
            if (row_offset + block.nr > dense.nr || col_offset + block.nc > dense.nc)
            {
                std::ostringstream oss;
                oss << "ABACUS dense chi0 restore block exceeds dense matrix bounds for atom pair ("
                    << atom_i_pair.first << ", " << atom_j_pair.first << ")";
                throw std::runtime_error(oss.str());
            }
            for (int row = 0; row < block.nr; ++row)
            {
                for (int col = 0; col < block.nc; ++col)
                {
                    const auto value = block(row, col);
                    dense(row_offset + row, col_offset + col) = value;
                    if (atom_i_pair.first != atom_j_pair.first)
                    {
                        dense(col_offset + col, row_offset + row) = std::conj(value);
                    }
                }
            }
        }
    }
    return dense;
}

LIBRPA::abacus_atom_block_matrix_map_t build_abacus_blocks_from_dense_matrix(
    const ComplexMatrix& dense_matrix,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto offsets = build_abacus_atom_offsets(atom_nabf);
    LIBRPA::abacus_atom_block_matrix_map_t atom_blocks;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        const int row_offset = offsets[atom_i];
        const int nrows = static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom_i)));
        for (std::size_t atom_j = atom_i; atom_j < atom_nabf.size(); ++atom_j)
        {
            const int col_offset = offsets[atom_j];
            const int ncols = static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom_j)));
            ComplexMatrix block(nrows, ncols);
            for (int row = 0; row < nrows; ++row)
            {
                for (int col = 0; col < ncols; ++col)
                {
                    block(row, col) = dense_matrix(row_offset + row, col_offset + col);
                }
            }
            atom_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] = std::move(block);
        }
    }
    return atom_blocks;
}

#ifdef LIBRPA_USE_LIBRI
LIBRPA::abacus_atom_block_matrix_map_t gather_abacus_ibz_blocks_for_local_target_pairs(
    const LIBRPA::AbacusKStar& star,
    const LIBRPA::abacus_atom_block_matrix_map_t& blocks_ibz_local,
    const std::set<std::pair<atom_t, atom_t>>& local_target_pairs,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::array<double, 3>& q_key_array);
#else
LIBRPA::abacus_atom_block_matrix_map_t gather_abacus_ibz_blocks_for_local_target_pairs(
    const LIBRPA::AbacusKStar&,
    const LIBRPA::abacus_atom_block_matrix_map_t& blocks_ibz_local,
    const std::set<std::pair<atom_t, atom_t>>&,
    const std::map<atom_t, size_t>&,
    const std::array<double, 3>&)
{
    return blocks_ibz_local;
}
#endif

LIBRPA::abacus_atom_block_matrix_map_t symmetrize_abacus_abf_ibz_blocks(
    const LIBRPA::AbacusSymmetryContext& ctx,
    const LIBRPA::AbacusKStar& star,
    const LIBRPA::AbacusKStar* abf_star,
    const Vector3_Order<double>& q_ibz_frac,
    const LIBRPA::abacus_atom_block_matrix_map_t& blocks_ibz,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs)
{
    (void)star;
    return LIBRPA::symmetrize_abacus_abf_ibz_kspace_operator_blocks(
        ctx, q_ibz_frac, blocks_ibz, atom_nabf, coord_frac, abf_star, &target_atom_pairs);
}

atom_mapping<ComplexMatrix>::pair_t_old symmetrize_abacus_chi0_ibz_blocks_if_needed(
    const atom_mapping<ComplexMatrix>::pair_t_old& blocks_ibz,
    const Vector3_Order<double>& q_ibz_internal)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    if (Params::debug)
    {
        ofs_myid << get_timestamp()
                 << " chi0 ibz restore helper entry for q=(" << q_ibz_internal.x << ", "
                 << q_ibz_internal.y << ", " << q_ibz_internal.z << ")"
                 << ", use_abacus_gw_symmetry=" << Params::use_abacus_gw_symmetry
                 << ", ctx_available=" << ctx.available
                 << ", has_abf_shell_layout=" << ctx.has_abf_shell_layout()
                 << ", atom_to_type_size=" << ctx.atom_to_type.size()
                 << ", klist_size=" << klist.size()
                 << ", kfrac_list_size=" << kfrac_list.size()
                 << ", local_input_block_count=" << collect_abacus_atom_pairs(blocks_ibz).size()
                 << endl;
    }
    if (!Params::use_abacus_gw_symmetry || !ctx.available || !ctx.has_abf_shell_layout()
        || ctx.atom_to_type.empty())
    {
        if (Params::debug)
        {
            ofs_myid << get_timestamp()
                     << " chi0 ibz restore helper bypass: symmetry context unavailable for q=("
                     << q_ibz_internal.x << ", " << q_ibz_internal.y << ", " << q_ibz_internal.z
                     << ")" << endl;
        }
        return blocks_ibz;
    }

    const auto q_iter = find_matching_abacus_qpoint(klist, q_ibz_internal);
    if (q_iter == klist.end())
    {
        if (Params::debug)
        {
            ofs_myid << get_timestamp()
                     << " chi0 ibz restore helper bypass: q not found in klist for q=("
                     << q_ibz_internal.x << ", " << q_ibz_internal.y << ", " << q_ibz_internal.z
                     << ")" << endl;
        }
        return blocks_ibz;
    }
    const auto iq_ibz = static_cast<std::size_t>(std::distance(klist.cbegin(), q_iter));
    if (iq_ibz >= kfrac_list.size())
    {
        if (Params::debug)
        {
            ofs_myid << get_timestamp()
                     << " chi0 ibz restore helper bypass: iq_ibz out of range for q=("
                     << q_ibz_internal.x << ", " << q_ibz_internal.y << ", " << q_ibz_internal.z
                     << "), iq_ibz=" << iq_ibz << ", kfrac_list_size=" << kfrac_list.size()
                     << endl;
        }
        return blocks_ibz;
    }

    const auto target_atom_pairs = collect_abacus_atom_pairs(blocks_ibz);
    auto output_atom_pairs = target_atom_pairs;

    std::map<atom_t, size_t> atom_nabf;
    const auto atom_nabf_vec = LIBRPA::atomic_basis_abf.get_atom_nbs();
    for (std::size_t atom = 0; atom < atom_nabf_vec.size(); ++atom)
    {
        atom_nabf[static_cast<atom_t>(atom)] = atom_nabf_vec[atom];
    }

    const auto& q_ibz_frac = kfrac_list[iq_ibz];
    const auto& star = LIBRPA::find_abacus_kstar_for_ibz_kpoint(ctx, q_ibz_frac);
    const LIBRPA::AbacusKStar* abf_star = nullptr;
    if (!ctx.abf_kstars.empty())
    {
        if (ctx.abf_kstars.size() != ctx.kstars.size())
        {
            throw std::runtime_error(
                "ABF k-space symmetry sidecar count is inconsistent with symrot_k.txt");
        }
        abf_star = &LIBRPA::find_abacus_kstar_for_kpoint(
            ctx.abf_kstars, q_ibz_frac, "ABF k-stars");
    }

    auto blocks_for_symmetrization = blocks_ibz;
#ifdef LIBRPA_USE_LIBRI
    const bool need_dense_restore_collective = mpi_comm_global_h.nprocs > 1;
    if (need_dense_restore_collective)
    {
        // For chi0, the sparse IBZ source blocks are naturally distributed by atom pair.
        // Reusing LibRI's sparse gather inside the symmetry restore path can deadlock on
        // this distribution. Instead, assemble the local Hermitian IBZ matrix densely and
        // sum it directly across all ranks. Every rank receives the same restored dense
        // matrix, which avoids the extra reduce+broadcast pair that proved fragile on the
        // multi-node AlAs symmetry benchmark.
        ComplexMatrix dense_ibz_global;
        {
            auto dense_ibz_local = build_dense_abacus_hermitian_matrix_from_local_blocks(
                blocks_ibz, atom_nabf);
            if (Params::debug)
            {
                ofs_myid << get_timestamp()
                         << " chi0 ibz dense restore start for q=(" << q_ibz_internal.x << ", "
                         << q_ibz_internal.y << ", " << q_ibz_internal.z << ")"
                         << ", local_block_count=" << target_atom_pairs.size()
                         << ", local_dims=" << dense_ibz_local.nr << "x" << dense_ibz_local.nc
                         << ", local_size=" << dense_ibz_local.size
                         << ", local_max_abs=" << dense_ibz_local.get_max_abs() << endl;
            }
            dense_ibz_global = allreduce_dense_complex_matrix_via_real_imag(dense_ibz_local);
            if (Params::debug)
            {
                ofs_myid << get_timestamp()
                         << " chi0 ibz dense restore finished for q=(" << q_ibz_internal.x
                         << ", " << q_ibz_internal.y << ", " << q_ibz_internal.z << ")" << endl;
            }
        }
        blocks_for_symmetrization =
            build_abacus_blocks_from_dense_matrix(dense_ibz_global, atom_nabf);
        // After the IBZ dense matrix is reconstructed on every rank, return the complete
        // upper-triangular atom-pair set so the subsequent Wc dense path can consume the same
        // symmetrized chi0(q, iω) locally without an additional MPI reduction.
        output_atom_pairs = collect_all_upper_atom_pairs(atom_nabf);
    }
#endif

    if (output_atom_pairs.empty())
    {
        return blocks_ibz;
    }

    return symmetrize_abacus_abf_ibz_blocks(
        ctx, star, abf_star, q_ibz_frac, blocks_for_symmetrization, atom_nabf, output_atom_pairs);
}

std::set<std::pair<atom_t, atom_t>> collect_abacus_atom_pairs(
    const LIBRPA::abacus_atom_block_matrix_map_t& atom_blocks)
{
    std::set<std::pair<atom_t, atom_t>> atom_pairs;
    for (const auto& atom_i_pair : atom_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            atom_pairs.insert({atom_i_pair.first, atom_j_pair.first});
        }
    }
    return atom_pairs;
}

std::vector<int> build_abacus_target_to_source_atom_map(const LIBRPA::AbacusKStarMember& member,
                                                        const std::size_t natoms)
{
    std::vector<int> target_to_source(natoms, -1);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from >= 0
            && atom_rotation.atom_from < static_cast<int>(natoms))
        {
            target_to_source[static_cast<std::size_t>(atom_rotation.atom_from)] =
                atom_rotation.atom_to;
        }
    }
    return target_to_source;
}

#ifdef LIBRPA_USE_LIBRI
using abacus_ibz_tensor_map_t =
    std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<std::complex<double>>>>;

std::pair<std::set<int>, std::set<int>> collect_abacus_required_source_atom_sets(
    const LIBRPA::AbacusKStar& star,
    const std::set<std::pair<atom_t, atom_t>>& local_target_pairs,
    const std::size_t natoms)
{
    std::pair<std::set<int>, std::set<int>> source_atom_sets;
    for (const auto& member : star.members)
    {
        const auto target_to_source = build_abacus_target_to_source_atom_map(member, natoms);
        for (const auto& atom_pair : local_target_pairs)
        {
            const int source_i = target_to_source.at(static_cast<std::size_t>(atom_pair.first));
            const int source_j = target_to_source.at(static_cast<std::size_t>(atom_pair.second));
            if (source_i < 0 || source_j < 0)
            {
                throw std::runtime_error(
                    "ABACUS GW restore found an incomplete target-to-source atom map");
            }
            source_atom_sets.first.insert(source_i);
            source_atom_sets.second.insert(source_j);
        }
    }
    return source_atom_sets;
}

abacus_ibz_tensor_map_t convert_abacus_blocks_to_tensor_map(
    const LIBRPA::abacus_atom_block_matrix_map_t& atom_blocks,
    const std::array<double, 3>& q_key_array)
{
    abacus_ibz_tensor_map_t tensor_map;
    for (const auto& atom_i_pair : atom_blocks)
    {
        const auto atom_i = static_cast<int>(atom_i_pair.first);
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const auto atom_j = static_cast<int>(atom_j_pair.first);
            auto tensor_storage =
                std::make_shared<std::valarray<std::complex<double>>>(
                    atom_j_pair.second.c, atom_j_pair.second.size);
            tensor_map[atom_i][{atom_j, q_key_array}] =
                RI::Tensor<std::complex<double>>(
                    {static_cast<std::size_t>(atom_j_pair.second.nr),
                     static_cast<std::size_t>(atom_j_pair.second.nc)},
                    tensor_storage);
        }
    }
    return tensor_map;
}

LIBRPA::abacus_atom_block_matrix_map_t convert_tensor_map_to_abacus_blocks(
    const abacus_ibz_tensor_map_t& tensor_map,
    const std::array<double, 3>& q_key_array,
    const std::map<atom_t, size_t>& atom_nabf)
{
    LIBRPA::abacus_atom_block_matrix_map_t atom_blocks;
    for (const auto& atom_i_pair : tensor_map)
    {
        const auto atom_i = static_cast<atom_t>(atom_i_pair.first);
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            if (atom_j_pair.first.second != q_key_array)
            {
                continue;
            }
            const auto atom_j = static_cast<atom_t>(atom_j_pair.first.first);
            const int nrows = static_cast<int>(atom_nabf.at(atom_i));
            const int ncols = static_cast<int>(atom_nabf.at(atom_j));
            ComplexMatrix block(nrows, ncols);
            for (int row = 0; row < nrows; ++row)
            {
                for (int col = 0; col < ncols; ++col)
                {
                    block(row, col) = atom_j_pair.second(row, col);
                }
            }
            atom_blocks[atom_i][atom_j] = std::move(block);
        }
    }
    return atom_blocks;
}

LIBRPA::abacus_atom_block_matrix_map_t gather_abacus_ibz_blocks_for_local_target_pairs(
    const LIBRPA::AbacusKStar& star,
    const LIBRPA::abacus_atom_block_matrix_map_t& blocks_ibz_local,
    const std::set<std::pair<atom_t, atom_t>>& local_target_pairs,
    const std::map<atom_t, size_t>& atom_nabf,
    const std::array<double, 3>& q_key_array)
{
    if (mpi_comm_global_h.nprocs <= 1 || local_target_pairs.empty())
    {
        return blocks_ibz_local;
    }

    const auto source_atom_sets = collect_abacus_required_source_atom_sets(
        star, local_target_pairs, atom_nabf.size());
    const auto gathered_tensor_map = comm_map2_first(
        mpi_comm_global_h.comm,
        convert_abacus_blocks_to_tensor_map(blocks_ibz_local, q_key_array),
        source_atom_sets.first,
        source_atom_sets.second);
    return convert_tensor_map_to_abacus_blocks(gathered_tensor_map, q_key_array, atom_nabf);
}
#endif

abf_qspace_complex_block_map_t restore_abacus_abf_full_qspace_operator(
    const atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old&
        blocks_by_q_ibz,
    const std::map<atom_t, size_t>& atom_nabf)
{
    abf_qspace_complex_block_map_t blocks_by_q_full;
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const auto kstar_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(ctx, klist, kfrac_list, map_irk_ks);

    for (const auto& mapping_entry : kstar_mapping)
    {
        const auto& star = ctx.kstars[static_cast<std::size_t>(mapping_entry.star_list_index)];
        const LIBRPA::AbacusKStar* abf_star = nullptr;
        if (!ctx.abf_kstars.empty())
        {
            if (ctx.abf_kstars.size() != ctx.kstars.size())
            {
                throw std::runtime_error(
                    "ABF k-space symmetry sidecar count is inconsistent with symrot_k.txt");
            }
            abf_star = &ctx.abf_kstars.at(static_cast<std::size_t>(mapping_entry.star_list_index));
        }
        const auto& q_ibz_key = klist[static_cast<std::size_t>(mapping_entry.iq_ibz)];
        const auto& k_ibz_frac = kfrac_list[static_cast<std::size_t>(mapping_entry.iq_ibz)];
        const auto blocks_ibz_local =
            collect_abacus_abf_ibz_blocks_for_q(blocks_by_q_ibz, q_ibz_key);
        if (Params::debug)
        {
            std::fprintf(stderr,
                         "[rank %d] restore_abf_full_qspace start iq_ibz=%d star=%d q=(%.10f, %.10f, %.10f) local_blocks=%zu members=%zu\n",
                         mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index,
                         q_ibz_key.x, q_ibz_key.y, q_ibz_key.z, blocks_ibz_local.size(),
                         star.members.size());
            std::fflush(stderr);
        }
        if (Params::debug)
        {
            LIBRPA::utils::lib_printf_root(
                "ABACUS GW restore debug: iq_ibz=%d star=%d k_ibz=(%.7f %.7f %.7f) blocks=%zu members=%zu\n",
                mapping_entry.iq_ibz, star.star_index, k_ibz_frac.x, k_ibz_frac.y, k_ibz_frac.z,
                blocks_ibz_local.size(), star.members.size());
        }

        const auto local_target_pairs = collect_abacus_atom_pairs(blocks_ibz_local);
        auto target_atom_pairs = local_target_pairs;
        auto blocks_ibz = blocks_ibz_local;
#ifdef LIBRPA_USE_LIBRI
        if (mpi_comm_global_h.nprocs > 1)
        {
            // `Wc(q_ibz, iω)` is distributed by atom pair. Reusing LibRI's sparse
            // `comm_map2_first()` here can stall when some ranks hold no local target pairs for a
            // given `(q_ibz, iω)` block. Restore the Hermitian IBZ matrix densely on every rank
            // instead, exactly as in the stabilized chi0 symmetry path.
            const auto dense_ibz_local =
                build_dense_abacus_hermitian_matrix_from_local_blocks(blocks_ibz_local, atom_nabf);
            const auto dense_ibz_global =
                allreduce_dense_complex_matrix_via_real_imag(dense_ibz_local);
            blocks_ibz = build_abacus_blocks_from_dense_matrix(dense_ibz_global, atom_nabf);
            // After the dense IBZ operator is reconstructed on every rank, drive the
            // q-star restore with the complete upper-triangular atom-pair list so that
            // all ranks participate in the same collective rotation path.
            target_atom_pairs = collect_all_upper_atom_pairs(atom_nabf);
            if (Params::debug)
            {
                std::fprintf(stderr,
                             "[rank %d] restore_abf_full_qspace after dense allreduce iq_ibz=%d star=%d local_target_pairs=%zu target_pairs=%zu\n",
                             mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index,
                             local_target_pairs.size(), target_atom_pairs.size());
                std::fflush(stderr);
            }
        }
#endif
        if (target_atom_pairs.empty())
        {
            if (Params::debug)
            {
                std::fprintf(stderr,
                             "[rank %d] restore_abf_full_qspace skip iq_ibz=%d star=%d because target_atom_pairs is empty\n",
                             mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index);
                std::fflush(stderr);
            }
            continue;
        }

        // Unlike the bare Coulomb exported by ABACUS, `W(q_ibz)` is built numerically inside
        // LibRPA. We therefore enforce the little-group average on the IBZ representative before
        // expanding it to the full star, so the result does not depend on the specific member
        // chosen as the representative.
        blocks_ibz = symmetrize_abacus_abf_ibz_blocks(
            ctx, star, abf_star, k_ibz_frac, blocks_ibz, atom_nabf, target_atom_pairs);
        log_abacus_atom_block_summary_once(
            "symmetrized_ibz", mapping_entry.iq_ibz, star.star_index, -1, blocks_ibz);
        if (Params::debug)
        {
            std::fprintf(stderr,
                         "[rank %d] restore_abf_full_qspace after ibz symmetrize iq_ibz=%d star=%d target_pairs=%zu\n",
                         mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index,
                         target_atom_pairs.size());
            std::fflush(stderr);
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& abf_member =
                (abf_star == nullptr) ? member : find_matching_abf_kstar_member(*abf_star, member);
            if (Params::debug)
            {
                LIBRPA::utils::lib_printf_root(
                    "ABACUS GW restore debug:   member isym=%d k_bz=(%.7f %.7f %.7f)\n",
                    member.isym, member.k_bz.x, member.k_bz.y, member.k_bz.z);
                std::fprintf(stderr,
                             "[rank %d] restore_abf_full_qspace before member rotate iq_ibz=%d star=%d imember=%zu isym=%d\n",
                             mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index, imember,
                             member.isym);
                std::fflush(stderr);
            }
            const bool use_time_reversal = member.isym >= nsym_space;
            const auto q_bz_target_frac_vec =
                latvec * mapping_entry.member_q_bz_keys[static_cast<std::size_t>(imember)];
            const Vector3_Order<double> q_bz_target_frac{
                q_bz_target_frac_vec.x, q_bz_target_frac_vec.y, q_bz_target_frac_vec.z};
            LIBRPA::abacus_atom_block_matrix_map_t rotated_blocks;
            try
            {
                rotated_blocks = LIBRPA::rotate_abacus_abf_kspace_operator_blocks(
                    ctx, abf_member, blocks_ibz, atom_nabf, k_ibz_frac, coord_frac,
                    use_time_reversal, &target_atom_pairs, &q_bz_target_frac);
            }
            catch (const std::exception& ex)
            {
                std::ostringstream oss;
                oss << "ABACUS Wc q-star restore failed for iq_ibz=" << mapping_entry.iq_ibz
                    << ", star=" << star.star_index << ", member=" << imember
                    << ", isym=" << member.isym << ": " << ex.what();
                throw std::runtime_error(oss.str());
            }
            if (Params::debug)
            {
                std::fprintf(stderr,
                             "[rank %d] restore_abf_full_qspace after member rotate iq_ibz=%d star=%d imember=%zu rotated_blocks=%zu\n",
                             mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index, imember,
                             rotated_blocks.size());
                std::fflush(stderr);
            }
            log_abacus_atom_block_summary_once(
                "rotated_member", mapping_entry.iq_ibz, star.star_index,
                static_cast<int>(imember), rotated_blocks);
            for (const auto& atom_i_pair : rotated_blocks)
            {
                for (const auto& atom_j_pair : atom_i_pair.second)
                {
                    const auto& q_bz_key =
                        mapping_entry.member_q_bz_keys[static_cast<std::size_t>(imember)];
                    blocks_by_q_full[atom_i_pair.first][atom_j_pair.first][q_bz_key] =
                        to_row_major_matrix_m(atom_j_pair.second);
                }
            }
        }
        if (Params::debug)
        {
            std::fprintf(stderr,
                         "[rank %d] restore_abf_full_qspace finish iq_ibz=%d star=%d\n",
                         mpi_comm_global_h.myid, mapping_entry.iq_ibz, star.star_index);
            std::fflush(stderr);
        }
    }

    return blocks_by_q_full;
}

LIBRPA::abacus_irreducible_sector_t filter_abacus_irreducible_sector_by_rlist(
    const LIBRPA::abacus_irreducible_sector_t& irreducible_sector,
    const std::vector<Vector3_Order<int>>& Rlist)
{
    LIBRPA::abacus_irreducible_sector_t filtered_sector;
    const std::set<Vector3_Order<int>> requested_rset(Rlist.begin(), Rlist.end());
    for (const auto& pair_Rs : irreducible_sector)
    {
        for (const auto& R_array : pair_Rs.second)
        {
            const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
            if (requested_rset.count(R) == 0)
            {
                continue;
            }
            filtered_sector[pair_Rs.first].insert(R_array);
        }
    }
    return filtered_sector;
}

std::set<std::pair<atom_t, atom_t>> build_abacus_irreducible_target_atom_pairs(
    const LIBRPA::abacus_irreducible_sector_t& irreducible_sector)
{
    std::set<std::pair<atom_t, atom_t>> target_atom_pairs;
    for (const auto& pair_Rs : irreducible_sector)
    {
        if (!pair_Rs.second.empty())
        {
            target_atom_pairs.insert(pair_Rs.first);
        }
    }
    return target_atom_pairs;
}

std::set<std::pair<atom_t, atom_t>> collect_local_target_atom_pairs_from_qspace(
    const abf_qspace_complex_block_map_t& blocks_by_q)
{
    std::set<std::pair<atom_t, atom_t>> target_atom_pairs;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            if (!atom_j_pair.second.empty())
            {
                target_atom_pairs.insert({atom_i_pair.first, atom_j_pair.first});
            }
        }
    }
    return target_atom_pairs;
}

struct AbacusIrreducibleWRPlan
{
    bool available = false;
    LIBRPA::abacus_irreducible_sector_t local_irreducible_sector;
    LIBRPA::abacus_rspace_sector_stars_t local_sector_stars;
    std::set<std::pair<atom_t, atom_t>> local_irreducible_pairs;
    std::vector<LIBRPA::AbacusKStarGridMappingEntry> kstar_grid_mapping;
    int nsym_space = 0;
};

AbacusIrreducibleWRPlan build_abacus_irreducible_wr_plan(
    const std::set<std::pair<atom_t, atom_t>>& local_target_pairs,
    const std::vector<Vector3_Order<int>>& Rlist)
{
    AbacusIrreducibleWRPlan plan;
    if (local_target_pairs.empty())
    {
        return plan;
    }

    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const auto filtered_sector = filter_abacus_irreducible_sector_by_rlist(ctx.irreducible_sector, Rlist);
    if (filtered_sector.empty())
    {
        return plan;
    }

    LIBRPA::abacus_rspace_sector_stars_t sector_stars;
    const Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    LIBRPA::build_abacus_rspace_sector_stars(ctx, coord_frac, period, Rlist, sector_stars, nullptr);

    for (const auto& pair_star : sector_stars)
    {
        const auto& ir_pair = pair_star.first;
        for (const auto& R_members : pair_star.second)
        {
            std::vector<LIBRPA::AbacusRSpaceRestoreMember> local_members;
            for (const auto& restore_member : R_members.second)
            {
                if (local_target_pairs.count(restore_member.full_atom_pair) != 0)
                {
                    local_members.push_back(restore_member);
                }
            }
            if (local_members.empty())
            {
                continue;
            }

            plan.local_sector_stars[ir_pair][R_members.first] = std::move(local_members);
            plan.local_irreducible_sector[ir_pair].insert(
                {R_members.first.x, R_members.first.y, R_members.first.z});
        }
    }

    if (plan.local_irreducible_sector.empty())
    {
        return plan;
    }

    plan.local_irreducible_pairs =
        build_abacus_irreducible_target_atom_pairs(plan.local_irreducible_sector);
    plan.kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(LIBRPA::abacus_symmetry_ctx, klist, kfrac_list, map_irk_ks);
    plan.nsym_space = static_cast<int>(ctx.rspace_operations.size());
    plan.available = true;
    return plan;
}

abf_rspace_dense_block_map_t allocate_abacus_irreducible_wr_storage(
    const LIBRPA::abacus_irreducible_sector_t& irreducible_sector,
    const std::map<atom_t, size_t>& atom_nabf)
{
    abf_rspace_dense_block_map_t blocks_by_R_dense;
    for (const auto& pair_Rs : irreducible_sector)
    {
        const auto atom_i = pair_Rs.first.first;
        const auto atom_j = pair_Rs.first.second;
        const int n_i = static_cast<int>(atom_nabf.at(atom_i));
        const int n_j = static_cast<int>(atom_nabf.at(atom_j));
        for (const auto& R_array : pair_Rs.second)
        {
            const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
            blocks_by_R_dense[atom_i][atom_j][R] = ComplexMatrix(n_i, n_j);
        }
    }
    return blocks_by_R_dense;
}

abf_rspace_complex_block_map_t convert_dense_rspace_blocks_to_row_major(
    const abf_rspace_dense_block_map_t& dense_blocks)
{
    abf_rspace_complex_block_map_t row_major_blocks;
    for (const auto& atom_i_pair : dense_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            for (const auto& R_block : atom_j_pair.second)
            {
                row_major_blocks[atom_i_pair.first][atom_j_pair.first][R_block.first] =
                    to_row_major_matrix_m(R_block.second);
            }
        }
    }
    return row_major_blocks;
}

abf_rspace_dense_block_map_t restore_abacus_abf_rspace_dense_blocks(
    const abf_rspace_dense_block_map_t& tensors_ir,
    const LIBRPA::AbacusSymmetryContext& symmetry_ctx,
    const LIBRPA::abacus_rspace_sector_stars_t& sector_stars)
{
    abf_rspace_dense_block_map_t tensors_full;
    for (const auto& i_entry : tensors_ir)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& jr_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(jr_entry.first);
            const auto pair_iter = sector_stars.find({ir_I, ir_J});
            if (pair_iter == sector_stars.end())
            {
                throw std::runtime_error(
                    "Failed to match an irreducible W(R) atom pair with the ABACUS restore map");
            }
            for (const auto& R_matrix : jr_entry.second)
            {
                const auto& ir_R = R_matrix.first;
                if (pair_iter->second.count(ir_R) == 0)
                {
                    std::ostringstream oss;
                    oss << "Failed to match an irreducible W(R) block with the ABACUS restore map"
                        << " for I=" << ir_I << " J=" << ir_J << " R=(" << ir_R.x << ","
                        << ir_R.y << "," << ir_R.z << ")";
                    throw std::runtime_error(oss.str());
                }

                for (const auto& restore_member : pair_iter->second.at(ir_R))
                {
                    ComplexMatrix w_full = LIBRPA::rotate_abacus_abf_rspace_matrix(
                        symmetry_ctx, restore_member.isym, ir_I, ir_J, R_matrix.second);
                    auto& target =
                        tensors_full[restore_member.full_atom_pair.first]
                                    [restore_member.full_atom_pair.second][restore_member.full_R];
                    if (target.c == nullptr)
                    {
                        target = std::move(w_full);
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Duplicate full-sector W(R) block appears during ABACUS symmetry restore");
                    }
                }
            }
        }
    }
    return tensors_full;
}

bool can_use_abacus_irreducible_sector_wr_restore(const std::map<atom_t, size_t>& atom_nabf)
{
    (void)atom_nabf;
    // Keep the direct `IBZ W(q) -> irreducible W(R) -> full W(R)` path compiled but disabled.
    // The current LibRPA GW workflow still restores full `W(R)` before `set_Ws`, so this path
    // does not reduce the dominant LibRI contraction cost yet. On the current AlAs benchmark it
    // also does not improve wall time. Re-enable it only together with a symmetry-aware `Ws`
    // access/contraction path that can consume irreducible-sector `W(R)` directly.
    return false;
}

abf_rspace_complex_block_map_t accumulate_abacus_full_wr_from_ibz_q(
    const abf_qspace_complex_block_map_t& Wc_q,
    const int n_k_points,
    const std::vector<Vector3_Order<int>>& Rlist,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto local_target_pairs = collect_local_target_atom_pairs_from_qspace(Wc_q);
    const auto plan = build_abacus_irreducible_wr_plan(local_target_pairs, Rlist);
    if (!plan.available)
    {
        return {};
    }

    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    auto blocks_by_R_ir = allocate_abacus_irreducible_wr_storage(plan.local_irreducible_sector, atom_nabf);

    for (const auto& star_mapping : plan.kstar_grid_mapping)
    {
        const auto& star = ctx.kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        const LIBRPA::AbacusKStar* abf_star = nullptr;
        if (!ctx.abf_kstars.empty())
        {
            if (ctx.abf_kstars.size() != ctx.kstars.size())
            {
                throw std::runtime_error(
                    "ABF k-space symmetry sidecar count is inconsistent with symrot_k.txt");
            }
            abf_star = &ctx.abf_kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        }

        const auto q_ibz_internal = klist.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto k_ibz_frac = kfrac_list.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto blocks_ibz_local = collect_abacus_abf_ibz_blocks_for_q(Wc_q, q_ibz_internal);
        if (blocks_ibz_local.empty())
        {
            continue;
        }

        const std::array<double, 3> q_ibz_array{q_ibz_internal.x, q_ibz_internal.y, q_ibz_internal.z};
        auto blocks_ibz = gather_abacus_ibz_blocks_for_local_target_pairs(
            star, blocks_ibz_local, plan.local_irreducible_pairs, atom_nabf, q_ibz_array);
        if (blocks_ibz.empty())
        {
            continue;
        }

        blocks_ibz = symmetrize_abacus_abf_ibz_blocks(
            ctx, star, abf_star, k_ibz_frac, blocks_ibz, atom_nabf, plan.local_irreducible_pairs);
        if (star.members.size() != star_mapping.member_q_bz_keys.size())
        {
            throw std::runtime_error("ABACUS q-star mapping is inconsistent with the loaded full-q keys");
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& abf_member =
                (abf_star == nullptr) ? member : find_matching_abf_kstar_member(*abf_star, member);
            const bool use_time_reversal = member.isym >= plan.nsym_space;
            const auto q_bz_target_frac_vec =
                latvec * star_mapping.member_q_bz_keys[static_cast<std::size_t>(imember)];
            const Vector3_Order<double> q_bz_target_frac{
                q_bz_target_frac_vec.x, q_bz_target_frac_vec.y, q_bz_target_frac_vec.z};
            LIBRPA::abacus_atom_block_matrix_map_t rotated_blocks;
            try
            {
                rotated_blocks = LIBRPA::rotate_abacus_abf_kspace_operator_blocks(
                    ctx, abf_member, blocks_ibz, atom_nabf, star.k_ibz, coord_frac, use_time_reversal,
                    &plan.local_irreducible_pairs, &q_bz_target_frac);
            }
            catch (const std::exception& ex)
            {
                std::ostringstream oss;
                oss << "ABACUS irreducible-sector W(q)->W(R) accumulation failed for star="
                    << star.star_index << ", member=" << imember << ", isym=" << member.isym
                    << ": " << ex.what();
                throw std::runtime_error(oss.str());
            }

            const auto& q_internal = star_mapping.member_q_bz_keys[imember];
            for (const auto& atom_i_pair : rotated_blocks)
            {
                for (const auto& atom_j_pair : atom_i_pair.second)
                {
                    const auto sector_iter =
                        plan.local_irreducible_sector.find({atom_i_pair.first, atom_j_pair.first});
                    if (sector_iter == plan.local_irreducible_sector.end())
                    {
                        continue;
                    }

                    for (const auto& R_array : sector_iter->second)
                    {
                        const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
                        const auto phase = build_ft_wq_phase(q_internal, R, n_k_points);
                        add_scaled_complex_matrix(
                            blocks_by_R_ir.at(atom_i_pair.first).at(atom_j_pair.first).at(R),
                            atom_j_pair.second, phase);
                    }
                }
            }
        }
    }

    const auto blocks_by_R_full =
        restore_abacus_abf_rspace_dense_blocks(blocks_by_R_ir, ctx, plan.local_sector_stars);
    return convert_dense_rspace_blocks_to_row_major(blocks_by_R_full);
}

} // namespace

CorrEnergy compute_RPA_correlation_blacs_2d_gamma_only(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0)
        lib_printf("Calculating EcRPA with BLACS/ScaLAPACK 2D gamma_only\n");
    // lib_printf("Calculating EcRPA with BLACS, pid:  %d\n", mpi_comm_global_h.myid);
    const auto &mf = chi0.mf;
    const double CONE = 1.0;
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    // std::cout << "n_abf " << n_abf << std::endl;
    // std::cout << "n_atoms " << LIBRPA::atomic_basis_abf.n_atoms << std::endl;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();
    // std::cout << "part_range " << part_range[0] << " " << part_range[1] << std::endl;
    auto nbs_ = LIBRPA::atomic_basis_abf.get_atom_nbs();
    // std::cout << "nbs_ " << nbs_[0] << " " << nbs_[1] << std::endl;

    mpi_comm_global_h.barrier();

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about
    // illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto chi0_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);
    auto coul_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);

    vector<Vector3_Order<double>> qpts;
    // for (const auto &qMuNuchi : chi0.get_chi0_q().at(chi0.tfg.get_freq_nodes()[0]))
    //     qpts.push_back(qMuNuchi.first);
    for (const auto &q : chi0.klist)
    {
        qpts.push_back(q);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        printf("processId:%d, q: (%f, %f, %f)\n", mpi_comm_global_h.myid, q.x, q.y, q.z);
#endif
    }

    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    if (mpi_comm_global_h.is_root()) lib_printf("Finish init RPA blacs 2d\n");
#ifdef LIBRPA_USE_LIBRI
    for (const auto &q : qpts)
    {
        coul_block.zero_out();

        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<double>>>
                coul_libri;

            for (const auto &Mu_Nu : local_atpair)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "myid " << blacs_ctxt_global_h.myid << "Mu " << Mu << " Nu " << Nu <<
                // endl;
                if (coulmat.count(Mu) == 0 || coulmat.at(Mu).count(Nu) == 0 ||
                    coulmat.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                matrix tmp_vq_real = (*Vq).real();
                std::valarray<double> Vq_va(tmp_vq_real.c, Vq->size);
                auto pvq = std::make_shared<std::valarray<double>>();
                *pvq = Vq_va;
                coul_libri[Mu][{Nu, std::array<double, 3>{0, 0, 0}}] =
                    Tensor<double>({n_mu, n_nu}, pvq);
                coulmat.at(Mu).at(Nu).at(q).reset();
            }

            LIBRPA::utils::release_free_mem();

            // printf("Finish RPA blacs 2d  vq arr\n");
            double arr_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            double comm_begin = omp_get_wtime();
            // printf("Begin comm_map2_first  myid: %d\n",mpi_comm_global_h.myid);
            const auto IJq_coul =
                comm_map2_first(mpi_comm_global_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            mpi_comm_global_h.barrier();

            double block_begin = omp_get_wtime();

            collect_block_from_ALL_IJ_Tensor(coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);

            double block_end = omp_get_wtime();
            lib_printf(
                "Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f   pair_size: %d\n",
                mpi_comm_global_h.myid, arr_end - vq_begin, comm_end - comm_begin,
                block_end - block_begin, set_IJ_nabf_nabf.size());
            mpi_comm_global_h.barrier();
            double vq_end = omp_get_wtime();

            if (mpi_comm_global_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",
                           vq_end - vq_begin, comm_begin - vq_begin, block_begin - comm_begin,
                           vq_end - block_begin);
        }

        double chi_arr_time = 0.0;
        double chi_comm_time = 0.0;
        double chi_2d_time = 0.0;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            double pi_freq_begin = omp_get_wtime();
            chi0_block.zero_out();
            {
                double chi_begin_arr = omp_get_wtime();
                std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<double>>>
                    chi0_libri;
                // const auto &chi0_wq = chi0.get_chi0_q().at(freq).at(q);
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                if (!chi0.get_chi0_q().empty())
                {
                    chi0_wq = symmetrize_abacus_chi0_ibz_blocks_if_needed(
                        chi0.get_chi0_q().at(freq).at(q), q);
                }

                if (!chi0.get_chi0_q().empty())
                    for (const auto &M_Nchi : chi0_wq)
                    {
                        const auto &M = M_Nchi.first;
                        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                        for (const auto &N_chi : M_Nchi.second)
                        {
                            const auto &N = N_chi.first;
                            const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                            const auto &chi = N_chi.second.real();
                            std::valarray<double> chi_va(chi.c, chi.size);
                            auto pchi = std::make_shared<std::valarray<double>>();
                            *pchi = chi_va;
                            chi0_libri[M][{N, std::array<double, 3>{0, 0, 0}}] =
                                Tensor<double>({n_mu, n_nu}, pchi);
                        }
                    }

                // if(mpi_comm_global_h.is_root())
                // {
                //     lib_printf("Begin to clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }
                if (!chi0.get_chi0_q().empty()) chi0.free_chi0_q(freq, q);

                LIBRPA::utils::release_free_mem();

                // if(mpi_comm_global_h.is_root())
                // {
                //     lib_printf("After clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }

                mpi_comm_global_h.barrier();
                double chi_end_arr = omp_get_wtime();
                // ofs_myid << "chi0_libri" << endl << chi0_libri;

                const auto IJq_chi0 =
                    comm_map2_first(mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                double chi_end_comm = omp_get_wtime();

                collect_block_from_ALL_IJ_Tensor(chi0_block, desc_nabf_nabf,
                                                 LIBRPA::atomic_basis_abf, qa, true, CONE, IJq_chi0,
                                                 MAJOR::ROW);
                // printf("End collect block myid: %d ifreq: %d   TIME_USED:
                // %f\n",mpi_comm_global_h.myid,ifreq,chi_end_comm-chi_end_arr);
                mpi_comm_global_h.barrier();
                double chi_end_2d = omp_get_wtime();

                chi_arr_time = (chi_end_arr - chi_begin_arr);
                chi_comm_time = (chi_end_comm - chi_end_arr);
                chi_2d_time = (chi_end_2d - chi_end_comm);
            }

            double pi_begin = omp_get_wtime();
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coul_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc, 0.0, coul_chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc);
            // char fnp[100];
            // sprintf(fnp, "pi_ifreq_%d_iq_%d.mtx", ifreq, iq);
            double pi_end = omp_get_wtime();
            // printf("End pgemm  myid: %d ifreq: %d \n",mpi_comm_global_h.myid,ifreq);
            double trace_pi = 0.0;
            double trace_pi_loc = 0.0;
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (ilo >= 0 && jlo >= 0) trace_pi_loc += coul_chi0_block(ilo, jlo);
            }

            coul_chi0_block *= -1.0;
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (ilo >= 0 && jlo >= 0) coul_chi0_block(ilo, jlo) += CONE;
            }

            int *ipiv = new int[desc_nabf_nabf.m_loc() * 10];
            int info;
            // printf("begin det  myid: %d ifreq: %d \n",mpi_comm_global_h.myid,ifreq);
            double ln_det =
                compute_pi_det_blacs_2d_gamma_only(coul_chi0_block, desc_nabf_nabf, ipiv, info);
            // printf("End det  myid: %d ifreq: %d \n",mpi_comm_global_h.myid,ifreq);
            double det_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            MPI_Allreduce(&trace_pi_loc, &trace_pi, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_global_h.comm);
            double pi_freq_end = omp_get_wtime();

            if (mpi_comm_global_h.myid == 0)
            {
                lib_printf(
                    "| TIME of DET-freq-q:  %f,  q: ( %f, %f, %f)  TOT: %f  CHI_arr: %f  CHI_comm: "
                    "%f, CHI_2d: %f, Pi: %f, Det: %f\n",
                    freq, q.x, q.y, q.z, pi_freq_end - pi_freq_begin, chi_arr_time, chi_comm_time,
                    chi_2d_time, pi_end - pi_begin, det_end - pi_end);
                complex<double> rpa_for_omega_q = complex<double>(trace_pi + ln_det);
                /*std::cout << "q: " << iq << ", freq: " << ifreq << ", ln_det:" << ln_det
                          << ", trace_pi: " << trace_pi << ", rpa_for_omega_q" << rpa_for_omega_q
                          << ", contribution: "
                          << rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI << std::endl;*/
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;  //! check
                // std::cout << "rpa_for_omega_q: " << rpa_for_omega_q
                //          << ", freq_weight: " << freq_weight << ", irk_weight[q]:" <<
                //          irk_weight[q]
                //          << ", cRPA_q[q]: " << cRPA_q[q] << std::endl;
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    if (mpi_comm_global_h.myid == 0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
    }
    mpi_comm_global_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation_blacs_2d(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat)
{
    lib_printf("Begin to compute_RPA_correlation_blacs_2d  myid: %d\n", mpi_comm_global_h.myid);
    system("free -m");
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0) lib_printf("Calculating EcRPA with BLACS/ScaLAPACK 2D\n");
    // lib_printf("Calculating EcRPA with BLACS, pid:  %d\n", mpi_comm_global_h.myid);
    const auto &mf = chi0.mf;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    mpi_comm_global_h.barrier();

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about
    // illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
// ofs_myid << "Iset Jset " << s0_s1 << endl;
// ofs_myid << "atpair_unordered_local of myid " << blacs_ctxt_global_h.myid << " " <<
// atpair_unordered_local << endl;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    // printf("success before vector qpts
    // processid:%d,chi0.tfg.get_freq_nodes()[0]:%f,chi0.get_chi0_q().size():%d\n",
    // mpi_comm_global_h.myid,
    //    chi0.tfg.get_freq_nodes()[0], chi0.get_chi0_q().size());
    // printf("chi0.get_chi0_q().empty():%d\n", chi0.get_chi0_q().empty());
    printf("processId:%d,chi0.klist.size():%zu\n", mpi_comm_global_h.myid, chi0.klist.size());
// for(const auto &k : chi0.klist)
// {
//     printf("processId:%d, k: (%f, %f, %f)\n", mpi_comm_global_h.myid, k.x, k.y, k.z);
// }
#endif
    vector<Vector3_Order<double>> qpts;

    // for (const auto &qMuNuchi : chi0.get_chi0_q().at(chi0.tfg.get_freq_nodes()[0]))
    // {
    //     qpts.push_back(qMuNuchi.first);
    //     #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    //     const auto &q = qMuNuchi.first;
    //     printf("processId:%d, q: (%f, %f, %f)\n", mpi_comm_global_h.myid, q.x, q.y, q.z);
    //     #endif
    // }
    for (const auto &q : chi0.klist)
    {
        qpts.push_back(q);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        printf("processId:%d, q: (%f, %f, %f)\n", mpi_comm_global_h.myid, q.x, q.y, q.z);
#endif
    }
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    if (mpi_comm_global_h.is_root()) lib_printf("Finish init RPA blacs 2d\n");
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    printf("success before for loop processid:%d\n", mpi_comm_global_h.myid);
#endif
#ifdef LIBRPA_USE_LIBRI

    for (const auto &q : qpts)
    {
        coul_block.zero_out();

        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                coul_libri;
            coul_libri.clear();
            for (const auto &Mu_Nu : local_atpair)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
// ofs_myid << "myid " << blacs_ctxt_global_h.myid << "Mu " << Mu << " Nu " << Nu <<
// endl;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before if coulmat.count:%d\n", mpi_comm_global_h.myid);
#endif
                if (coulmat.count(Mu) == 0 || coulmat.at(Mu).count(Nu) == 0 ||
                    coulmat.at(Mu).at(Nu).count(q) == 0)
                    continue;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after if coulmat.count:%d\n", mpi_comm_global_h.myid);
#endif
                const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                coul_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            // printf("Finish RPA blacs 2d  vq arr\n");
            double arr_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            double comm_begin = omp_get_wtime();
            // printf("Begin comm_map2_first  myid: %d\n",mpi_comm_global_h.myid);
            const auto IJq_coul =
                comm_map2_first(mpi_comm_global_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            // printf("End vq comm_map2_first  myid: %d   TIME_USED:
            // %f\n",mpi_comm_global_h.myid,comm_end-comm_begin);
            //  ofs_myid << "IJq_coul" << endl << IJq_coul;
            // printf("Finish RPA blacs 2d  vq 2d\n");
            double block_begin = omp_get_wtime();
            // for (const auto &IJ: set_IJ_nabf_nabf)
            // {
            //     const auto &I = IJ.first;
            //     const auto &J = IJ.second;
            //     // cout << IJq_coul.at(I).at({J, qa});
            //     collect_block_from_IJ_storage_syhe(
            //         coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
            //         IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            //     // lib_printf("myid %d I %d J %d nr %d nc %d\n%s",
            //     //        blacs_ctxt_global_h.myid, I, J,
            //     //        coul_block.nr(), coul_block.nc(),
            //     //        str(coul_block).c_str());
            // }
            collect_block_from_ALL_IJ_Tensor(coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            double block_end = omp_get_wtime();
            lib_printf(
                "Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f   pair_size: %d\n",
                mpi_comm_global_h.myid, arr_end - vq_begin, comm_end - comm_begin,
                block_end - block_begin, set_IJ_nabf_nabf.size());
            mpi_comm_global_h.barrier();
            double vq_end = omp_get_wtime();

            if (mpi_comm_global_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",
                           vq_end - vq_begin, comm_begin - vq_begin, block_begin - comm_begin,
                           vq_end - block_begin);
        }

        // if(mpi_comm_global_h.is_root())
        // printf("Finish RPA blacs 2d  vq comm\n");
        //  char fn[100];
        //  sprintf(fn, "coul_iq_%d.mtx", iq);
        //  print_matrix_mm_file_parallel(fn, coul_block, desc_nabf_nabf);
        //  ofs_myid << str(coul_block);
        //  lib_printf("coul_block\n%s", str(coul_block).c_str());
        double chi_arr_time = 0.0;
        double chi_comm_time = 0.0;
        double chi_2d_time = 0.0;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            double pi_freq_begin = omp_get_wtime();
            chi0_block.zero_out();
            {
                double chi_begin_arr = omp_get_wtime();
                std::map<int,
                         std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                    chi0_libri;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before chi0.get_chi0_q().at(freq).at(q) processId:%d\n", mpi_comm_global_h.myid);
// printf("processId:%d,chi0.get_chi0_q().empty():%d\n", mpi_comm_global_h.myid,
// chi0.get_chi0_q().empty());
#endif
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                if (!chi0.get_chi0_q().empty())
                {
                    chi0_wq = symmetrize_abacus_chi0_ibz_blocks_if_needed(
                        chi0.get_chi0_q().at(freq).at(q), q);
                }
// const auto &chi0_wq = chi0.get_chi0_q().at(freq).at(q);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after chi0.get_chi0_q().at(freq).at(q) processId:%d\n", mpi_comm_global_h.myid);
#endif
                chi0_libri.clear();
                if (!chi0.get_chi0_q().empty())
                    for (const auto &M_Nchi : chi0_wq)
                    {
                        const auto &M = M_Nchi.first;
                        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                        for (const auto &N_chi : M_Nchi.second)
                        {
                            const auto &N = N_chi.first;
                            const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                            const auto &chi = N_chi.second;
                            std::valarray<complex<double>> chi_va(chi.c, chi.size);
                            auto pchi = std::make_shared<std::valarray<complex<double>>>();
                            *pchi = chi_va;
                            chi0_libri[M][{N, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pchi);
                        }
                    }
                if (mpi_comm_global_h.is_root())
                {
                    lib_printf("Begin to clean chi0 !!! \n");
                    LIBRPA::utils::display_free_mem();
                    lib_printf("chi0_freq_q size: %d,  freq: %f, q:( %f, %f, %f )\n",
                               chi0_wq.size(), freq, q.x, q.y, q.z);
                }
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before chi0.free_chi0_q(freq, q) processId:%d\n", mpi_comm_global_h.myid);
#endif
                if (!chi0.get_chi0_q().empty()) chi0.free_chi0_q(freq, q);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after chi0.free_chi0_q(freq, q) processId:%d\n", mpi_comm_global_h.myid);
#endif

                LIBRPA::utils::release_free_mem();
                // if(mpi_comm_global_h.is_root())
                // {
                //     lib_printf("After clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }
                mpi_comm_global_h.barrier();
                double chi_end_arr = omp_get_wtime();
                // ofs_myid << "chi0_libri" << endl << chi0_libri;

                const auto IJq_chi0 =
                    comm_map2_first(mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                double chi_end_comm = omp_get_wtime();
                collect_block_from_ALL_IJ_Tensor(chi0_block, desc_nabf_nabf,
                                                 LIBRPA::atomic_basis_abf, qa, true, CONE, IJq_chi0,
                                                 MAJOR::ROW);
                mpi_comm_global_h.barrier();
                double chi_end_2d = omp_get_wtime();

                chi_arr_time = (chi_end_arr - chi_begin_arr);
                chi_comm_time = (chi_end_comm - chi_end_arr);
                chi_2d_time = (chi_end_2d - chi_end_comm);
                // char fnc[100];
                // sprintf(fnc, "chi_ifreq_%d_iq_%d.mtx", ifreq, iq);
                // if( ifreq== 0)
                //     print_matrix_mm_file_parallel(fnc, chi0_block, desc_nabf_nabf);
            }

            double pi_begin = omp_get_wtime();
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coul_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc, 0.0, coul_chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf.desc);
            // char fnp[100];
            // sprintf(fnp, "pi_ifreq_%d_iq_%d.mtx", ifreq, iq);
            double pi_end = omp_get_wtime();

            complex<double> trace_pi(0.0, 0.0);
            complex<double> trace_pi_loc(0.0, 0.0);
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (ilo >= 0 && jlo >= 0) trace_pi_loc += coul_chi0_block(ilo, jlo);
            }

            coul_chi0_block *= -1.0;
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (ilo >= 0 && jlo >= 0) coul_chi0_block(ilo, jlo) += CONE;
                // std::cout << "1-Pi: " << ilo << "," << jlo << "," << coul_chi0_block(ilo, jlo)
                //<< std::endl;
            }
            // if( ifreq== 0 && mpi_comm_global_h.is_root() )
            //     print_whole_matrix("pi-2D-loc", coul_chi0_block);

            int *ipiv = new int[desc_nabf_nabf.m_loc() * 10];
            int info;
            complex<double> ln_det =
                compute_pi_det_blacs_2d(coul_chi0_block, desc_nabf_nabf, ipiv, info);
            double det_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            MPI_Allreduce(&trace_pi_loc, &trace_pi, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                          mpi_comm_global_h.comm);
            double pi_freq_end = omp_get_wtime();
            // double task_end = omp_get_wtime();
            //  if(mpi_comm_global_h.is_root())
            //      lib_printf("| After det for freq:  %f,  q: ( %f, %f, %f)   TIME_LOCMAT: %f
            //      TIME_DET: %f  TIME_CAL_Pi: %f, TIME_TRAN_LOC: %f\n",ifreq,
            //      q.x,q.y,q.z,task_mid-task_begin,task_end-task_mid,pi_time,loc_tran_time);
            // para_mpi.mpi_barrier();

            if (mpi_comm_global_h.myid == 0)
            {
                lib_printf(
                    "| TIME of DET-freq-q:  %f,  q: ( %f, %f, %f)  TOT: %f  CHI_arr: %f  CHI_comm: "
                    "%f, CHI_2d: %f, Pi: %f, Det: %f\n",
                    freq, q.x, q.y, q.z, pi_freq_end - pi_freq_begin, chi_arr_time, chi_comm_time,
                    chi_2d_time, pi_end - pi_begin, det_end - pi_end);
                complex<double> rpa_for_omega_q = trace_pi + ln_det;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;  //! check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    if (mpi_comm_global_h.myid == 0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            // cout << q_crpa.first << q_crpa.second << endl;
        }
        // cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)
        // <<tot_RPA_energy << endl;
    }
    mpi_comm_global_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    return corr;
}
double compute_pi_det_blacs_2d_gamma_only(matrix_m<double> &loc_piT, const Array_Desc &arrdesc_pi,
                                          int *ipiv, int &info)
{
    int one = 1;
    int range_all = N_all_mu;
    int DESCPI_T[9];

    double det_begin = omp_get_wtime();

    ScalapackConnector::pgetrf_f(range_all, range_all, loc_piT.ptr(), one, one, arrdesc_pi.desc,
                                 ipiv, info);
    double trf_end = omp_get_wtime();

    double ln_det_loc = 0.0;
    double ln_det_all = 0.0;

    for (int ig = 0; ig != range_all; ig++)
    {
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            double tmp_ln_det;
            if (loc_piT(locr, locc) > 0)
            {
                tmp_ln_det = std::log(loc_piT(locr, locc));
            }
            else
            {
                tmp_ln_det = std::log(-loc_piT(locr, locc));
            }
            ln_det_loc += tmp_ln_det;
        }
    }
    double ln_end = omp_get_wtime();

    MPI_Allreduce(&ln_det_loc, &ln_det_all, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_global_h.comm);
    double det_end = omp_get_wtime();
    return ln_det_all;
}
complex<double> compute_pi_det_blacs_2d(matrix_m<complex<double>> &loc_piT,
                                        const Array_Desc &arrdesc_pi, int *ipiv, int &info)
{
    int one = 1;
    int range_all = N_all_mu;
    int DESCPI_T[9];
// if(out_pi)
// {
//     print_complex_real_matrix("first_pi",pi_freq_q.at(0).at(0));
//     print_complex_real_matrix("first_loc_piT_mat",loc_piT);
// }
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    printf(
        "success before pzgetrf_ processid:%d,range_all: %d, loc_piT.nr(): %d, loc_piT.nc(): %d\n",
        mpi_comm_global_h.myid, range_all, loc_piT.nr(), loc_piT.nc());
#endif
    double det_begin = omp_get_wtime();
    // ScalapackConnector::transpose_desc(DESCPI_T, arrdesc_pi.desc);
    pzgetrf_(&range_all, &range_all, loc_piT.ptr(), &one, &one, arrdesc_pi.desc, ipiv, &info);
    double trf_end = omp_get_wtime();
    // ScalapackConnector::pgetrf_f(range_all,range_all,loc_piT.c,one,one,DESCPI_T,ipiv, info);
    // printf("   after LU myid: %d\n",mpi_comm_global_h.myid);
    // printf("desc myid: %d,  m n: %d,%d,  mb nb: %d, %d,  loc_m_n: %d, %d, myp: %d,%d, npr,npc:
    // %d, %d\n",mpi_comm_global_h.myid, arrdesc_pi.m(),arrdesc_pi.n(),
    // arrdesc_pi.mb(),arrdesc_pi.nb(),
    // arrdesc_pi.m_loc(),arrdesc_pi.n_loc(),arrdesc_pi.myprow(),arrdesc_pi.mypcol(),arrdesc_pi.nprows(),arrdesc_pi.npcols());
    complex<double> ln_det_loc(0.0, 0.0);
    complex<double> ln_det_all(0.0, 0.0);
    // complex<double> det_loc(1.0,0.0);
    // complex<double> det_glo(0.0,0.0);
    // vector<complex<double>>  det_dig;
    // vector<complex<double>>  ln_det_dig;
    // vector<complex<double>>  det_dig_r;
    // vector<complex<double>>  det_dig_c;
    // printf(" myid: %d ig=25, locr,locc: %d,
    // %d)\n",mpi_comm_global_h.myid,arrdesc_pi.indx_g2l_r(25),arrdesc_pi.indx_g2l_c(25));
    for (int ig = 0; ig != range_all; ig++)
    {
        // int locr=para_mpi.localIndex(ig,row_nblk,para_mpi.nprow,para_mpi.myprow);
        // int locc=para_mpi.localIndex(ig,col_nblk,para_mpi.npcol,para_mpi.mypcol);
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            // if(ipiv[locr]!=(ig+1))
            // 	det_loc=-1*det_loc * loc_piT(locc,locr);
            // else
            // 	det_loc=det_loc * loc_piT(locc,locr);
            // det_dig.push_back(loc_piT(locr,locc));
            // det_dig_r.push_back(locr);
            // det_dig_c.push_back(locc);
            complex<double> tmp_ln_det;
            if (loc_piT(locr, locc).real() > 0)
            {
                tmp_ln_det = std::log(loc_piT(locr, locc));
                // ln_det_dig.push_back(tmp_ln_det);
            }
            else
            {
                tmp_ln_det = std::log(-loc_piT(locr, locc));
                // ln_det_dig.push_back(tmp_ln_det);
            }
            ln_det_loc += tmp_ln_det;
        }
    }
    double ln_end = omp_get_wtime();
    //     ComplexMatrix det_mm(loc_piT.nr(),loc_piT.nc());
    //     for(int i=0;i!=loc_piT.nr();i++)
    //         for(int j=0;j!=loc_piT.nc();j++)
    //             det_mm(i,j)=loc_piT(i,j);
    //    // sort(det_dig.rbegin(),det_dig.rend());
    //     ComplexMatrix det_dig_mm(det_dig.size(),4);
    //     for(int i=0;i!=det_dig.size();i++)
    //     {
    //         det_dig_mm(i,0) =det_dig_r[i];
    //         det_dig_mm(i,1) =det_dig_c[i];
    //         det_dig_mm(i,2)=det_dig[i];
    //         det_dig_mm(i,3)=ln_det_dig[i];
    //     }
    //     char fn[100];
    //     sprintf(fn, "det_dig_myid_%d.mtx", mpi_comm_global_h.myid);
    //     print_complex_matrix_file("det_dig_loc", det_dig_mm, fn, false);

    //     sprintf(fn, "det_mat_myid_%d.mtx", mpi_comm_global_h.myid);
    //     print_complex_matrix_file("det_mat_loc", det_mm, fn, false);

    MPI_Allreduce(&ln_det_loc, &ln_det_all, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_global_h.comm);
    double det_end = omp_get_wtime();
    // if(mpi_comm_global_h.myid == 0)
    //     lib_printf("    | Det time   trf: %f   ln: %f   allreduce:
    //     %f\n",trf_end-det_begin,ln_end-trf_end, det_end-ln_end);
    // MPI_Allreduce(&det_loc,&det_glo,1,MPI_DOUBLE_COMPLEX,MPI_PROD,mpi_comm_global_h.comm);
    // ln_det_all=std::log(det_glo);
    return ln_det_all;
}

complex<double> compute_pi_det_blacs(ComplexMatrix &loc_piT, const Array_Desc &arrdesc_pi,
                                     int *ipiv, int &info)
{
    // int range_all = atom_mu_part_range[natom-1]+atom_mu[natom-1];
    // int desc_pi[9];
    // int loc_row, loc_col, info;
    // int row_nblk=1;
    // int col_nblk=1;
    int one = 1;
    int range_all = N_all_mu;
    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,range_all,range_all,row_nblk,col_nblk);
    // int *ipiv = new int [loc_row*10];
    // ComplexMatrix loc_piT(loc_col,loc_row);

    // for(int i=0;i!=loc_row;i++)
    // {
    //     int global_row = para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
    //     int mu;
    //     int I=atom_mu_glo2loc(global_row,mu);
    //     for(int j=0;j!=loc_col;j++)
    //     {
    //         int global_col = para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
    //         int nu;
    //         int J=atom_mu_glo2loc(global_col,nu);

    //         if( global_col == global_row)
    //         {
    //             loc_piT(j,i)=complex<double>(1.0,0.0) - pi_freq_q.at(I).at(J)(mu,nu);
    //         }
    //         else
    //         {
    //             loc_piT(j,i)=-1*  pi_freq_q.at(I).at(J)(mu,nu);
    //         }

    //     }
    // }
    int DESCPI_T[9];
    // if(out_pi)
    // {
    //     print_complex_real_matrix("first_pi",pi_freq_q.at(0).at(0));
    //     print_complex_real_matrix("first_loc_piT_mat",loc_piT);
    // }

    ScalapackConnector::transpose_desc(DESCPI_T, arrdesc_pi.desc);

    // para_mpi.mpi_barrier();
    // printf("   before LU Myid: %d        Available DOS memory = %ld
    // bytes\n",mpi_comm_global_h.myid, memavail()); printf("   before LU myid: %d  range_all: %d,
    // loc_mat.size: %d\n",mpi_comm_global_h.myid,range_all,loc_piT.size);
    pzgetrf_(&range_all, &range_all, loc_piT.c, &one, &one, DESCPI_T, ipiv, &info);
    // printf("   after LU myid: %d\n",mpi_comm_global_h.myid);
    complex<double> ln_det_loc(0.0, 0.0);
    complex<double> ln_det_all(0.0, 0.0);
    for (int ig = 0; ig != range_all; ig++)
    {
        // int locr=para_mpi.localIndex(ig,row_nblk,para_mpi.nprow,para_mpi.myprow);
        // int locc=para_mpi.localIndex(ig,col_nblk,para_mpi.npcol,para_mpi.mypcol);
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            // if(ipiv[locr]!=(ig+1))
            // 	det_loc=-1*det_loc * loc_piT(locc,locr);
            // else
            // 	det_loc=det_loc * loc_piT(locc,locr);
            if (loc_piT(locc, locr).real() > 0)
                ln_det_loc += std::log(loc_piT(locc, locr));
            else
                ln_det_loc += std::log(-loc_piT(locc, locr));
        }
    }
    MPI_Allreduce(&ln_det_loc, &ln_det_all, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_global_h.comm);
    return ln_det_all;
}

CorrEnergy compute_RPA_correlation_blacs(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0) lib_printf("Calculating EcRPA with BLACS/ScaLAPACK row\n");

    const auto &mf = chi0.mf;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    mpi_comm_global_h.barrier();

    LIBRPA::Array_Desc arrdesc_pi(blacs_ctxt_global_h);
    arrdesc_pi.init_square_blk(n_abf, n_abf, 0, 0);
    int loc_row = arrdesc_pi.m_loc(), loc_col = arrdesc_pi.n_loc(), info;

    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,N_all_mu,N_all_mu,row_nblk,col_nblk);
    int *ipiv = new int[loc_row * 10];
    // double vq_begin_m2t= omp_get_wtime();
    // std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
    // vq_libri; for(auto &Ip:Vq)
    // {
    //     auto I=Ip.first;
    //     const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(I);
    //     for(auto &Jp:Ip.second)
    //     {
    //         auto J=Jp.first;
    //         const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(J);
    //         for(auto &qp:Jp.second)
    //         {
    //             auto q=qp.first;
    //             std::array<double, 3> qa = {q.x, q.y, q.z};
    //             const auto &vq_ptr=qp.second;
    //             std::valarray<complex<double>> Vq_va(vq_ptr->c, vq_ptr->size);
    //             auto pvq = std::make_shared<std::valarray<complex<double>>>();
    //             *pvq = Vq_va;
    //             vq_libri[I][{J, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
    //             if(I!=J)
    //             {
    //                 auto vqT=transpose(*vq_ptr, 1);
    //                 std::valarray<complex<double>> VqT_va(vqT.c, vqT.size);
    //                 auto pvqT = std::make_shared<std::valarray<complex<double>>>();
    //                 *pvqT = VqT_va;
    //                 vq_libri[J][{I, qa}] = Tensor<complex<double>>({n_nu, n_mu}, pvqT);
    //             }
    //         }
    //     }
    // }
    // double vq_end_m2t = omp_get_wtime();
    // set<int> loc_atp_IJ;
    // for(auto &atp:local_atpair)
    // {
    //     loc_atp_IJ.insert(atp.first);
    //     loc_atp_IJ.insert(atp.second);
    // }
    // set<int> all_atom_set;
    // for(int I=0;I!=natom;I++)
    //     all_atom_set.insert(I);
    // const auto IJq_coul = Communicate_Tensors_Map_Judge::comm_map2_first(mpi_comm_global_h.comm,
    // vq_libri, all_atom_set, loc_atp_IJ); atpair_k_cplx_mat_t Vq_loc; double vq_end_comm =
    // omp_get_wtime(); for(auto Ip:IJq_coul)
    // {
    //     auto I=Ip.first;
    //     auto n_mu=atom_mu[I];
    //     for(auto &Jqp:Ip.second)
    //     {
    //         auto J=Jqp.first.first;
    //         auto n_nu=atom_mu[J];
    //         auto qa=Jqp.first.second;
    //         Vector3_Order<double> q{qa[0],qa[1],qa[2]};
    //         shared_ptr<ComplexMatrix> vq_ptr = make_shared<ComplexMatrix>();
    //         vq_ptr->create(n_mu, n_nu);
    //         const auto length=sizeof(complex<double>)* n_mu *n_nu;
    //         memcpy((*vq_ptr).c, Jqp.second.ptr(),length);
    //         Vq_loc[I][J][q]=vq_ptr;
    //         //printf("| process %d, I: %d  J: %d\n",mpi_comm_global_h.myid, I,J );
    //     }
    // }
    // double vq_end_t2m = omp_get_wtime();
    // mpi_comm_global_h.barrier();
    // if(mpi_comm_global_h.is_root())
    //     lib_printf("| Vq_time %f, TIME_m2t: %f   TIME_comm: %f  TIME_t2m:
    //     %f\n",vq_end_t2m-vq_begin_m2t,vq_end_m2t-vq_begin_m2t,vq_end_comm-vq_end_m2t,vq_end_t2m-vq_end_comm);
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    for (const auto &freq_q_MuNuchi0 : chi0.get_chi0_q())
    {
        const auto freq = freq_q_MuNuchi0.first;
        const double freq_weight = chi0.tfg.find_freq_weight(freq);
        for (const auto &q_MuNuchi0 : freq_q_MuNuchi0.second)
        {
            double task_begin = omp_get_wtime();
            const auto q = q_MuNuchi0.first;
            auto &MuNuchi0 = q_MuNuchi0.second;

            // ComplexMatrix loc_piT(loc_col,loc_row);
            auto loc_piT = init_local_mat<complex<double>>(arrdesc_pi, MAJOR::COL);
            complex<double> trace_pi(0.0, 0.0);
            double vq_time = 0.0;
            double pi_time = 0.0;
            double loc_tran_time = 0.0;
            for (int Mu = 0; Mu != natom; Mu++)
            {
                double Mu_begin = omp_get_wtime();
                // lib_printf(" |process %d,  Mu:  %d\n",mpi_comm_global_h.myid,Mu);
                const size_t n_mu = atom_mu[Mu];
                atom_mapping<ComplexMatrix>::pair_t_old Vq_row = gather_vq_row_q(Mu, coulmat, q);
                double Mu_after_vq = omp_get_wtime();
                // atom_mapping<ComplexMatrix>::pair_t_old Vq_row;
                // const auto IJq_coul =
                // Communicate_Tensors_Map_Judge::comm_map2_first(mpi_comm_global_h.comm, vq_libri,
                // {Mu}, loc_atp_atoms); double Mu_vq_comm = omp_get_wtime(); for(auto Ip:IJq_coul)
                // {
                //     auto I=Ip.first;
                //     auto n_mu=atom_mu[I];
                //     for(auto &Jqp:Ip.second)
                //     {
                //         auto J=Jqp.first.first;
                //         auto n_nu=atom_mu[J];
                //         auto q=Jqp.first.second;
                //         Vq_row[I][J].create(n_mu,n_nu);
                //         const auto length=sizeof(complex<double>)* n_mu *n_nu;
                //         memcpy(Vq_row[I][J].c, Jqp.second.ptr(),length);
                //     }
                // }
                // double Mu_after_vq=omp_get_wtime();
                // printf("   |process %d, Mu: %d  vq_row.size:
                // %d\n",para_mpi.get_myid(),Mu,Vq_row[Mu].size()); ComplexMatrix
                // loc_pi_row=compute_Pi_freq_q_row(q,MuNuchi0,Vq_loc,Mu,q);
                ComplexMatrix loc_pi_row = compute_Pi_freq_q_row(q, MuNuchi0, Vq_row, Mu);
                // printf("   |process %d,   compute_pi\n",para_mpi.get_myid());
                ComplexMatrix glo_pi_row(n_mu, N_all_mu);
                mpi_comm_global_h.barrier();
                mpi_comm_global_h.allreduce_ComplexMatrix(loc_pi_row, glo_pi_row);
                double Mu_after_pi_loc = omp_get_wtime();
                // cout<<"  glo_pi_rowT nr,nc: "<<glo_pi_row.nr<<" "<<glo_pi_row.nc<<endl;

                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    trace_pi += glo_pi_row(i_mu, atom_mu_part_range[Mu] + i_mu);
                // select glo_pi_rowT to pi_blacs
                for (int i = 0; i != loc_row; i++)
                {
                    // int global_row =
                    // para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
                    int global_row = arrdesc_pi.indx_l2g_r(i);
                    int mu_blacs;
                    int I_blacs = atom_mu_glo2loc(global_row, mu_blacs);
                    if (I_blacs == Mu)
                        for (int j = 0; j != loc_col; j++)
                        {
                            // int global_col =
                            // para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
                            int global_col = arrdesc_pi.indx_l2g_c(j);
                            int nu_blacs;
                            int J_blacs = atom_mu_glo2loc(global_col, nu_blacs);
                            // cout<<" Mu: "<<Mu<<"  i,j: "<<i<<"  "<<j<<"    glo_row,col:
                            // "<<global_row<<"  "<<global_col<<"  J:"<<J_blacs<< "  index i,j:
                            // "<<atom_mu_part_range[J_blacs] + mu_blacs<<" "<<nu_blacs<<endl;
                            if (global_col == global_row)
                            {
                                loc_piT(i, j) =
                                    complex<double>(1.0, 0.0) -
                                    glo_pi_row(mu_blacs, atom_mu_part_range[J_blacs] + nu_blacs);
                            }
                            else
                            {
                                loc_piT(i, j) =
                                    -glo_pi_row(mu_blacs, atom_mu_part_range[J_blacs] + nu_blacs);
                            }
                        }
                }
                double Mu_after_loc_tran = omp_get_wtime();
                vq_time += (Mu_after_vq - Mu_begin);
                pi_time += (Mu_after_pi_loc - Mu_after_vq);
                loc_tran_time += (Mu_after_loc_tran - Mu_after_pi_loc);
            }
            // if(freq == chi0.tfg.get_freq_nodes()[0] && mpi_comm_global_h.is_root())
            //     print_complex_matrix(" loc_piT",loc_piT);
            double task_mid = omp_get_wtime();
            // printf("|process  %d, before det\n",mpi_comm_global_h.myid);
            complex<double> ln_det = compute_pi_det_blacs_2d(loc_piT, arrdesc_pi, ipiv, info);
            double task_end = omp_get_wtime();
            if (mpi_comm_global_h.is_root())
                lib_printf(
                    "| After det for freq:  %f,  q: ( %f, %f, %f)   TIME_Vq_COMM: %f   TIME_DET: "
                    "%f  TIME_CAL_Pi: %f, TIME_TRAN_LOC: %f\n",
                    freq, q.x, q.y, q.z, vq_time, task_end - task_mid, pi_time, loc_tran_time);
            // para_mpi.mpi_barrier();
            if (mpi_comm_global_h.myid == 0)
            {
                complex<double> rpa_for_omega_q = trace_pi + ln_det;
                // cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "
                // lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;  //! check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
    }

    if (mpi_comm_global_h.myid == 0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            // cout << q_crpa.first << q_crpa.second << endl;
        }
        // cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)
        // <<tot_RPA_energy << endl;
    }
    mpi_comm_global_h.barrier();
    corr.value = tot_RPA_energy;
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    if (mpi_comm_global_h.myid == 0) lib_printf("Calculating EcRPA without BLACS/ScaLAPACK\n");
    // lib_printf("Begin cal cRPA , pid:  %d\n", mpi_comm_global_h.myid);
    const auto &mf = chi0.mf;

    // freq, q
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>
        pi_freq_q_Mu_Nu;
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::ATOM_PAIR ||
        LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI)
        pi_freq_q_Mu_Nu = compute_Pi_q_MPI(chi0, coulmat);
    else
        pi_freq_q_Mu_Nu = compute_Pi_q(chi0, coulmat);
    lib_printf("Finish Pi freq on Proc %4d, size %zu\n", mpi_comm_global_h.myid,
               pi_freq_q_Mu_Nu.size());
    // mpi_comm_global_h.barrier();

    int range_all = N_all_mu;

    vector<int> part_range;
    part_range.resize(atom_mu.size());
    part_range[0] = 0;
    int count_range = 0;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before part_range processid:%d, atom_mu.size(): %zu\n",
//    mpi_comm_global_h.myid, atom_mu.size());
#endif
    for (int I = 0; I != atom_mu.size() - 1; I++)
    {
        count_range += atom_mu[I];
        part_range[I + 1] = count_range;
    }
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after part_range processid:%d, atom_mu.size(): %zu\n",
//        mpi_comm_global_h.myid, atom_mu.size());
#endif

    // cout << "part_range:" << endl;
    // for (int I = 0; I != atom_mu.size(); I++)
    // {
    //     cout << part_range[I] << endl;
    // }
    // cout << "part_range over" << endl;

    // pi_freq_q contains all atoms
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("| process %d, qpts.size(): %zu,freq.size():%zu\n", mpi_comm_global_h.myid,
// chi0.klist.size(),chi0.tfg.get_freq_nodes().size());
#endif
    for (const auto &freq : chi0.tfg.get_freq_nodes())
    {
        // printf("| process %d, freq: %f\n", mpi_comm_global_h.myid, freq);
        map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old> freq_q_MuNupi;
        if (!chi0.get_chi0_q().empty()) freq_q_MuNupi = pi_freq_q_Mu_Nu.at(freq);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before freq_q_MuNupi processid:%d, freq_q_MuNupi.size(): %zu\n",
//        mpi_comm_global_h.myid, freq_q_MuNupi.size());
#endif
        for (const auto &q : chi0.klist)
        {
            atom_mapping<ComplexMatrix>::pair_t_old q_MuNupi;
            if (!chi0.get_chi0_q().empty()) q_MuNupi = freq_q_MuNupi.at(q);
            const auto MuNupi = q_MuNupi;
            pi_freq_q[freq][q].create(range_all, range_all);

            ComplexMatrix pi_munu_tmp(range_all, range_all);
            pi_munu_tmp.zero_out();
            if (!chi0.get_chi0_q().empty())
                for (const auto &Mu_Nupi : MuNupi)
                {
                    const auto Mu = Mu_Nupi.first;
                    const auto Nupi = Mu_Nupi.second;
                    const size_t n_mu = atom_mu[Mu];
                    for (const auto &Nu_pi : Nupi)
                    {
                        const auto Nu = Nu_pi.first;
                        const auto pimat = Nu_pi.second;
                        const size_t n_nu = atom_mu[Nu];

                        for (size_t mu = 0; mu != n_mu; ++mu)
                        {
                            for (size_t nu = 0; nu != n_nu; ++nu)
                            {
                                pi_munu_tmp(part_range[Mu] + mu, part_range[Nu] + nu) +=
                                    pimat(mu, nu);
                            }
                        }
                    }
                }
            if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::ATOM_PAIR ||
                LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI)
            {
                mpi_comm_global_h.reduce_ComplexMatrix(pi_munu_tmp, pi_freq_q.at(freq).at(q), 0);
            }
            else
            {
                pi_freq_q.at(freq).at(q) = std::move(pi_munu_tmp);
            }
        }
    }
    // lib_printf("Finish Pi communicate %4d, size %zu\n", mpi_comm_global_h.myid,
    // pi_freq_q_Mu_Nu.size());
    mpi_comm_global_h.barrier();
    // if (mpi_comm_global_h.myid == 0)
    {
        complex<double> tot_RPA_energy(0.0, 0.0);
        map<Vector3_Order<double>, complex<double>> cRPA_q;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        int num_iteration = 0;
#endif
        for (const auto &freq_qpi : pi_freq_q)
        {
            const auto freq = freq_qpi.first;
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            for (const auto &q_pi : freq_qpi.second)
            {
                const auto q = q_pi.first;
                const auto pimat = q_pi.second;
                complex<double> rpa_for_omega_q(0.0, 0.0);
                ComplexMatrix identity(range_all, range_all);
                ComplexMatrix identity_minus_pi(range_all, range_all);
                identity.set_as_identity_matrix();
                identity_minus_pi = identity - pi_freq_q[freq][q];
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
                // if(num_iteration==0)
                // if(mpi_comm_global_h.myid == 1)
                // {
                //     complex<double>* test_c= identity_minus_pi.c;
                //     for(int i=0;i<range_all;i++){
                //         for(int j=0;j<range_all;j++){
                //             printf("%f+%fi ",
                //                    test_c[i*range_all+j].real(), test_c[i*range_all+j].imag());
                //         }
                //         printf("\n");
                //     }
                // }
                num_iteration++;
#endif
                complex<double> det_for_rpa(1.0, 0.0);
                int info_LU = 0;
                int *ipiv = new int[range_all];
                LapackConnector::zgetrf(range_all, range_all, identity_minus_pi, range_all, ipiv,
                                        &info_LU);
                for (int ib = 0; ib != range_all; ib++)
                {
                    if (ipiv[ib] != (ib + 1))
                        det_for_rpa = -det_for_rpa * identity_minus_pi(ib, ib);
                    else
                        det_for_rpa = det_for_rpa * identity_minus_pi(ib, ib);
                }
                delete[] ipiv;

                complex<double> trace_pi;
                complex<double> ln_det;
                ln_det = std::log(det_for_rpa);
                trace_pi = trace(pi_freq_q.at(freq).at(q));
                // cout << "PI trace vector:" << endl;
                // cout << endl;
                rpa_for_omega_q = ln_det + trace_pi;
                // cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "
                // lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
        // lib_printf("Finish EcRPA %4d, size %zu\n", mpi_comm_global_h.myid,
        // pi_freq_q_Mu_Nu.size());
        mpi_comm_global_h.barrier();
        map<Vector3_Order<double>, complex<double>> global_cRPA_q;
        for (auto q_weight : irk_weight)
        {
            MPI_Reduce(&cRPA_q[q_weight.first], &global_cRPA_q[q_weight.first], 1,
                       MPI_DOUBLE_COMPLEX, MPI_SUM, 0, mpi_comm_global_h.comm);
        }

        for (auto &q_crpa : global_cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
        complex<double> gather_tot_RPA_energy(0.0, 0.0);
        MPI_Reduce(&tot_RPA_energy, &gather_tot_RPA_energy, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0,
                   mpi_comm_global_h.comm);
        corr.value = gather_tot_RPA_energy;
    }
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_MP2_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    corr.etype = CorrEnergy::type::MP2;
    return corr;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    lib_printf("Begin compute_Pi_q , pid:  %d\n", mpi_comm_global_h.myid);
    for (auto const &freq_qJQchi0 : chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second)
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = atom_mu[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = atom_mu[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (int I = 0; I != natom; I++)
                    {
                        // const size_t I = I_p.first;
                        const size_t I_mu = atom_mu[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q) pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             lib_printf("  |process  %d, pi atpair: %d, %d
            //             \n",mpi_comm_global_h.myid,Ip.first,Jp.first);
        }
    }

    // ofstream fp;
    // std::stringstream ss;
    // ss<<"out_pi_rank_"<<mpi_comm_global_h.myid<<".txt";
    // fp.open(ss.str());
    for (auto &freq_p : chi0.get_chi0_q())
    {
        const double freq = freq_p.first;
        const auto chi0_freq = freq_p.second;
        for (auto &k_pair : chi0_freq)
        {
            Vector3_Order<double> ik_vec = k_pair.first;
            auto chi0_freq_k = k_pair.second;
            for (auto &J_p : chi0_freq_k)
            {
                const size_t J = J_p.first;
                for (auto &Q_p : J_p.second)
                {
                    const size_t Q = Q_p.first;
                    auto &chi0_mat = Q_p.second;
                    for (int I = 0; I != natom; I++)
                    {
                        // const size_t I = I_p.first;
                        // printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", mpi_comm_global_h.myid, I,
                        // J, Q);
                        //   cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"
                        //   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);
                        if (I <= J)
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n",
                            //     mpi_comm_global_h.myid, I, J, Q,1);
                            //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) +=
                                (*Vq.at(I).at(J).at(ik_vec)) * chi0_mat;
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            // {
                            //     std:stringstream sm;
                            //     complex<double> trace_pi;
                            //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                            //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " <<
                            //     ik_vec << "  trace_pi:  " << trace_pi << endl;
                            //     print_complex_matrix_file(sm.str().c_str(),
                            //     (*Vq.at(I).at(J).at(ik_vec)),fp,false);
                            //     print_complex_matrix_file("chi0:", chi0_mat,fp,false);
                            //     print_complex_matrix_file("pi_mat:",
                            //     pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                            // }
                        }
                        else
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n",
                            //     mpi_comm_global_h.myid, I, J, Q,2);
                            //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0, 0) <<
                            //      endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) +=
                                transpose(*Vq.at(J).at(I).at(ik_vec), 1) * chi0_mat;
                        }

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            if (I <= Q)
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d
                                //     \n", mpi_comm_global_h.myid, I, J, Q,3);
                                //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) +=
                                    (*Vq.at(I).at(Q).at(ik_vec)) * chi0_QJ;
                            }
                            else
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d
                                //     \n", mpi_comm_global_h.myid, I, J, Q,4);
                                //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0,
                                //      0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) +=
                                    transpose(*Vq.at(Q).at(I).at(ik_vec), 1) * chi0_QJ;
                            }
                        }
                    }
                }
            }
        }
    }
    // fp.close();
    // print_complex_matrix("
    // first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0));
    /* print_complex_matrix("
     * last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    return pi;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q_MPI(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    lib_printf("Begin compute_Pi_q_MPI , pid:  %d\n", mpi_comm_global_h.myid);
    for (auto const &freq_qJQchi0 : chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second)
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = atom_mu[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = atom_mu[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (int I = 0; I != natom; I++)
                    {
                        // const size_t I = I_p.first;
                        const size_t I_mu = atom_mu[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q) pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             lib_printf("  |process  %d, pi atpair: %d, %d
            //             \n",mpi_comm_global_h.myid,Ip.first,Jp.first);
        }
    }

// ofstream fp;
// std::stringstream ss;
// ss<<"out_pi_rank_"<<mpi_comm_global_h.myid<<".txt";
// fp.open(ss.str());
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before irk_weight, pid: %d\n", mpi_comm_global_h.myid);
#endif
    for (auto &k_pair : irk_weight)
    {
        Vector3_Order<double> ik_vec = k_pair.first;
        for (int I = 0; I != natom; I++)
        {
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before gather_vp_row_q irk_weight, pid: %d\n", mpi_comm_global_h.myid);
#endif
            atom_mapping<ComplexMatrix>::pair_t_old Vq_row = gather_vq_row_q(I, coulmat, ik_vec);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after gather_vp_row_q irk_weight, pid: %d\n", mpi_comm_global_h.myid);
#endif
            for (auto &freq_p : chi0.get_chi0_q())
            {
                const double freq = freq_p.first;
                const auto chi0_freq = freq_p.second;

                auto chi0_freq_k = freq_p.second.at(ik_vec);

                for (auto &J_p : chi0_freq_k)
                {
                    const size_t J = J_p.first;
                    for (auto &Q_p : J_p.second)
                    {
                        const size_t Q = Q_p.first;
                        auto &chi0_mat = Q_p.second;

                        // const size_t I = I_p.first;
                        // printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", mpi_comm_global_h.myid, I,
                        // J, Q);
                        //   cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"
                        //   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);

                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n",
                        //     mpi_comm_global_h.myid, I, J, Q,1);
                        //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                        pi.at(freq).at(ik_vec).at(I).at(Q) += Vq_row.at(I).at(J) * chi0_mat;
                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        // {
                        //     std:stringstream sm;
                        //     complex<double> trace_pi;
                        //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                        //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " << ik_vec
                        //     << "  trace_pi:  " << trace_pi << endl;
                        //     print_complex_matrix_file(sm.str().c_str(),
                        //     Vq_row.at(I).at(J),fp,false); print_complex_matrix_file("chi0:",
                        //     chi0_mat,fp,false); print_complex_matrix_file("pi_mat:",
                        //     pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                        // }

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n",
                            //     mpi_comm_global_h.myid, I, J,Q,3);
                            //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
                        }
                    }
                }
            }
        }
    }
    // fp.close();
    //  print_complex_matrix("
    //  first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0));
    /* print_complex_matrix("
     * last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    lib_printf("End compute_Pi_q_MPI , pid:  %d\n", mpi_comm_global_h.myid);
    return pi;
}

ComplexMatrix compute_Pi_freq_q_row(const Vector3_Order<double> &ik_vec,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &Vq_row,
                                    const int &I)
{
    map<size_t, ComplexMatrix> pi;
    // lib_printf("Begin cal_pi_k , pid:  %d\n", para_mpi.get_myid());
    auto I_mu = atom_mu[I];
    for (int J = 0; J != natom; J++) pi[J].create(I_mu, atom_mu[J]);

    omp_lock_t pi_lock;
    omp_init_lock(&pi_lock);
#pragma omp parallel for schedule(dynamic)
    for (int iap = 0; iap != local_atpair.size(); iap++)
    {
        const size_t J = local_atpair[iap].first;
        const size_t Q = local_atpair[iap].second;
        auto &chi0_mat = chi0_freq_q.at(J).at(Q);
        auto tmp_pi_mat = Vq_row.at(I).at(J) * chi0_mat;
        ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
        auto tmp_pi_mat2 = Vq_row.at(I).at(Q) * chi0_QJ;
        omp_set_lock(&pi_lock);
        pi.at(Q) += tmp_pi_mat;
        if (J != Q)
        {
            pi.at(J) += tmp_pi_mat2;
        }
        omp_unset_lock(&pi_lock);
    }
    omp_destroy_lock(&pi_lock);
    // for (auto &J_p : chi0_freq_q)
    // {
    //     const size_t J = J_p.first;
    //     for (auto &Q_p : J_p.second)
    //     {
    //         const size_t Q = Q_p.first;
    //         auto &chi0_mat = Q_p.second;
    //         pi.at(Q) += Vq_row.at(I).at(J) * chi0_mat;
    //         if (J != Q)
    //         {
    //             ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
    //             pi.at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
    //         }
    //     }
    // }
    // Pi_rowT
    // ComplexMatrix pi_row(N_all_mu,atom_mu[I]);
    // complex<double> *pi_row_ptr=pi_row.c;
    // for(auto &Jp:pi)
    // {
    //     auto J=Jp.first;
    //     auto J_mu=atom_mu[J];
    //     const auto length=sizeof(complex<double>)* I_mu *J_mu;
    //     memcpy(pi_row_ptr, pi.at(J).c,length);
    //     pi_row_ptr+=I_mu *J_mu;
    // }
    ComplexMatrix pi_row(atom_mu[I], N_all_mu);
    for (int i = 0; i != pi_row.nr; i++)
        for (int J = 0; J != natom; J++)
            for (int j = 0; j != atom_mu[J]; j++)
                pi_row(i, atom_mu_part_range[J] + j) = pi.at(J)(i, j);
    return pi_row;
}

ComplexMatrix compute_Pi_freq_q_row_ri(const Vector3_Order<double> &ik_vec,
                                       const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q,
                                       const atpair_k_cplx_mat_t &Vq_loc, const int &I,
                                       const Vector3_Order<double> &q)
{
    map<size_t, ComplexMatrix> pi;
    // lib_printf("Begin cal_pi_k , pid:  %d\n", mpi_comm_global_h.myid);
    auto I_mu = atom_mu[I];
    for (int J = 0; J != natom; J++) pi[J].create(I_mu, atom_mu[J]);

    omp_lock_t pi_lock;
    omp_init_lock(&pi_lock);
#pragma omp parallel for schedule(dynamic)
    for (int iap = 0; iap != local_atpair.size(); iap++)
    {
        const size_t J = local_atpair[iap].first;
        const size_t Q = local_atpair[iap].second;
        auto &chi0_mat = chi0_freq_q.at(J).at(Q);
        // printf("| IN cal Pi process %d, I: %d  J: %d  Q: %d\n",mpi_comm_global_h.myid, I,J,Q );
        auto tmp_pi_mat = *Vq_loc.at(I).at(J).at(q) * chi0_mat;
        ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
        auto tmp_pi_mat2 = *Vq_loc.at(I).at(Q).at(q) * chi0_QJ;
        omp_set_lock(&pi_lock);
        pi.at(Q) += tmp_pi_mat;
        if (J != Q)
        {
            pi.at(J) += tmp_pi_mat2;
        }
        omp_unset_lock(&pi_lock);
    }
    omp_destroy_lock(&pi_lock);
    // for (auto &J_p : chi0_freq_q)
    // {
    //     const size_t J = J_p.first;
    //     for (auto &Q_p : J_p.second)
    //     {
    //         const size_t Q = Q_p.first;
    //         auto &chi0_mat = Q_p.second;
    //         pi.at(Q) += Vq_row.at(I).at(J) * chi0_mat;
    //         if (J != Q)
    //         {
    //             ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
    //             pi.at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
    //         }
    //     }
    // }
    // Pi_rowT
    // ComplexMatrix pi_row(N_all_mu,atom_mu[I]);
    // complex<double> *pi_row_ptr=pi_row.c;
    // for(auto &Jp:pi)
    // {
    //     auto J=Jp.first;
    //     auto J_mu=atom_mu[J];
    //     const auto length=sizeof(complex<double>)* I_mu *J_mu;
    //     memcpy(pi_row_ptr, pi.at(J).c,length);
    //     pi_row_ptr+=I_mu *J_mu;
    // }
    ComplexMatrix pi_row(atom_mu[I], N_all_mu);
    for (int i = 0; i != pi_row.nr; i++)
        for (int J = 0; J != natom; J++)
            for (int j = 0; j != atom_mu[J]; j++)
                pi_row(i, atom_mu_part_range[J] + j) = pi.at(J)(i, j);
    return pi_row;
}

atom_mapping<ComplexMatrix>::pair_t_old gather_vq_row_q(const int &I,
                                                        const atpair_k_cplx_mat_t &coulmat,
                                                        const Vector3_Order<double> &ik_vec)
{
    auto I_mu = atom_mu[I];
    atom_mapping<ComplexMatrix>::pair_t_old Vq_row;
    for (int J_tmp = 0; J_tmp != natom; J_tmp++)
    {
        auto J_mu = atom_mu[J_tmp];
        ComplexMatrix loc_vq(atom_mu[I], atom_mu[J_tmp]);
        Vq_row[I][J_tmp].create(atom_mu[I], atom_mu[J_tmp]);
        // const auto length=sizeof(complex<double>)* I_mu *J_mu;
        // complex<double> *loc_vq_ptr=loc_vq.c;
        if (I <= J_tmp)
        {
            if (Vq.count(I))
                if (Vq.at(I).count(J_tmp)) loc_vq = *Vq.at(I).at(J_tmp).at(ik_vec);
        }
        else
        {
            if (Vq.count(J_tmp))
                if (Vq.at(J_tmp).count(I)) loc_vq = transpose(*Vq.at(J_tmp).at(I).at(ik_vec), 1);
        }
        mpi_comm_global_h.allreduce_ComplexMatrix(loc_vq, Vq_row[I][J_tmp]);
    }
    return Vq_row;
}

map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
compute_Wc_freq_q(Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps,
                  atpair_k_cplx_mat_t &coulmat_wc,
                  const vector<std::complex<double>> &epsmac_LF_imagfreq)
{
    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_freq_q;
    const int range_all = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    if (mpi_comm_global_h.myid == 0)
    {
        cout << "Calculating Wc using LAPACK" << endl;
    }

    mpi_comm_global_h.barrier();
    // use q-points as the outmost loop, so that square root of Coulomb will not be recalculated at
    // each frequency point
    vector<Vector3_Order<double>> qpts;
    for (const auto &qMuNuchi : chi0.get_chi0_q().at(chi0.tfg.get_freq_nodes()[0]))
        qpts.push_back(qMuNuchi.first);

    for (const auto &q : qpts)
    {
        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        char fn[80];

        ComplexMatrix Vq_all(range_all, range_all);
        for (const auto &Mu_NuqVq : coulmat_eps)
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for (auto &Nu_qVq : Mu_NuqVq.second)
            {
                auto Nu = Nu_qVq.first;
                if (0 == Nu_qVq.second.count(q)) continue;
                auto n_nu = atom_mu[Nu];
                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    for (int i_nu = 0; i_nu != n_nu; i_nu++)
                    {
                        Vq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                            (*Nu_qVq.second.at(q))(i_mu, i_nu);
                        Vq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                            conj((*Nu_qVq.second.at(q))(i_mu, i_nu));
                    }
            }
        }
        if (Params::debug)
        {
            sprintf(fn, "Vq_all_q_%d.mtx", iq);
            print_complex_matrix_mm(Vq_all, Params::output_dir + "/" + fn, 1e-15);
        }
        auto sqrtVq_all = power_hemat(Vq_all, 0.5, true, false, Params::sqrt_coulomb_threshold);
        // Vq_all is now eigenvectors of the original Coulomb matrix
        const auto &Vq_eigen = Vq_all;
        if (Params::debug)
        {
            sprintf(fn, "sqrtVq_all_q_%d.mtx", iq);
            print_complex_matrix_mm(sqrtVq_all, Params::output_dir + "/" + fn, 1e-15);
            // sprintf(fn, "rotated_sqrtVq_all_q_%d.mtx", iq);
            // print_complex_matrix_mm(Vq_all * sqrtVq_all * transpose(Vq_all, true), fn, 1e-15);
            // print_complex_matrix_mm(transpose(Vq_all, true) * sqrtVq_all * Vq_all, fn, 1e-15);
            sprintf(fn, "Vqeigenvec_q_%d.mtx", iq);
            print_complex_matrix_mm(Vq_eigen, Params::output_dir + "/" + fn, 1e-15);
        }

        // truncated (cutoff) Coulomb
        ComplexMatrix Vqcut_all(range_all, range_all);
        for (auto &Mu_NuqVq : coulmat_wc)
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for (auto &Nu_qVq : Mu_NuqVq.second)
            {
                auto Nu = Nu_qVq.first;
                if (0 == Nu_qVq.second.count(q)) continue;
                auto n_nu = atom_mu[Nu];
                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    for (int i_nu = 0; i_nu != n_nu; i_nu++)
                    {
                        Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                            (*Nu_qVq.second.at(q))(i_mu, i_nu);
                        Vqcut_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                            conj((*Nu_qVq.second.at(q))(i_mu, i_nu));
                    }
            }
        }
        auto sqrtVqcut_all =
            power_hemat(Vqcut_all, 0.5, false, true, Params::sqrt_coulomb_threshold);
        // sprintf(fn, "sqrtVqcut_all_q_%d.mtx", iq);
        // print_complex_matrix_mm(sqrtVqcut_all, fn, 1e-15);
        sprintf(fn, "Vqcut_all_filtered_q_%d.mtx", iq);
        // print_complex_matrix_mm(Vqcut_all, fn, 1e-15);
        // save the filtered truncated Coulomb back to the atom mapping object
        // TODO: revise the necessity
        for (auto &Mu_NuqVq : coulmat_wc)
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for (auto &Nu_qVq : Mu_NuqVq.second)
            {
                auto Nu = Nu_qVq.first;
                if (0 == Nu_qVq.second.count(q)) continue;
                auto n_nu = atom_mu[Nu];
                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    for (int i_nu = 0; i_nu != n_nu; i_nu++)
                        (*Nu_qVq.second.at(q))(i_mu, i_nu) =
                            Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu);
            }
        }

        ComplexMatrix chi0fq_all(range_all, range_all);
        for (const auto &freq_qMuNuchi : chi0.get_chi0_q())
        {
            auto freq = freq_qMuNuchi.first;
            auto ifreq = chi0.tfg.get_freq_index(freq);
            auto MuNuchi = freq_qMuNuchi.second.at(q);
            for (const auto &Mu_Nuchi : MuNuchi)
            {
                auto Mu = Mu_Nuchi.first;
                auto n_mu = atom_mu[Mu];
                for (auto &Nu_chi : Mu_Nuchi.second)
                {
                    auto Nu = Nu_chi.first;
                    auto n_nu = atom_mu[Nu];
                    for (int i_mu = 0; i_mu != n_mu; i_mu++)
                        for (int i_nu = 0; i_nu != n_nu; i_nu++)
                        {
                            chi0fq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                                Nu_chi.second(i_mu, i_nu);
                            chi0fq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                                conj(Nu_chi.second(i_mu, i_nu));
                        }
                }
            }
            sprintf(fn, "chi0fq_all_q_%d_freq_%d.mtx", iq, ifreq);
            print_complex_matrix_mm(chi0fq_all, Params::output_dir + "/" + fn, 1e-15);

            ComplexMatrix identity(range_all, range_all);
            identity.set_as_identity_matrix();
            auto eps_fq = sqrtVq_all * chi0fq_all * sqrtVq_all;
            eps_fq = transpose(Vq_eigen, true) * eps_fq * Vq_eigen;
            if (!epsmac_LF_imagfreq.empty() && is_gamma_point(q))
            {
                // rotate to Coulomb-diagonal basis
                // lib_printf("Largest off-diagonal = %f\n", eps_fq.get_max_abs_offdiag());
                // print_matrix("rotated eps_fq: ", eps_fq.real());
                // replacing the element corresponding to largest Coulomb eigenvalue with dielectric
                // function
                lib_printf("%22.12f %22.12f %22.12f %22.12f\n", freq, eps_fq(0, 0).real(),
                           eps_fq(eps_fq.nr - 1, eps_fq.nc - 1).real(),
                           epsmac_LF_imagfreq[ifreq].real());
                // eps_fq(eps_fq.nr - 1, eps_fq.nc - 1) = epsmac_LF_imagfreq[ifreq];
                eps_fq(0, 0) = 1.0 - epsmac_LF_imagfreq[ifreq];
            }
            if (Params::debug)
            {
                sprintf(fn, "rotated_vsxvs_q_%d_freq_%d.mtx", iq, ifreq);
                print_complex_matrix_mm(eps_fq, Params::output_dir + "/" + fn, 1e-10);
            }
            // rotate back to ABF
            eps_fq = Vq_eigen * eps_fq * transpose(Vq_eigen, true);
            eps_fq = identity - eps_fq;
            if (Params::debug)
            {
                sprintf(fn, "eps_q_%d_freq_%d.mtx", iq, ifreq);
                print_complex_matrix_mm(eps_fq, Params::output_dir + "/" + fn, 1e-10);
            }

            // invert the epsilon matrix
            power_hemat_onsite(eps_fq, -1);
            auto wc_all = sqrtVqcut_all * (eps_fq - identity) * sqrtVqcut_all;
            // sprintf(fn, "inveps_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(eps_fq, fn, 1e-15);
            // sprintf(fn, "wc_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(wc_all, fn, 1e-15);

            // save result to the atom mapping object
            for (auto &Mu_Nuchi : MuNuchi)
            {
                auto Mu = Mu_Nuchi.first;
                auto n_mu = atom_mu[Mu];
                for (auto &Nu_chi : Mu_Nuchi.second)
                {
                    auto Nu = Nu_chi.first;
                    auto n_nu = atom_mu[Nu];
                    shared_ptr<ComplexMatrix> wc_ptr = make_shared<ComplexMatrix>();
                    wc_ptr->create(n_mu, n_nu);
                    for (int i_mu = 0; i_mu != n_mu; i_mu++)
                        for (int i_nu = 0; i_nu != n_nu; i_nu++)
                        {
                            (*wc_ptr)(i_mu, i_nu) =
                                wc_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu);
                        }
                    Wc_freq_q[freq][Mu][Nu][q] =
                        matrix_m<complex<double>>(n_mu, n_nu, wc_ptr->c, MAJOR::ROW, MAJOR::ROW);
                }
            }
        }
    }

    return Wc_freq_q;
}

// Done: converge compute_Wc_freq_q_blacs and compute_Wc_freq_q_blacs_wing
map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
compute_Wc_freq_q_blacs(Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps,
                        atpair_k_cplx_mat_t &coulmat_wc,
                        const vector<std::complex<double>> &epsmac_LF_imagfreq)
{
    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_freq_q;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    if (mpi_comm_global_h.myid == 0)
    {
        cout << "Calculating Wc using ScaLAPACK" << endl;
    }
    mpi_comm_global_h.barrier();

    Profiler::start("compute_Wc_freq_q_blacs_init");
    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    // Use a square blocksize instead max block, otherwise heev and inversion will complain about
    // illegal parameter Maximal blocksize ensure that atom indices related to the rows/columns of a
    // local matrix is minimized.
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    // This, however, is not optimal for matrix operations, and may lead to segment fault during
    // MPI operations with parallel linear algebra subroutine. Thus we define an optimal blocksize
    Array_Desc desc_nabf_nabf_opt(blacs_ctxt_global_h);
    const int nb_opt = min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);
    // obtain the indices of atom-pair block necessary to build 2D block of a Hermitian/symmetric
    // matrix
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    const bool use_abacus_symmetry_dense_chi0_collect =
        Params::use_abacus_gw_symmetry && LIBRPA::abacus_symmetry_ctx.available
        && LIBRPA::abacus_symmetry_ctx.has_abf_shell_layout();
    std::vector<int> abf_atom_offsets;
    {
        const auto atom_nabf_vec = LIBRPA::atomic_basis_abf.get_atom_nbs();
        abf_atom_offsets.resize(atom_nabf_vec.size() + 1, 0);
        for (std::size_t atom = 0; atom < atom_nabf_vec.size(); ++atom)
        {
            abf_atom_offsets[atom + 1] =
                abf_atom_offsets[atom] + static_cast<int>(atom_nabf_vec[atom]);
        }
    }
    // temp_block is used to collect data from IJ-pair data structure with comm_map2_first
    auto temp_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    // Below are the working arrays for matrix operations
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);

    const double mem_blocks = (chi0_block.size() + coul_block.size() + coul_eigen_block.size() +
                               coul_chi0_block.size() + coulwc_block.size()) *
                              16.0e-6;
    ofs_myid << get_timestamp()
             << " Memory consumption of task-local blocks for screened Coulomb [MB]: " << mem_blocks
             << endl;

    const auto atpair_local = dispatch_upper_trangular_tasks(
        natom, blacs_ctxt_global_h.myid, blacs_ctxt_global_h.nprows, blacs_ctxt_global_h.npcols,
        blacs_ctxt_global_h.myprow, blacs_ctxt_global_h.mypcol);
#ifdef LIBRPA_DEBUG
    ofs_myid << get_timestamp() << " atpair_local " << atpair_local << endl;
    ofs_myid << get_timestamp() << " s0_s1 " << s0_s1 << endl;
#endif

    // IJ pair of Wc to be returned
    pair<set<int>, set<int>> Iset_Jset_Wc;
    for (const auto &ap : atpair_local)
    {
        Iset_Jset_Wc.first.insert(ap.first);
        Iset_Jset_Wc.second.insert(ap.second);
    }

    // Prepare local basis indices for 2D->IJ map
    int I, iI;
    map<int, vector<int>> map_lor_v;
    map<int, vector<int>> map_loc_v;
    for (int i_lo = 0; i_lo != desc_nabf_nabf.m_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_r(i_lo);
        LIBRPA::atomic_basis_abf.get_local_index(i_glo, I, iI);
        map_lor_v[I].push_back(iI);
    }
    for (int i_lo = 0; i_lo != desc_nabf_nabf.n_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_c(i_lo);
        LIBRPA::atomic_basis_abf.get_local_index(i_glo, I, iI);
        map_loc_v[I].push_back(iI);
    }

    vector<Vector3_Order<double>> qpts;
    for (const auto &q_weight : irk_weight) qpts.push_back(q_weight.first);

    vec<double> eigenvalues(n_abf);
    Profiler::cease("compute_Wc_freq_q_blacs_init");
    LIBRPA::utils::lib_printf_root("Time for Wc initialization (seconds, Wall/CPU): %f %f\n",
                                   Profiler::get_wall_time_last("compute_Wc_freq_q_blacs_init"),
                                   Profiler::get_cpu_time_last("compute_Wc_freq_q_blacs_init"));

    Profiler::start("compute_Wc_freq_q_work");
#ifdef LIBRPA_USE_LIBRI
    for (const auto &q : qpts)
    {
        const int iq = std::distance(qpts.cbegin(), std::find(qpts.cbegin(), qpts.cend(), q));
        const int iq_in_k =
            std::distance(klist.cbegin(), std::find(klist.cbegin(), klist.cend(), q));
        // q-point in fractional coordinates
        const auto &qf = kfrac_list[iq_in_k];
        LIBRPA::utils::lib_printf_root("Computing Wc(q), %d / %d, q=(%f, %f, %f)\n", iq + 1,
                                       qpts.size(), qf.x, qf.y, qf.z);
        coul_block.zero_out();
        coulwc_block.zero_out();
        // lib_printf("coul_block\n%s", str(coul_block).c_str());

        // q-array for LibRI object
        std::array<double, 3> qa = {q.x, q.y, q.z};

        // collect the block elements of truncated coulomb matrices first
        // as we reuse coul_eigen_block to reduce memory usage
        Profiler::start("epsilon_prepare_coulwc_sqrt", "Prepare sqrt of truncated Coulomb");
        {
            size_t n_singular_coulwc;
            // LibRI tensor for communication, release once done
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                couleps_libri;
            Profiler::start("epsilon_prepare_coulwc_sqrt_1", "Setup libRI object");
            for (const auto &Mu_Nu : atpair_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_wc.count(Mu) == 0 || coulmat_wc.at(Mu).count(Nu) == 0 ||
                    coulmat_wc.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat_wc.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            Profiler::stop("epsilon_prepare_coulwc_sqrt_1");

            Profiler::start("epsilon_prepare_coulwc_sqrt_2", "libRI Communicate");
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_2");

            Profiler::start("epsilon_prepare_coulwc_sqrt_3", "Collect 2D-block from IJ");
            // for (const auto &IJ: set_IJ_nabf_nabf)
            // {
            //     const auto &I = IJ.first;
            //     const auto &J = IJ.second;
            //     collect_block_from_IJ_storage_syhe(
            //         coulwc_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
            //         IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            // }
            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_ctxt_global_h.ictxt);
            std::ostringstream vqcut_debug_name;
            vqcut_debug_name << std::fixed << std::setprecision(10)
                             << "Vqcut_block_qx_" << q.x << "_qy_" << q.y << "_qz_" << q.z
                             << ".mtx";
            dump_blacs_debug_matrix(
                vqcut_debug_name.str(), coulwc_block, desc_nabf_nabf_opt);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_3");
            Profiler::start("epsilon_prepare_coulwc_sqrt_4", "Perform square root");
            power_hemat_blacs(coulwc_block, desc_nabf_nabf_opt, coul_eigen_block,
                              desc_nabf_nabf_opt, n_singular_coulwc, eigenvalues.c, 0.5,
                              Params::sqrt_coulomb_threshold);
            Profiler::stop("epsilon_prepare_coulwc_sqrt_4");
        }
        Profiler::stop("epsilon_prepare_coulwc_sqrt");
        LIBRPA::utils::lib_printf_root(
            "Time to prepare sqrt root of Coulomb for Wc(q) (seconds, Wall/CPU): %f %f\n",
            Profiler::get_wall_time_last("epsilon_prepare_coulwc_sqrt"),
            Profiler::get_cpu_time_last("epsilon_prepare_coulwc_sqrt"));
        ofs_myid << get_timestamp() << " Done coulwc sqrt" << endl;

        Profiler::start("epsilon_prepare_couleps_sqrt", "Prepare sqrt of bare Coulomb");
        // collect the block elements of coulomb matrices
        {
            // LibRI tensor for communication, release once done
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                couleps_libri;
            ofs_myid << get_timestamp() << " Start build couleps_libri" << endl;
            for (const auto &Mu_Nu : atpair_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_eps.count(Mu) == 0 || coulmat_eps.at(Mu).count(Nu) == 0 ||
                    coulmat_eps.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat_eps.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            ofs_myid << get_timestamp() << " Done build couleps_libri" << endl;
            // ofs_myid << "Couleps_libri" << endl << couleps_libri;
            // if (couleps_libri.size() == 0)
            //     throw std::logic_error("data at q-point not found in coulmat_eps");

            // perform communication
            ofs_myid << get_timestamp() << " Start collect couleps_libri, targets" << endl;
#ifdef LIBRPA_DEBUG
            ofs_myid << set_IJ_nabf_nabf << endl;
            ofs_myid << "Extended blocks" << endl;
            ofs_myid << "atom 1: " << s0_s1.first << endl;
            ofs_myid << "atom 2: " << s0_s1.second << endl;
#endif
            // ofs_myid << "Owned blocks\n";
            // print_keys(ofs_myid, couleps_libri);
            // mpi_comm_global_h.barrier();
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            ofs_myid << get_timestamp() << " Done collect couleps_libri, collected blocks" << endl;

            ofs_myid << get_timestamp() << " Start construct couleps 2D block" << endl;
            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_ctxt_global_h.ictxt);
            std::ostringstream vqeps_debug_name;
            vqeps_debug_name << std::fixed << std::setprecision(10)
                             << "Vqeps_block_qx_" << q.x << "_qy_" << q.y << "_qz_" << q.z
                             << ".mtx";
            dump_blacs_debug_matrix(
                vqeps_debug_name.str(), coul_block, desc_nabf_nabf_opt);
            ofs_myid << get_timestamp() << " Done construct couleps 2D block" << endl;
        }
        // char fn[100];
        // sprintf(fn, "couleps_iq_%d.mtx", iq);
        // print_matrix_mm_file_parallel(fn, coul_block, desc_nabf_nabf);
        // ofs_myid << str(coul_block);
        // lib_printf("coul_block\n%s", str(coul_block).c_str());

        size_t n_singular;
        ofs_myid << get_timestamp() << " Start power hemat couleps\n";
        matrix_m<std::complex<double>> sqrtveig_blacs;
        if (is_gamma_point(q))
        {
            // choice of power_hemat_blacs_real/power_hemat_blacs_desc
            // leads to sub-meV difference
            sqrtveig_blacs = power_hemat_blacs_real(
                coul_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt, n_singular,
                eigenvalues.c, 0.5, Params::sqrt_coulomb_threshold);
            if (Params::replace_w_head && Params::option_dielect_func == 3)
            {
                df_headwing.wing_mu_to_lambda(sqrtveig_blacs, desc_nabf_nabf_opt);
            }
        }
        else
        {
            sqrtveig_blacs = power_hemat_blacs(coul_block, desc_nabf_nabf_opt, coul_eigen_block,
                                               desc_nabf_nabf_opt, n_singular, eigenvalues.c, 0.5,
                                               Params::sqrt_coulomb_threshold);
        }
        ofs_myid << get_timestamp() << " Done power hemat couleps\n";
        // lib_printf("nabf %d nsingu %lu\n", n_abf, n_singular);
        // release sqrtv when the q-point is not Gamma, or macroscopic dielectric constant at
        // imaginary frequency is not prepared
        if (epsmac_LF_imagfreq.empty() || !is_gamma_point(q)) sqrtveig_blacs.clear();
        const size_t n_nonsingular = n_abf - n_singular;
        Profiler::stop("epsilon_prepare_couleps_sqrt");
        LIBRPA::utils::lib_printf_root(
            "Time to prepare sqrt root of Coulomb for Epsilon(q) (seconds, Wall/CPU): %f %f\n",
            Profiler::get_wall_time_last("epsilon_prepare_couleps_sqrt"),
            Profiler::get_cpu_time_last("epsilon_prepare_couleps_sqrt"));
        ofs_myid << get_timestamp() << " Done couleps sqrt\n";
        std::flush(ofs_myid);

        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            Profiler::start("epsilon_wc_work_q_omega");
            Profiler::start("epsilon_prepare_chi0_2d", "Prepare Chi0 2D block");
            chi0_block.zero_out();
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                       RI::Tensor<complex<double>>>>
                    chi0_libri;
                std::size_t chi0_local_block_count = 0;
                std::ostringstream chi0_local_block_keys;
                ComplexMatrix chi0_dense_local;
                if (use_abacus_symmetry_dense_chi0_collect)
                {
                    chi0_dense_local.create(n_abf, n_abf, true);
                }
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                const bool has_local_chi0_q =
                    chi0.get_chi0_q().count(freq) > 0 && chi0.get_chi0_q().at(freq).count(q) > 0;
                if (use_abacus_symmetry_dense_chi0_collect)
                {
                    const atom_mapping<ComplexMatrix>::pair_t_old empty_blocks;
                    const auto& chi0_blocks_ibz =
                        has_local_chi0_q ? chi0.get_chi0_q().at(freq).at(q) : empty_blocks;
                    if (Params::debug)
                    {
                        ofs_myid << get_timestamp()
                                 << " chi0 dense restore call site for q=(" << q.x << ", "
                                 << q.y << ", " << q.z << "), ifreq=" << ifreq
                                 << ", has_local_chi0_q=" << has_local_chi0_q
                                 << ", local_input_block_count="
                                 << collect_abacus_atom_pairs(chi0_blocks_ibz).size() << endl;
                    }
                    chi0_wq =
                        symmetrize_abacus_chi0_ibz_blocks_if_needed(chi0_blocks_ibz, q);
                    if (Params::debug && is_gamma_point(q) && ifreq == 0)
                    {
                        std::fprintf(stderr,
                                     "[rank %d] after sym_restore q=(%.10f, %.10f, %.10f) ifreq=%d empty=%d\n",
                                     mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq,
                                     chi0_wq.empty() ? 1 : 0);
                        std::fflush(stderr);
                    }
                }
                else if (has_local_chi0_q)
                {
                    chi0_wq = symmetrize_abacus_chi0_ibz_blocks_if_needed(
                        chi0.get_chi0_q().at(freq).at(q), q);
                }
                if (!chi0_wq.empty())
                {
                    if (Params::debug && is_gamma_point(q) && ifreq == 0)
                    {
                        std::fprintf(stderr,
                                     "[rank %d] enter chi0_wq loop q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                     mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                        std::fflush(stderr);
                    }
                    for (const auto &M_Nchi : chi0_wq)
                    {
                        const auto &M = M_Nchi.first;
                        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                        for (const auto &N_chi : M_Nchi.second)
                        {
                            const auto &N = N_chi.first;
                            const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                            const auto &chi = N_chi.second;
                            if (Params::debug && is_gamma_point(q) && ifreq == 0)
                            {
                                std::fprintf(stderr,
                                             "[rank %d] chi0_wq block M=%d N=%d dims=%d x %d ifreq=%d\n",
                                             mpi_comm_global_h.myid, M, N, chi.nr, chi.nc, ifreq);
                                std::fflush(stderr);
                            }
                            // The symmetry-restored chi0 blocks must match the active ABF layout
                            // exactly. Otherwise RI::Tensor would be constructed with a shape
                            // that disagrees with the payload, which can corrupt the subsequent
                            // MPI redistribution and hang inside comm_map2_first().
                            if (chi.nr != n_mu || chi.nc != n_nu)
                            {
                                std::ostringstream oss;
                                oss << "Symmetry-restored chi0 block dimension mismatch at q=("
                                    << q.x << ", " << q.y << ", " << q.z << "), freq index "
                                    << ifreq << ", atom pair (" << M << ", " << N << "): block="
                                    << chi.nr << "x" << chi.nc << ", expected=" << n_mu << "x"
                                    << n_nu;
                                throw std::runtime_error(oss.str());
                            }
                            std::valarray<complex<double>> chi_va(chi.c, chi.size);
                            auto pchi = std::make_shared<std::valarray<complex<double>>>();
                            *pchi = chi_va;
                            chi0_libri[M][{N, qa}] =
                                RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                            if (use_abacus_symmetry_dense_chi0_collect)
                            {
                                const int row_offset = abf_atom_offsets[static_cast<std::size_t>(M)];
                                const int col_offset = abf_atom_offsets[static_cast<std::size_t>(N)];
                                for (int row = 0; row < chi.nr; ++row)
                                {
                                    for (int col = 0; col < chi.nc; ++col)
                                    {
                                        const auto value = chi(row, col);
                                        chi0_dense_local(row_offset + row, col_offset + col) = value;
                                        if (M != N)
                                        {
                                            chi0_dense_local(col_offset + col, row_offset + row) =
                                                std::conj(value);
                                        }
                                    }
                                }
                            }
                            ++chi0_local_block_count;
                            if (chi0_local_block_count == 1)
                            {
                                chi0_local_block_keys << "(" << M << "," << N << ")";
                            }
                            else
                            {
                                chi0_local_block_keys << " (" << M << "," << N << ")";
                            }
                        }
                    }
                    if (Params::debug && is_gamma_point(q) && ifreq == 0)
                    {
                        std::fprintf(stderr,
                                     "[rank %d] finished chi0_wq loop q=(%.10f, %.10f, %.10f) ifreq=%d has_local=%d\n",
                                     mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq,
                                     has_local_chi0_q ? 1 : 0);
                        std::fflush(stderr);
                    }
                    // Release only the original local chi0(q, iω) block. Under the ABACUS
                    // symmetry restore path, ranks without a local source block still receive a
                    // non-empty restored chi0_wq; attempting to free the absent original entry on
                    // those ranks throws std::out_of_range inside chi0's map storage.
                    if (has_local_chi0_q)
                    {
                        if (Params::debug && is_gamma_point(q) && ifreq == 0)
                        {
                            std::fprintf(stderr,
                                         "[rank %d] before chi0.free_chi0_q q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                         mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                            std::fflush(stderr);
                        }
                        chi0.free_chi0_q(freq, q);
                        if (Params::debug && is_gamma_point(q) && ifreq == 0)
                        {
                            std::fprintf(stderr,
                                         "[rank %d] after chi0.free_chi0_q q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                         mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                            std::fflush(stderr);
                        }
                    }
                }
                if (Params::use_abacus_gw_symmetry && Params::debug)
                {
                    ofs_myid << get_timestamp() << " chi0_libri local blocks before comm_map2_first: "
                             << chi0_local_block_count << " at q=(" << q.x << ", " << q.y << ", "
                             << q.z << "), ifreq=" << ifreq;
                    if (chi0_local_block_count > 0)
                    {
                        ofs_myid << ", keys=" << chi0_local_block_keys.str();
                    }
                    ofs_myid << endl;
                }
                // ofs_myid << "chi0_libri" << endl << chi0_libri;
                if (use_abacus_symmetry_dense_chi0_collect)
                {
                    Profiler::start("epsilon_prepare_chi0_2d_collect_block");
                    // The symmetry restore path already replicated the full upper-triangular
                    // symmetrized chi0(q, iω) onto every rank. Reusing another world reduction
                    // here is unnecessary and can deadlock when some ranks own no original
                    // atom-pair blocks. Consume the identical dense matrix locally instead.
                    if (Params::debug)
                    {
                        ofs_myid << get_timestamp()
                                 << " chi0 dense local fill start for q=(" << q.x << ", " << q.y
                                 << ", " << q.z << "), ifreq=" << ifreq
                                 << ", n_abf=" << n_abf
                                 << ", local_dims=" << chi0_dense_local.nr << "x"
                                 << chi0_dense_local.nc
                                 << ", local_size=" << chi0_dense_local.size
                                 << ", local_max_abs=" << chi0_dense_local.get_max_abs()
                                 << endl;
                    }
                    for (int ilo = 0; ilo != desc_nabf_nabf.m_loc(); ++ilo)
                    {
                        const int i_gl = desc_nabf_nabf.indx_l2g_r(ilo);
                        for (int jlo = 0; jlo != desc_nabf_nabf.n_loc(); ++jlo)
                        {
                            const int j_gl = desc_nabf_nabf.indx_l2g_c(jlo);
                            temp_block(ilo, jlo) = chi0_dense_local(i_gl, j_gl);
                        }
                    }
                    if (Params::debug)
                    {
                        ofs_myid << get_timestamp()
                                 << " chi0 dense local fill finished for q=(" << q.x << ", "
                                 << q.y << ", " << q.z << "), ifreq=" << ifreq << endl;
                    }
                }
                else
                {
                    Profiler::start("epsilon_prepare_chi0_2d_comm_map2");
                    const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                        mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                    Profiler::stop("epsilon_prepare_chi0_2d_comm_map2");
                    if (Params::use_abacus_gw_symmetry && Params::debug)
                    {
                        ofs_myid << get_timestamp()
                                 << " chi0_libri redistribution finished for q=(" << q.x << ", "
                                 << q.y << ", " << q.z << "), ifreq=" << ifreq << endl;
                    }
                    // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                    // for (const auto &IJ: set_IJ_nabf_nabf)
                    // {
                    //     const auto &I = IJ.first;
                    //     const auto &J = IJ.second;
                    //     collect_block_from_IJ_storage_syhe(
                    //         chi0_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
                    //         IJ.second, true, CONE, IJq_chi0.at(I).at({J, qa}).ptr(), MAJOR::ROW);
                    // }
                    Profiler::start("epsilon_prepare_chi0_2d_collect_block");
                    collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf,
                                                     LIBRPA::atomic_basis_abf, qa, true, CONE,
                                                     IJq_chi0, MAJOR::ROW);
                }
                if (Params::debug)
                {
                    ofs_myid << get_timestamp() << " chi0 block redistribution before pgemr2d for q=("
                             << q.x << ", " << q.y << ", " << q.z << "), ifreq=" << ifreq
                             << endl;
                }
                if (Params::debug && is_gamma_point(q) && ifreq == 0)
                {
                    std::fprintf(stderr,
                                 "[rank %d] before chi0 pgemr2d q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                 mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                    std::fflush(stderr);
                }
                ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1,
                                              desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                              desc_nabf_nabf_opt.desc, blacs_ctxt_global_h.ictxt);
                if (Params::debug)
                {
                    ofs_myid << get_timestamp() << " chi0 block redistribution after pgemr2d for q=("
                             << q.x << ", " << q.y << ", " << q.z << "), ifreq=" << ifreq
                             << endl;
                }
                if (Params::debug && is_gamma_point(q) && ifreq == 0)
                {
                    std::fprintf(stderr,
                                 "[rank %d] after chi0 pgemr2d q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                 mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                    std::fflush(stderr);
                }
                std::ostringstream chi0_debug_name;
                chi0_debug_name << std::fixed << std::setprecision(10)
                                << "chi0_block_qx_" << q.x << "_qy_" << q.y << "_qz_" << q.z
                                << "_freq_" << ifreq << ".mtx";
                if (Params::debug)
                {
                    ofs_myid << get_timestamp() << " chi0 block dump start for q=(" << q.x << ", "
                             << q.y << ", " << q.z << "), ifreq=" << ifreq << endl;
                }
                if (Params::debug && is_gamma_point(q) && ifreq == 0)
                {
                    std::fprintf(stderr,
                                 "[rank %d] before chi0 dump q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                 mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                    std::fflush(stderr);
                }
                dump_blacs_debug_matrix(
                    chi0_debug_name.str(), chi0_block, desc_nabf_nabf_opt);
                if (Params::debug)
                {
                    ofs_myid << get_timestamp() << " chi0 block dump finished for q=(" << q.x
                             << ", " << q.y << ", " << q.z << "), ifreq=" << ifreq << endl;
                }
                if (Params::debug && is_gamma_point(q) && ifreq == 0)
                {
                    std::fprintf(stderr,
                                 "[rank %d] after chi0 dump q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                                 mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                    std::fflush(stderr);
                }
                Profiler::stop("epsilon_prepare_chi0_2d_collect_block");
                // sprintf(fn, "chi_ifreq_%d_iq_%d.mtx", ifreq, iq);
                // print_matrix_mm_file_parallel(fn, chi0_block, desc_nabf_nabf);
            }
            Profiler::stop("epsilon_prepare_chi0_2d");
            if (Params::debug && is_gamma_point(q) && ifreq == 0)
            {
                std::fprintf(stderr,
                             "[rank %d] before epsilon_compute_eps q=(%.10f, %.10f, %.10f) ifreq=%d\n",
                             mpi_comm_global_h.myid, q.x, q.y, q.z, ifreq);
                std::fflush(stderr);
            }

            Profiler::start("epsilon_compute_eps", "Compute dielectric matrix");

            // for Gamma point, overwrite the head term
            if (epsmac_LF_imagfreq.size() > 0 && is_gamma_point(q))
            {
                ofs_myid << get_timestamp() << " Entering dielectric matrix head overwrite" << endl;
                // rotate to Coulomb-eigenvector basis
                // descending order
                ScalapackConnector::pgemm_f(
                    'N', 'N', n_abf, n_nonsingular, n_abf, 1.0, chi0_block.ptr(), 1, 1,
                    desc_nabf_nabf_opt.desc, sqrtveig_blacs.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                    0.0, coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                ScalapackConnector::pgemm_f('C', 'N', n_nonsingular, n_nonsingular, n_abf, 1.0,
                                            sqrtveig_blacs.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            0.0, chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

                if (Params::option_dielect_func == 3)
                {
                    chi0_block *= -1.0;
                    for (int i = 0; i != n_nonsingular; i++)
                    {
                        const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                        if (ilo < 0) continue;
                        const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                        if (jlo < 0) continue;
                        chi0_block(ilo, jlo) += 1.0;
                    }
                    ofs_myid << get_timestamp() << "Perform the head & wing element overwrite"
                             << endl;
                    df_headwing.rewrite_eps(chi0_block, ifreq, desc_nabf_nabf_opt);

                    if (Params::debug)
                    {
                        const int ilo = desc_nabf_nabf_opt.indx_g2l_r(0);
                        const int jlo = desc_nabf_nabf_opt.indx_g2l_c(0);
                        if (ilo >= 0 && jlo >= 0)
                            std::cout << "inv_eps(0,0)=" << chi0_block(ilo, jlo) << endl;
                    }
                }
                else
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(0);
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(0);
                    if (ilo >= 0 && jlo >= 0)
                    {
                        ofs_myid << get_timestamp() << "Perform the head element overwrite" << endl;
                        chi0_block(ilo, jlo) = 1.0 - epsmac_LF_imagfreq[ifreq];
                    }
                }
                // rotate back to ABF
                // descending order
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_nonsingular, n_nonsingular, 1.0,
                                            coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc, 0.0,
                                            coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, n_nonsingular, 1.0,
                                            coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            0.0, chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                if (Params::option_dielect_func != 3)
                {
                    // now chi0_block is actually v1/2 chi v1/2
                    chi0_block *= -1.0;
                    for (int i = 0; i != n_abf; i++)
                    {
                        const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                        if (ilo < 0) continue;
                        const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                        if (jlo < 0) continue;
                        chi0_block(ilo, jlo) += 1.0;
                    }
                    // now chi0_block is actually the dielectric matrix
                    // perform inversion
                    Profiler::start("epsilon_invert_eps", "Invert dielectric matrix");
                    invert_scalapack(chi0_block, desc_nabf_nabf_opt);
                }
                // subtract 1 from diagonal
                for (int i = 0; i != n_abf; i++)
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                    if (ilo < 0) continue;
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                    if (jlo < 0) continue;
                    chi0_block(ilo, jlo) -= 1.0;
                }
            }
            else
            {
                Profiler::start("epsilon_compute_eps_pgemm_1");
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coul_block.ptr(), 1,
                                            1, desc_nabf_nabf_opt.desc, chi0_block.ptr(), 1, 1,
                                            desc_nabf_nabf_opt.desc, 0.0, coul_chi0_block.ptr(), 1,
                                            1, desc_nabf_nabf_opt.desc);
                Profiler::cease("epsilon_compute_eps_pgemm_1");
                Profiler::start("epsilon_compute_eps_pgemm_2");
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0,
                                            coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                            coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc, 0.0,
                                            chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                Profiler::cease("epsilon_compute_eps_pgemm_2");
                // now chi0_block is actually v1/2 chi v1/2
                chi0_block *= -1.0;
                for (int i = 0; i != n_abf; i++)
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                    if (ilo < 0) continue;
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                    if (jlo < 0) continue;
                    chi0_block(ilo, jlo) += 1.0;
                }
                Profiler::stop("epsilon_compute_eps");
                // now chi0_block is actually the dielectric matrix
                // perform inversion
                Profiler::start("epsilon_invert_eps", "Invert dielectric matrix");
                invert_scalapack(chi0_block, desc_nabf_nabf_opt);
                // subtract 1 from diagonal
                for (int i = 0; i != n_abf; i++)
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                    if (ilo < 0) continue;
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                    if (jlo < 0) continue;
                    chi0_block(ilo, jlo) -= 1.0;
                }
                Profiler::stop("epsilon_invert_eps");
            }
            std::ostringstream epsinv_debug_name;
            epsinv_debug_name << std::fixed << std::setprecision(10)
                              << "epsinv_minus_identity_qx_" << q.x
                              << "_qy_" << q.y
                              << "_qz_" << q.z
                              << "_freq_" << ifreq << ".mtx";
            dump_blacs_debug_matrix(
                epsinv_debug_name.str(), chi0_block, desc_nabf_nabf_opt, 1e-10);
            // debug for Coulomb, epsilon^{-1} - 1 = -0.75
            // for (int i = 0; i != n_abf; i++)
            // {
            //     for (int j = 0; j != n_abf; j++)
            //     {
            //         const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
            //         if (ilo < 0) continue;
            //         const int jlo = desc_nabf_nabf_opt.indx_g2l_c(j);
            //         if (jlo < 0) continue;
            //         if (i == j)
            //             chi0_block(ilo, jlo) = -0.75;
            //         else
            //             chi0_block(ilo, jlo) = 0.0;
            //     }
            // }
            // debug for unfold shrink Wc
            // for (int i = 0; i != n_abf; i++)
            //{
            //     const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
            //     if (ilo < 0) continue;
            //     for (int j = 0; j != n_abf; j++)
            //     {
            //         const int jlo = desc_nabf_nabf_opt.indx_g2l_c(j);
            //         if (jlo < 0) continue;
            //         if (i == j)
            //             chi0_block(ilo, jlo) = 1.0;
            //         else
            //             chi0_block(ilo, jlo) = 0.0;
            //     }
            // }
            // debug end

            Profiler::start("epsilon_multiply_coulwc", "Multiply truncated Coulomb");
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coulwc_block.ptr(), 1,
                                        1, desc_nabf_nabf_opt.desc, chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf_opt.desc, 0.0, coul_chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf_opt.desc);
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coul_chi0_block.ptr(),
                                        1, 1, desc_nabf_nabf_opt.desc, coulwc_block.ptr(), 1, 1,
                                        desc_nabf_nabf_opt.desc, 0.0, chi0_block.ptr(), 1, 1,
                                        desc_nabf_nabf_opt.desc);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, chi0_block.ptr(), 1, 1,
                                          desc_nabf_nabf_opt.desc, temp_block.ptr(), 1, 1,
                                          desc_nabf_nabf.desc, blacs_ctxt_global_h.ictxt);
            Profiler::stop("epsilon_multiply_coulwc");
            // lib_printf("chi0_block\n%s", str(chi0_block).c_str());
            // now chi0_block is the screened Coulomb interaction Wc (i.e. W-V)

            Profiler::start("epsilon_convert_wc_2d_to_ij", "Convert Wc, 2D -> IJ");
            Profiler::start("epsilon_convert_wc_map_block", "Initialize Wc atom-pair map");
            map<int, map<int, matrix_m<complex<double>>>> Wc_MNmap;
            // map_block_to_IJ_storage(Wc_MNmap, LIBRPA::atomic_basis_abf,
            //                         LIBRPA::atomic_basis_abf, chi0_block,
            //                         desc_nabf_nabf, MAJOR::ROW);
            map_block_to_IJ_storage_new(Wc_MNmap, LIBRPA::atomic_basis_abf, map_lor_v, map_loc_v,
                                        temp_block, desc_nabf_nabf, MAJOR::ROW);
            Profiler::stop("epsilon_convert_wc_map_block");

            Profiler::start("epsilon_convert_wc_communicate", "Communicate");
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                       RI::Tensor<complex<double>>>>
                    Wc_libri;
                Profiler::start("epsilon_convert_wc_communicate_1");
                for (const auto &M_NWc : Wc_MNmap)
                {
                    const auto &M = M_NWc.first;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    for (const auto &N_Wc : M_NWc.second)
                    {
                        const auto &N = N_Wc.first;
                        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                        const auto &Wc = N_Wc.second;
                        // std::valarray<complex<double>> Wc_va(Wc.ptr(), Wc.size());
                        // auto pWc = std::make_shared<std::valarray<complex<double>>>();
                        // *pWc = Wc_va;
                        /*if (iq == 10 && ifreq == 10)
                        {
                            char fn[100];
                            sprintf(fn, "Wc_M_%zu_N_%zu.dat", M, N);
                            print_matrix_mm_file(Wc, Params::output_dir + "/" + fn);
                        }*/
                        Wc_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, Wc.sptr());
                    }
                }
                Profiler::stop("epsilon_convert_wc_communicate_1");
                Profiler::start("epsilon_convert_wc_communicate_2");
                // main timing
                // cout << Wc_libri;
                const auto IJq_Wc = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                    mpi_comm_global_h.comm, Wc_libri, Iset_Jset_Wc.first, Iset_Jset_Wc.second);
                Profiler::stop("epsilon_convert_wc_communicate_2");
                Profiler::start("epsilon_convert_wc_communicate_3");
                // parse collected to
                for (const auto &MN : atpair_local)
                {
                    const auto &M = MN.first;
                    const auto &N = MN.second;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                    // Use row major for later usage in LibRI
                    Wc_freq_q[freq][M][N][q] = matrix_m<complex<double>>(
                        n_mu, n_nu, IJq_Wc.at(M).at({N, qa}).data, MAJOR::ROW);
                }
                Profiler::stop("epsilon_convert_wc_communicate_3");
                // for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                //     for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                //     {
                //     }
            }
            Profiler::stop("epsilon_convert_wc_communicate");
            Profiler::stop("epsilon_convert_wc_2d_to_ij");
            Profiler::cease("epsilon_wc_work_q_omega");
            LIBRPA::utils::lib_printf_root(
                "Time for Wc(i_q=%d, i_omega=%d) (seconds, Wall/CPU): %f %f\n", iq + 1, ifreq + 1,
                Profiler::get_wall_time_last("epsilon_wc_work_q_omega"),
                Profiler::get_cpu_time_last("epsilon_wc_work_q_omega"));
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    Profiler::cease("compute_Wc_freq_q_work");
    LIBRPA::utils::lib_printf_root("Time for Wc computation (seconds, Wall/CPU): %f %f\n",
                                   Profiler::get_wall_time_last("compute_Wc_freq_q_work"),
                                   Profiler::get_cpu_time_last("compute_Wc_freq_q_work"));

    return Wc_freq_q;
}

LIBRPA::optics::ImagAxisOpticsData compute_absorption_imag_axis_blacs(
    Chi0& chi0,
    const atpair_k_cplx_mat_t& coulmat_eps)
{
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);

    Array_Desc desc_nabf_nabf_opt(blacs_ctxt_global_h);
    const int nb_opt = min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);

    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    const bool use_abacus_symmetry_dense_chi0_collect =
        Params::use_abacus_gw_symmetry && LIBRPA::abacus_symmetry_ctx.available
        && LIBRPA::abacus_symmetry_ctx.has_abf_shell_layout();

    std::vector<int> abf_atom_offsets;
    {
        const auto atom_nabf_vec = LIBRPA::atomic_basis_abf.get_atom_nbs();
        abf_atom_offsets.resize(atom_nabf_vec.size() + 1, 0);
        for (std::size_t atom = 0; atom < atom_nabf_vec.size(); ++atom)
        {
            abf_atom_offsets[atom + 1] =
                abf_atom_offsets[atom] + static_cast<int>(atom_nabf_vec[atom]);
        }
    }

    auto temp_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);

    const auto atpair_local = dispatch_upper_trangular_tasks(
        natom, blacs_ctxt_global_h.myid, blacs_ctxt_global_h.nprows, blacs_ctxt_global_h.npcols,
        blacs_ctxt_global_h.myprow, blacs_ctxt_global_h.mypcol);

    std::vector<Vector3_Order<double>> qpts;
    for (const auto& q_weight : irk_weight)
    {
        qpts.push_back(q_weight.first);
    }

    const auto gamma_q_iter = std::find_if(qpts.cbegin(), qpts.cend(), [](const auto& q) {
        return is_gamma_point(q);
    });
    if (gamma_q_iter == qpts.cend())
    {
        throw std::runtime_error("compute_absorption_imag_axis_blacs failed to locate Gamma q");
    }
    const auto gamma_q = *gamma_q_iter;
    const std::array<double, 3> qa{gamma_q.x, gamma_q.y, gamma_q.z};

    temp_block.zero_out();
    coul_block.zero_out();
    {
        std::map<int,
                 std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
            couleps_libri;
        for (const auto& Mu_Nu : atpair_local)
        {
            const auto Mu = Mu_Nu.first;
            const auto Nu = Mu_Nu.second;
            if (coulmat_eps.count(Mu) == 0 || coulmat_eps.at(Mu).count(Nu) == 0
                || coulmat_eps.at(Mu).at(Nu).count(gamma_q) == 0)
            {
                continue;
            }
            const auto& Vq = coulmat_eps.at(Mu).at(Nu).at(gamma_q);
            const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
            const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
            std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
            auto pvq = std::make_shared<std::valarray<complex<double>>>();
            *pvq = Vq_va;
            couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
        }

        const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
            mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
        collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                         qa, true, CONE, IJq_coul, MAJOR::ROW);
        ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                      coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                      blacs_ctxt_global_h.ictxt);
    }

    vec<double> eigenvalues(n_abf);
    size_t n_singular = 0;
    auto sqrtveig_blacs = power_hemat_blacs_real(coul_block, desc_nabf_nabf_opt, coul_eigen_block,
                                                 desc_nabf_nabf_opt, n_singular, eigenvalues.c,
                                                 0.5, Params::sqrt_coulomb_threshold);
    const size_t n_nonsingular = n_abf - n_singular;
    if (n_nonsingular == 0)
    {
        throw std::runtime_error("Gamma Coulomb square root has no non-singular subspace");
    }

    df_headwing.wing_mu_to_lambda(sqrtveig_blacs, desc_nabf_nabf_opt);

    std::vector<matrix_m<std::complex<double>>> epsinv_mats;
    std::vector<matrix_m<std::complex<double>>> epsm_tensors;
    std::vector<matrix_m<std::complex<double>>> head_tensors;
    epsinv_mats.reserve(chi0.tfg.get_freq_nodes().size());
    epsm_tensors.reserve(chi0.tfg.get_freq_nodes().size());
    head_tensors.reserve(chi0.tfg.get_freq_nodes().size());

    for (const auto& freq : chi0.tfg.get_freq_nodes())
    {
        const int ifreq = chi0.tfg.get_freq_index(freq);
        temp_block.zero_out();
        chi0_block.zero_out();
        coul_chi0_block.zero_out();

        std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
            chi0_libri;
        ComplexMatrix chi0_dense_local;
        atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;

        const bool has_local_chi0_q = chi0.get_chi0_q().count(freq) > 0
                                      && find_matching_abacus_qpoint(chi0.get_chi0_q().at(freq),
                                                                     gamma_q)
                                             != chi0.get_chi0_q().at(freq).end();
        if (use_abacus_symmetry_dense_chi0_collect)
        {
            chi0_dense_local.create(n_abf, n_abf, true);
            const atom_mapping<ComplexMatrix>::pair_t_old empty_blocks;
            const atom_mapping<ComplexMatrix>::pair_t_old* chi0_blocks_ibz_ptr = &empty_blocks;
            if (chi0.get_chi0_q().count(freq) > 0)
            {
                const auto& chi0_blocks_all_q = chi0.get_chi0_q().at(freq);
                const auto chi0_q_iter = has_local_chi0_q
                                             ? find_matching_abacus_qpoint(chi0_blocks_all_q,
                                                                           gamma_q)
                                             : chi0_blocks_all_q.end();
                if (has_local_chi0_q)
                {
                    chi0_blocks_ibz_ptr = &chi0_q_iter->second;
                }
            }
            chi0_wq =
                symmetrize_abacus_chi0_ibz_blocks_if_needed(*chi0_blocks_ibz_ptr, gamma_q);
        }
        else if (has_local_chi0_q)
        {
            const auto& chi0_blocks_all_q = chi0.get_chi0_q().at(freq);
            const auto chi0_q_iter = find_matching_abacus_qpoint(chi0_blocks_all_q, gamma_q);
            chi0_wq = symmetrize_abacus_chi0_ibz_blocks_if_needed(chi0_q_iter->second, gamma_q);
        }

        if (!chi0_wq.empty())
        {
            for (const auto& M_Nchi : chi0_wq)
            {
                const auto& M = M_Nchi.first;
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                for (const auto& N_chi : M_Nchi.second)
                {
                    const auto& N = N_chi.first;
                    const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                    const auto& chi = N_chi.second;
                    if (chi.nr != n_mu || chi.nc != n_nu)
                    {
                        throw std::runtime_error(
                            "Gamma chi0 block dimension mismatch during absorption build");
                    }

                    std::valarray<complex<double>> chi_va(chi.c, chi.size);
                    auto pchi = std::make_shared<std::valarray<complex<double>>>();
                    *pchi = chi_va;
                    chi0_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);

                    if (use_abacus_symmetry_dense_chi0_collect)
                    {
                        const int row_offset = abf_atom_offsets[static_cast<std::size_t>(M)];
                        const int col_offset = abf_atom_offsets[static_cast<std::size_t>(N)];
                        for (int row = 0; row < chi.nr; ++row)
                        {
                            for (int col = 0; col < chi.nc; ++col)
                            {
                                const auto value = chi(row, col);
                                chi0_dense_local(row_offset + row, col_offset + col) = value;
                                if (M != N)
                                {
                                    chi0_dense_local(col_offset + col, row_offset + row) =
                                        std::conj(value);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (has_local_chi0_q)
        {
            chi0.free_chi0_q(freq, gamma_q);
        }

        if (use_abacus_symmetry_dense_chi0_collect)
        {
            for (int ilo = 0; ilo != desc_nabf_nabf.m_loc(); ++ilo)
            {
                const int i_gl = desc_nabf_nabf.indx_l2g_r(ilo);
                for (int jlo = 0; jlo != desc_nabf_nabf.n_loc(); ++jlo)
                {
                    const int j_gl = desc_nabf_nabf.indx_l2g_c(jlo);
                    temp_block(ilo, jlo) = chi0_dense_local(i_gl, j_gl);
                }
            }
        }
        else
        {
            const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                mpi_comm_global_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_chi0, MAJOR::ROW);
        }

        ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                      chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                      blacs_ctxt_global_h.ictxt);

        ScalapackConnector::pgemm_f('N', 'N', n_abf, n_nonsingular, n_abf, 1.0, chi0_block.ptr(),
                                    1, 1, desc_nabf_nabf_opt.desc, sqrtveig_blacs.ptr(), 1, 1,
                                    desc_nabf_nabf_opt.desc, 0.0, coul_chi0_block.ptr(), 1, 1,
                                    desc_nabf_nabf_opt.desc);
        ScalapackConnector::pgemm_f('C', 'N', n_nonsingular, n_nonsingular, n_abf, 1.0,
                                    sqrtveig_blacs.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                    coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc, 0.0,
                                    chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

        chi0_block *= -1.0;
        for (std::size_t i = 0; i != n_nonsingular; ++i)
        {
            const int ilo = desc_nabf_nabf_opt.indx_g2l_r(static_cast<int>(i));
            const int jlo = desc_nabf_nabf_opt.indx_g2l_c(static_cast<int>(i));
            if (ilo >= 0 && jlo >= 0)
            {
                chi0_block(ilo, jlo) += 1.0;
            }
        }

        auto desc_body = df_headwing.get_body_inv(chi0_block, desc_nabf_nabf_opt);
        df_headwing.cal_eps(ifreq, desc_nabf_nabf_opt, desc_body);

        std::complex<double> epsinv00 = 0.0;
        const auto& epsinv_avg = df_headwing.get_inverse_dielectric_average();
        const int ilo = desc_nabf_nabf_opt.indx_g2l_r(0);
        const int jlo = desc_nabf_nabf_opt.indx_g2l_c(0);
        if (ilo >= 0 && jlo >= 0)
        {
            epsinv00 = epsinv_avg(ilo, jlo);
        }
        MPI_Allreduce(MPI_IN_PLACE, &epsinv00, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                      mpi_comm_global_h.comm);

        matrix_m<std::complex<double>> epsinv00_mat(1, 1, MAJOR::COL);
        epsinv00_mat(0, 0) = epsinv00;
        epsinv_mats.push_back(epsinv00_mat);
        epsm_tensors.push_back(df_headwing.get_lind_tensor().copy());
        head_tensors.push_back(df_headwing.get_head_tensors().at(ifreq).copy());
        df_headwing.clear_eps_workspace();
    }

    return LIBRPA::optics::build_imag_axis_data(chi0.tfg.get_freq_nodes(), epsinv_mats,
                                                epsm_tensors, head_tensors);
}

map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
FT_Wc_freq_q(
    const map<double,
              atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist)
{
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;

    map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_freq_R;
    const int ngrids = tfg.get_n_grids();
    if (Params::debug)
    {
        if (mpi_comm_global_h.is_root()) lib_printf("Converting Wc q,w -> R,w\n");
        mpi_comm_global_h.barrier();
    }
    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &freq_MuNuqWc : Wc_freq_q)
    {
        for (const auto &Mu_NuqWc : freq_MuNuqWc.second)
        {
            const auto Mu = Mu_NuqWc.first;
            for (const auto &Nu_qWc : Mu_NuqWc.second)
            {
                const auto Nu = Nu_qWc.first;
                atpairs_unique.insert({Mu, Nu});
                for (const auto &q_Wc : Nu_qWc.second)
                {
                    assert(q_Wc.second.major() == major_Wc);
                }
            }
        }
    }

    vector<pair<pair<int, Vector3_Order<int>>, pair<atom_t, atom_t>>> ifreqR_atpair_all;
    // allocate space before hand
    for (auto R : Rlist)
    {
        for (int ifreq = 0; ifreq != ngrids; ifreq++)
        {
            auto freq = tfg.get_freq_nodes()[ifreq];
            for (auto atpair_unique : atpairs_unique)
            {
                const auto Mu = atpair_unique.first;
                const int n_mu = atom_mu[Mu];
                const auto Nu = atpair_unique.second;
                const int n_nu = atom_mu[Nu];
                Wc_freq_R[freq][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
                ifreqR_atpair_all.push_back({{ifreq, R}, atpair_unique});
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (auto ifreqR_atpair : ifreqR_atpair_all)
    {
        const auto ifreq = ifreqR_atpair.first.first;
        const auto freq = tfg.get_freq_nodes()[ifreq];
        const auto R = ifreqR_atpair.first.second;
        const auto Mu = ifreqR_atpair.second.first;
        const auto Nu = ifreqR_atpair.second.second;
        const int n_mu = atom_mu[Mu];
        const int n_nu = atom_mu[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WfR_temp(n_mu, n_nu, major_Wc);

        if (Wc_freq_q.count(freq) == 0) continue;
        if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
        if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;

        for (auto &Wc_q : Wc_freq_q.at(freq).at(Mu).at(Nu))
        {
            const auto q = Wc_q.first;
            const auto &Wc = Wc_q.second;
            for (auto q_bz : map_irk_ks[q])
            {
                const double ang = -q_bz * (R * latvec) * TWO_PI;
                const complex<double> weight =
                    complex<double>(cos(ang), sin(ang)) / double(n_k_points);
                if (q == q_bz)
                    WfR_temp += Wc * weight;
                else
                    WfR_temp += conj(Wc) * weight;
            }
        }
        // omp_set_lock(&lock_Wc);
        Wc_freq_R[freq][Mu][Nu][R] += WfR_temp;
        // omp_unset_lock(&lock_Wc);
    }

    if (mpi_comm_global_h.is_root())
    {
        lib_printf("Done converting Wc(q,w) -> Wc(R,w)\n");
    }
    mpi_comm_global_h.barrier();

    return Wc_freq_R;
}

map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
CT_FT_Wc_freq_q(
    const map<double,
              atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist)
{
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;
    const bool use_abacus_irreducible_wr =
        can_use_abacus_irreducible_sector_wr_restore(atom_mu);
    const bool use_abacus_full_q_restore =
        can_restore_abacus_abf_full_qspace_operator(atom_mu) && !use_abacus_irreducible_wr;

    map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_tau_R;
    std::map<double, abf_rspace_complex_block_map_t> Wc_freq_R;
    std::map<double, abf_qspace_complex_block_map_t> Wc_freq_q_full;
    if (!tfg.has_time_grids()) throw logic_error("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();

    if (use_abacus_irreducible_wr)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry accumulates irreducible-sector `W(R,w)` directly from IBZ q-stars\n");
        const abf_qspace_complex_block_map_t empty_blocks;
        for (int ifreq = 0; ifreq < ngrids; ++ifreq)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto& freq_blocks =
                (Wc_freq_q.count(freq) > 0) ? Wc_freq_q.at(freq) : empty_blocks;
            Wc_freq_R[freq] =
                accumulate_abacus_full_wr_from_ibz_q(freq_blocks, n_k_points, Rlist, atom_mu);
        }
    }
    else if (use_abacus_full_q_restore)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry restores the full ABF q-star before `CT_FT_Wc_freq_q`\n");
        const abf_qspace_complex_block_map_t empty_blocks;
        for (int ifreq = 0; ifreq < ngrids; ++ifreq)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto& freq_blocks =
                (Wc_freq_q.count(freq) > 0) ? Wc_freq_q.at(freq) : empty_blocks;
            Wc_freq_q_full[freq] =
                restore_abacus_abf_full_qspace_operator(freq_blocks, atom_mu);
        }
    }

    LIBRPA::utils::lib_printf_root("Converting Wc(q,w) -> W(R,t)\n");
    mpi_comm_global_h.barrier();

    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &freq_MuNuqWc : Wc_freq_q)
    {
        for (const auto &Mu_NuqWc : freq_MuNuqWc.second)
        {
            const auto Mu = Mu_NuqWc.first;
            for (const auto &Nu_qWc : Mu_NuqWc.second)
            {
                const auto Nu = Nu_qWc.first;
                atpairs_unique.insert({Mu, Nu});
                for (const auto &q_Wc : Nu_qWc.second)
                {
                    assert(q_Wc.second.major() == major_Wc);
                }
            }
        }
    }

    vector<pair<pair<int, Vector3_Order<int>>, pair<atom_t, atom_t>>> itauR_atpair_all;
    // allocate space before hand
    for (auto R : Rlist)
    {
        for (int itau = 0; itau != ngrids; itau++)
        {
            auto tau = tfg.get_time_nodes()[itau];
            for (auto atpair_unique : atpairs_unique)
            {
                const auto Mu = atpair_unique.first;
                const int n_mu = atom_mu[Mu];
                const auto Nu = atpair_unique.second;
                const int n_nu = atom_mu[Nu];
                Wc_tau_R[tau][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
                itauR_atpair_all.push_back({{itau, R}, atpair_unique});
            }
        }
    }

    LIBRPA::utils::lib_printf_coll("Task %4d: distributing %d {I, J, R, tau} on %d threads\n",
                                   LIBRPA::envs::myid_global, itauR_atpair_all.size(),
                                   omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto itauR_atpair : itauR_atpair_all)
    {
        const auto itau = itauR_atpair.first.first;
        const auto tau = tfg.get_time_nodes()[itau];
        const auto R = itauR_atpair.first.second;
        const auto Mu = itauR_atpair.second.first;
        const auto Nu = itauR_atpair.second.second;
        const int n_mu = atom_mu[Mu];
        const int n_nu = atom_mu[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WtR_temp(n_mu, n_nu, major_Wc);

        for (int ifreq = 0; ifreq < ngrids; ifreq++)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
            // ofs_myid << "f2t cos eff for freq " << freq << " -> tau " << tau  << ": " << f2t <<
            // "\n";
            if (Wc_freq_q.count(freq) == 0) continue;
            if (use_abacus_irreducible_wr)
            {
                if (Wc_freq_R.count(freq) == 0) continue;
                if (Wc_freq_R.at(freq).count(Mu) == 0) continue;
                if (Wc_freq_R.at(freq).at(Mu).count(Nu) == 0) continue;
                if (Wc_freq_R.at(freq).at(Mu).at(Nu).count(R) == 0) continue;
                WtR_temp += Wc_freq_R.at(freq).at(Mu).at(Nu).at(R) * f2t;
            }
            else if (use_abacus_full_q_restore)
            {
                if (Wc_freq_q_full.count(freq) == 0) continue;
                if (Wc_freq_q_full.at(freq).count(Mu) == 0) continue;
                if (Wc_freq_q_full.at(freq).at(Mu).count(Nu) == 0) continue;
                for (const auto& q_Wc : Wc_freq_q_full.at(freq).at(Mu).at(Nu))
                {
                    const auto& q = q_Wc.first;
                    const auto& Wc = q_Wc.second;
                    const double ang = -q * (R * latvec) * TWO_PI;
                    const complex<double> weight =
                        complex<double>(cos(ang), sin(ang)) * f2t / double(n_k_points);
                    WtR_temp += Wc * weight;
                }
            }
            else
            {
                if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
                if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;

                const auto &Wc_q_all = Wc_freq_q.at(freq).at(Mu).at(Nu);
                for (auto &Wc_q : Wc_q_all)
                {
                    const auto q = Wc_q.first;
                    const auto &Wc = Wc_q.second;
                    for (auto q_bz : map_irk_ks[q])
                    {
                        const double ang = -q_bz * (R * latvec) * TWO_PI;
                        const complex<double> weight =
                            complex<double>(cos(ang), sin(ang)) * f2t / double(n_k_points);
                        if (q == q_bz)
                            WtR_temp += Wc * weight;
                        else
                            WtR_temp += conj(Wc) * weight;
                    }
                }
            }
        }
        // omp_set_lock(&lock_Wc);
        Wc_tau_R[tau][Mu][Nu][R] += WtR_temp;
        // omp_unset_lock(&lock_Wc);
    }

    if (use_abacus_irreducible_wr)
    {
        Wc_freq_R.clear();
    }

    LIBRPA::utils::lib_printf_root("Done converting Wc q,w -> R,t\n");
    mpi_comm_global_h.barrier();

    // myz debug: check the imaginary part of the matrix
    // NOTE: if G(R) is real, is W(R) real as well?
    // if (Params::debug)
    // {
    //     for (const auto & tau_MuNuRWc: Wc_tau_R)
    //     {
    //         char fn[80];
    //         auto tau = tau_MuNuRWc.first;
    //         auto itau = tfg.get_time_index(tau);
    //         for (const auto & Mu_NuRWc: tau_MuNuRWc.second)
    //         {
    //             auto Mu = Mu_NuRWc.first;
    //             // const int n_mu = atom_mu[Mu];
    //             for (const auto & Nu_RWc: Mu_NuRWc.second)
    //             {
    //                 auto Nu = Nu_RWc.first;
    //                 // const int n_nu = atom_mu[Nu];
    //                 for (const auto & R_Wc: Nu_RWc.second)
    //                 {
    //                     auto R = R_Wc.first;
    //                     auto Wc = R_Wc.second;
    //                     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
    //                     auto iR = std::distance(Rlist.cbegin(), iteR);
    //                     sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_itau_%d_id_%d.mtx", Mu, Nu, iR,
    //                     itau, mpi_comm_global_h.myid); print_matrix_mm_file(Wc,
    //                     Params::output_dir + "/" + fn, 1e-10);
    //                 }
    //             }
    //         }
    //     }
    // }
    // end myz debug
    return Wc_tau_R;
}

map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
CT_FT_Wc_q2R_freq2time(
    map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist)
{
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;
    const bool use_abacus_irreducible_wr =
        can_use_abacus_irreducible_sector_wr_restore(atom_mu);
    const bool use_abacus_full_q_restore =
        can_restore_abacus_abf_full_qspace_operator(atom_mu) && !use_abacus_irreducible_wr;

    map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_tau_R, Wc_freq_R;
    std::map<double, abf_qspace_complex_block_map_t> Wc_freq_q_full;
    if (!tfg.has_time_grids()) throw logic_error("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();
    set<pair<atom_t, atom_t>> atpairs_unique;

    LIBRPA::utils::lib_printf_root("Converting Wc(q,w) -> Wc(R,w) -> W(R,t)\n");

    Profiler::start("construct_Wc_lower_half", "Construct Lower Half of Wc(q,w)");
    // NOTE: only upper half of Wc is built now
    //       here we recover the other half before transform to R space using the Hermitian property
    //       and Hermitize the diagonal blocks (due to numerical noise)
    for (int ifreq = 0; ifreq < ngrids; ifreq++)
    {
        const auto freq = tfg.get_freq_nodes()[ifreq];
        auto &Wc = Wc_freq_q.at(freq);
        vector<atom_t> iatoms_row;
        for (const auto &Mu_NuqWc : Wc) iatoms_row.push_back(Mu_NuqWc.first);
        for (auto iatom_row: iatoms_row)
        {
            vector<atom_t> iatoms_col;
            for (const auto &Nu_qWc : Wc.at(iatom_row))
            {
                iatoms_col.push_back(Nu_qWc.first);
            }
            for (auto iatom_col : iatoms_col)
            {
                atpairs_unique.insert({iatom_row, iatom_col});
                atpairs_unique.insert({iatom_col, iatom_row});
                for (const auto &q_Wc : Wc.at(iatom_row).at(iatom_col))
                {
                    assert(q_Wc.second.major() == major_Wc);
                    if(iatom_row != iatom_col)
                        Wc[iatom_col][iatom_row][q_Wc.first] = q_Wc.second.get_transpose(true);
                    else // Hermitize the diagonal blocks
                    {                        
                        auto Wc_mat = q_Wc.second;
                        Wc_mat = (Wc_mat + Wc_mat.get_transpose(true)) * 0.5;
                        Wc[iatom_row][iatom_row][q_Wc.first] = Wc_mat;
                    }
                }
            }
        }
    }
    mpi_comm_global_h.barrier();
    Profiler::stop("construct_Wc_lower_half");

    static bool dumped_input_qspace_freq = false;
    if (!dumped_input_qspace_freq && !Wc_freq_q.empty())
    {
        dump_abacus_abf_qspace_blocks(
            "Wc_input_qspace_ct_ifreq0", Wc_freq_q.begin()->second, 1e-15, true);
        dumped_input_qspace_freq = true;
    }

    if (use_abacus_irreducible_wr)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry accumulates irreducible-sector `W(R,w)` directly from IBZ q-stars\n");
        const abf_qspace_complex_block_map_t empty_blocks;
        for (int ifreq = 0; ifreq < ngrids; ++ifreq)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto& freq_blocks =
                (Wc_freq_q.count(freq) > 0) ? Wc_freq_q.at(freq) : empty_blocks;
            Wc_freq_R[freq] =
                accumulate_abacus_full_wr_from_ibz_q(freq_blocks, n_k_points, Rlist, atom_mu);
        }
    }
    else if (use_abacus_full_q_restore)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry restores the full ABF q-star before `CT_FT_Wc_q2R_freq2time`\n");
        const abf_qspace_complex_block_map_t empty_blocks;
        for (int ifreq = 0; ifreq < ngrids; ++ifreq)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto& freq_blocks =
                (Wc_freq_q.count(freq) > 0) ? Wc_freq_q.at(freq) : empty_blocks;
            Wc_freq_q_full[freq] =
                restore_abacus_abf_full_qspace_operator(freq_blocks, atom_mu);
        }

        static bool dumped_restored_fullq_freq = false;
        if (!dumped_restored_fullq_freq && !Wc_freq_q_full.empty())
        {
            dump_abacus_abf_qspace_blocks("Wc_restored_fullq_ct_ifreq0",
                                          Wc_freq_q_full.begin()->second, 1e-15, true);
            dumped_restored_fullq_freq = true;
        }
    }

    Profiler::start("Wc(q,w) -> Wc(R,w)", "Convert Wc(q,w) -> Wc(R,w)");
    if (!use_abacus_irreducible_wr)
    {
        vector<pair<pair<int, Vector3_Order<int>>, pair<atom_t, atom_t>>> ifreqR_atpair_all;
        // allocate Wc(R,w) before hand
        for (auto R : Rlist)
        {
            for (int ifreq = 0; ifreq != ngrids; ifreq++)
            {
                auto tau = tfg.get_time_nodes()[ifreq];
                auto freq = tfg.get_freq_nodes()[ifreq];
                (void)tau;
                for (auto atpair_unique : atpairs_unique)
                {
                    const auto Mu = atpair_unique.first;
                    const int n_mu = atom_mu[Mu];
                    const auto Nu = atpair_unique.second;
                    const int n_nu = atom_mu[Nu];
                    Wc_freq_R[freq][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
                    ifreqR_atpair_all.push_back({{ifreq, R}, atpair_unique});
                }
            }
        }

        LIBRPA::utils::lib_printf_coll("Task %4d: distributing %d {I, J, R, freq} on %d threads\n",
                                       LIBRPA::envs::myid_global, ifreqR_atpair_all.size(),
                                       omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
        for (auto ifreqR_atpair : ifreqR_atpair_all)
        {
            const auto ifreq = ifreqR_atpair.first.first;
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto R = ifreqR_atpair.first.second;
            const auto Mu = ifreqR_atpair.second.first;
            const auto Nu = ifreqR_atpair.second.second;
            const int n_mu = atom_mu[Mu];
            const int n_nu = atom_mu[Nu];

            matrix_m<complex<double>> WfR_temp(n_mu, n_nu, major_Wc);

            if (Wc_freq_q.count(freq) == 0) continue;
            if (use_abacus_full_q_restore)
            {
                if (Wc_freq_q_full.count(freq) == 0) continue;
                if (Wc_freq_q_full.at(freq).count(Mu) == 0) continue;
                if (Wc_freq_q_full.at(freq).at(Mu).count(Nu) == 0) continue;

                for (const auto& q_Wc : Wc_freq_q_full.at(freq).at(Mu).at(Nu))
                {
                    const auto& q = q_Wc.first;
                    const auto& Wc = q_Wc.second;
                    const double ang = -q * (R * latvec) * TWO_PI;
                    const complex<double> weight =
                        complex<double>(cos(ang), sin(ang)) / double(n_k_points);
                    WfR_temp += Wc * weight;
                }
            }
            else
            {
                if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
                if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;

                for (auto &Wc_q : Wc_freq_q.at(freq).at(Mu).at(Nu))
                {
                    const auto q = Wc_q.first;
                    const auto &Wc = Wc_q.second;
                    for (auto q_bz : map_irk_ks[q])
                    {
                        const double ang = -q_bz * (R * latvec) * TWO_PI;
                        const complex<double> weight =
                            complex<double>(cos(ang), sin(ang)) / double(n_k_points);
                        if (q == q_bz)
                            WfR_temp += Wc * weight;
                        else
                            WfR_temp += conj(Wc) * weight;
                    }
                }
            }
            Wc_freq_R[freq][Mu][Nu][R] += WfR_temp;
        }
    }
    mpi_comm_global_h.barrier();
    // HACK: Free up Wc_freq_q to save memory, especially for large Coulomb matrix case and many
    // minimax grids
    Wc_freq_q.clear();
    LIBRPA::utils::lib_printf_root("Done converting Wc(q,w) -> Wc(R,w)\n");

    Profiler::stop("Wc(q,w) -> Wc(R,w)");

    if (Params::output_Wc_Rf_mat > 0)
    {
        Profiler::start("write_Wc_freq_R", "Export Wc(R,w) to file");
        int write_freq = Params::output_Wc_Rf_mat==1 ? 1 : ngrids;
        for (int ifreq = 0; ifreq != write_freq; ifreq++)
        {
            char fn[80];
            const auto freq = tfg.get_freq_nodes()[ifreq];
            auto& freq_MuNuRWc = Wc_freq_R.at(freq);
            for (const auto &Mu_NuRWc : freq_MuNuRWc)
            {
                auto Mu = Mu_NuRWc.first;
                // const int n_mu = atom_mu[Mu];
                for (const auto &Nu_RWc : Mu_NuRWc.second)
                {
                    auto Nu = Nu_RWc.first;
                    // const int n_nu = atom_mu[Nu];
                    for (const auto &R_Wc : Nu_RWc.second)
                    {
                        auto R = R_Wc.first;
                        auto Wc = R_Wc.second;
                        auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                        auto iR = std::distance(Rlist.cbegin(), iteR);
                        sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_ifreq_%d.mtx", Mu, Nu, iR, ifreq);
                        std::string info = "Wc at iR " + std::to_string(iR) + 
                            " ( " + std::to_string(R.x) + " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                            " ) and ifreq " + std::to_string(ifreq) +
                            " ( " + std::to_string(freq) + " a.u. )";
                        print_matrix_mm_file(Wc, Params::output_dir + "/" + fn, info, 1e-10);
                    }
                }
            }
        }        
        Profiler::stop("write_Wc_freq_R");
    }

    Profiler::start("Wc(R,w) -> Wc(R,t)", "Convert Wc(R,w) -> Wc(R,t)");
    if (mpi_comm_global_h.is_root())
    {
        lib_printf("Start converting Wc(R,w) -> Wc(R,t)\n");
    }
    // allocate Wc(R,t) before hand
    for (auto R : Rlist)
    {
        for (int itau = 0; itau != ngrids; itau++)
        {
            auto tau = tfg.get_time_nodes()[itau];
            (void)tfg.get_freq_nodes()[itau];
            for (auto atpair_unique : atpairs_unique)
            {
                const auto Mu = atpair_unique.first;
                const int n_mu = atom_mu[Mu];
                const auto Nu = atpair_unique.second;
                const int n_nu = atom_mu[Nu];
                Wc_tau_R[tau][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            }
        }
    }

    vector<pair<pair<int, Vector3_Order<int>>, pair<atom_t, atom_t>>> itauR_atpair_all;
    for (auto R : Rlist)
    {
        for (int itau = 0; itau != ngrids; ++itau)
        {
            for (auto atpair_unique : atpairs_unique)
            {
                itauR_atpair_all.push_back({{itau, R}, atpair_unique});
            }
        }
    }

    LIBRPA::utils::lib_printf_coll("Task %4d: distributing %d {I, J, R, tau} on %d threads\n",
        LIBRPA::envs::myid_global, itauR_atpair_all.size(),
        omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto itauR_atpair : itauR_atpair_all)
    {
        const auto itau = itauR_atpair.first.first;
        const auto tau = tfg.get_time_nodes()[itau];
        const auto R = itauR_atpair.first.second;
        const auto Mu = itauR_atpair.second.first;
        const auto Nu = itauR_atpair.second.second;
        const int n_mu = atom_mu[Mu];
        const int n_nu = atom_mu[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WtR_temp(n_mu, n_nu, major_Wc);

        for (int ifreq = 0; ifreq < ngrids; ifreq++)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
            // ofs_myid << "f2t cos eff for freq " << freq << " -> tau " << tau  << ": " << f2t <<
            // "\n";
            if (Wc_freq_R.count(freq) == 0) continue;
            if (Wc_freq_R.at(freq).count(Mu) == 0) continue;
            if (Wc_freq_R.at(freq).at(Mu).count(Nu) == 0) continue;
            if (Wc_freq_R.at(freq).at(Mu).at(Nu).count(R) == 0) continue;
            // cout << "freq: " << freq << "\n";

            const auto &Wc = Wc_freq_R.at(freq).at(Mu).at(Nu).at(R);
            WtR_temp += Wc * f2t;
        }
        // omp_set_lock(&lock_Wc);
        Wc_tau_R[tau][Mu][Nu][R] += WtR_temp;
        // omp_unset_lock(&lock_Wc);
    }
    // NOTE: Wc(R,w) will not be used any more, clean to free up memory
    Wc_freq_R.clear();
    LIBRPA::utils::release_free_mem();
    mpi_comm_global_h.barrier();
    LIBRPA::utils::lib_printf_root("Done converting Wc(R,w) -> Wc(R,t)\n");
    Profiler::stop("Wc(R,w) -> Wc(R,t)");

    return Wc_tau_R;
}

map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
CT_Wc_freq2time_q(
    const map<double,
              atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist,
    const vector<Vector3_Order<double>> &qlist)
{
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;

    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_tau_q;
    if (!tfg.has_time_grids()) throw logic_error("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();

    LIBRPA::utils::lib_printf_root("Converting Wc(q,w) -> W(q,t)\n");
    mpi_comm_global_h.barrier();

    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &freq_MuNuqWc : Wc_freq_q)
    {
        for (const auto &Mu_NuqWc : freq_MuNuqWc.second)
        {
            const auto Mu = Mu_NuqWc.first;
            for (const auto &Nu_qWc : Mu_NuqWc.second)
            {
                const auto Nu = Nu_qWc.first;
                atpairs_unique.insert({Mu, Nu});
                for (const auto &q_Wc : Nu_qWc.second)
                {
                    assert(q_Wc.second.major() == major_Wc);
                }
            }
        }
    }
    vector<pair<int, pair<atom_t, atom_t>>> itau_atpair_all;
    // allocate space before hand

    for (int itau = 0; itau != ngrids; itau++)
    {
        auto tau = tfg.get_time_nodes()[itau];
        for (auto atpair_unique : atpairs_unique)
        {
            const auto Mu = atpair_unique.first;
            const int n_mu = atom_mu_s[Mu];
            const auto Nu = atpair_unique.second;
            const int n_nu = atom_mu_s[Nu];
            for (auto q : qlist)
                Wc_tau_q[tau][Mu][Nu][q] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            itau_atpair_all.push_back({itau, atpair_unique});
        }
    }

    LIBRPA::utils::lib_printf_coll("Task %4d: distributing %d {I, J, R, tau} on %d threads\n",
                                   LIBRPA::envs::myid_global, itau_atpair_all.size(),
                                   omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto itau_atpair : itau_atpair_all)
    {
        const auto itau = itau_atpair.first;
        const auto tau = tfg.get_time_nodes()[itau];
        const auto Mu = itau_atpair.second.first;
        const auto Nu = itau_atpair.second.second;
        const int n_mu = atom_mu_s[Mu];
        const int n_nu = atom_mu_s[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> Wtq_temp(n_mu, n_nu, major_Wc);

        for (int ifreq = 0; ifreq < ngrids; ifreq++)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
            // ofs_myid << "f2t cos eff for freq " << freq << " -> tau " << tau  << ": " << f2t <<
            // "\n";
            if (Wc_freq_q.count(freq) == 0) continue;
            if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
            if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;
            // cout << "freq: " << freq << "\n";

            const auto &Wc_q_all = Wc_freq_q.at(freq).at(Mu).at(Nu);
            for (auto &Wc_q : Wc_q_all)
            {
                const auto q = Wc_q.first;
                const auto &Wc = Wc_q.second;
                const double weight = f2t;
                Wtq_temp = Wc * weight;
                // omp_set_lock(&lock_Wc);
                Wc_tau_q[tau][Mu][Nu][q] += Wtq_temp;
                // omp_unset_lock(&lock_Wc);
            }
        }
    }

    LIBRPA::utils::lib_printf_root("Done converting Wc q,w -> q,t\n");
    mpi_comm_global_h.barrier();

    return Wc_tau_q;
}

/// @brief Wc(q,w) -> Wc(R,w) or Wc(q,t) -> W(R,t)
atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old FT_Wc_q2R(
    const atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old
        &Wc_q,
    const TFGrids &tfg, const int &n_kpoints, const vector<Vector3_Order<int>> &Rlist,
    const bool is_freq, const double debug_tau)
{
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;
    const bool is_rank_local_freq_export =
        is_freq && Params::output_Wc_Rf_mat == 1 && mpi_comm_global_h.nprocs > 1;
    const bool use_abacus_irreducible_wr =
        can_use_abacus_irreducible_sector_wr_restore(atom_mu_l);
    const bool can_use_abacus_full_q_restore =
        can_restore_abacus_abf_full_qspace_operator(atom_mu_l);
    const bool use_abacus_full_q_restore =
        can_use_abacus_full_q_restore && !use_abacus_irreducible_wr && !is_rank_local_freq_export;

    if (can_use_abacus_full_q_restore && !use_abacus_irreducible_wr && is_rank_local_freq_export)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry skips full-q restore for the MPI Wc(R,w=0) export path,\n"
            "because the temporary Wc(q,w=0) container only holds rank-local atom-pair blocks.\n");
    }

    atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old Wc_R;
    if (use_abacus_irreducible_wr)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry accumulates irreducible-sector `W(R)` directly from IBZ q-stars\n");
        Wc_R = accumulate_abacus_full_wr_from_ibz_q(Wc_q, n_kpoints, Rlist, atom_mu_l);
    }
    const auto Wc_q_full =
        use_abacus_full_q_restore ? restore_abacus_abf_full_qspace_operator(Wc_q, atom_mu_l)
                                  : abf_qspace_complex_block_map_t{};
    if (use_abacus_full_q_restore)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry restores the full ABF q-star before `FT_Wc_q2R`\n");
        log_abf_qspace_summary("Wc_q_restored_fullq", debug_tau, Wc_q_full);

        // Dump the reconstructed full-q blocks once so we can compare them against the
        // symmetry-off reference and isolate whether the mismatch starts before the q->R FT.
        static bool dumped_restored_fullq_freq = false;
        static bool dumped_restored_fullq_tau = false;
        if (is_freq)
        {
            if (!dumped_restored_fullq_freq)
            {
                dump_abacus_abf_qspace_blocks("Wc_restored_fullq_ifreq0", Wc_q_full);
                dumped_restored_fullq_freq = true;
            }
        }
        else if (!dumped_restored_fullq_tau)
        {
            dump_abacus_abf_qspace_blocks("Wc_restored_fullq_tau0", Wc_q_full);
            dumped_restored_fullq_tau = true;
        }
    }

    LIBRPA::utils::lib_printf_root("Converting Wc(q) -> W(R)\n");
    mpi_comm_global_h.barrier();

    if (use_abacus_irreducible_wr)
    {
        mpi_comm_global_h.barrier();
        LIBRPA::utils::lib_printf_root("Done converting Wc q -> R\n");

        if (is_freq && Params::output_Wc_Rf_mat==1)
        {
            Profiler::start("write_Wc_freq_R", "Export Wc(R,w) to file");
            int ifreq = 0;
            const auto freq = tfg.get_freq_nodes()[ifreq];
            char fn[80];
            for (const auto &Mu_NuRWc : Wc_R)
            {
                auto Mu = Mu_NuRWc.first;
                for (const auto &Nu_RWc : Mu_NuRWc.second)
                {
                    auto Nu = Nu_RWc.first;
                    for (const auto &R_Wc : Nu_RWc.second)
                    {
                        auto R = R_Wc.first;
                        auto Wc = R_Wc.second;
                        auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                        auto iR = std::distance(Rlist.cbegin(), iteR);
                        sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_ifreq_%d.mtx", Mu, Nu, iR, ifreq);
                        std::string info = "Wc at iR " + std::to_string(iR)
                            + " ( " + std::to_string(R.x) + " " + std::to_string(R.y) + " "
                            + std::to_string(R.z) + " ) and ifreq " + std::to_string(ifreq)
                            + " ( " + std::to_string(freq) + " a.u. )";
                        print_matrix_mm_file(Wc, Params::output_dir + "/" + fn, info, 1e-10);
                    }
                }
            }
            Profiler::stop("write_Wc_freq_R");
        }

        return Wc_R;
    }

    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &MuNuqWc : Wc_q)
    {
        const auto Mu = MuNuqWc.first;
        for (const auto &Nu_qWc : MuNuqWc.second)
        {
            const auto Nu = Nu_qWc.first;
            atpairs_unique.insert({Mu, Nu});
            for (const auto &q_Wc : Nu_qWc.second)
            {
                assert(q_Wc.second.major() == major_Wc);
            }
        }
    }

    vector<pair<Vector3_Order<int>, pair<atom_t, atom_t>>> iR_atpair_all;
    // allocate space before hand
    for (auto R : Rlist)
    {
        for (auto atpair_unique : atpairs_unique)
        {
            const auto Mu = atpair_unique.first;
            const int n_mu = atom_mu_l[Mu];
            const auto Nu = atpair_unique.second;
            const int n_nu = atom_mu_l[Nu];
            Wc_R[Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            iR_atpair_all.push_back({R, atpair_unique});
        }
    }

    LIBRPA::utils::lib_printf_coll("Task %4d: distributing %d {I, J, R} on %d threads\n",
                                   LIBRPA::envs::myid_global, iR_atpair_all.size(),
                                   omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto iR_atpair : iR_atpair_all)
    {
        const auto R = iR_atpair.first;
        const auto Mu = iR_atpair.second.first;
        const auto Nu = iR_atpair.second.second;
        const int n_mu = atom_mu_l[Mu];
        const int n_nu = atom_mu_l[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WR_temp(n_mu, n_nu, major_Wc);

        if (use_abacus_full_q_restore)
        {
            if (Wc_q_full.count(Mu) == 0) continue;
            if (Wc_q_full.at(Mu).count(Nu) == 0) continue;

            const auto& Wc_q_all = Wc_q_full.at(Mu).at(Nu);
            for (const auto& q_Wc : Wc_q_all)
            {
                const auto& q = q_Wc.first;
                const auto& Wc = q_Wc.second;
                const double ang = -q * (R * latvec) * TWO_PI;
                const complex<double> weight =
                    complex<double>(cos(ang), sin(ang)) / double(n_kpoints);
                WR_temp += Wc * weight;
            }
        }
        else
        {
            if (Wc_q.count(Mu) == 0) continue;
            if (Wc_q.at(Mu).count(Nu) == 0) continue;

            const auto &Wc_q_all = Wc_q.at(Mu).at(Nu);
            for (auto &Wc_q : Wc_q_all)
            {
                const auto q = Wc_q.first;
                const auto &Wc = Wc_q.second;
                for (auto q_bz : map_irk_ks[q])
                {
                    const double ang = -q_bz * (R * latvec) * TWO_PI;
                    const complex<double> weight =
                        complex<double>(cos(ang), sin(ang)) / double(n_kpoints);
                    if (q == q_bz)
                        WR_temp += Wc * weight;
                    else
                        WR_temp += conj(Wc) * weight;
                }
            }
        }
        // omp_set_lock(&lock_Wc);
        Wc_R[Mu][Nu][R] += WR_temp;
        // omp_unset_lock(&lock_Wc);
    }
    mpi_comm_global_h.barrier();
    LIBRPA::utils::lib_printf_root("Done converting Wc q -> R\n");

    if (is_freq && Params::output_Wc_Rf_mat==1)
    {
        Profiler::start("write_Wc_freq_R", "Export Wc(R,w) to file");
        int ifreq = 0;
        const auto freq = tfg.get_freq_nodes()[ifreq];
        char fn[80];
        for (const auto &Mu_NuRWc : Wc_R)
        {
            auto Mu = Mu_NuRWc.first;
            // const int n_mu = atom_mu[Mu];
            for (const auto &Nu_RWc : Mu_NuRWc.second)
            {
                auto Nu = Nu_RWc.first;
                // const int n_nu = atom_mu[Nu];
                for (const auto &R_Wc : Nu_RWc.second)
                {
                    auto R = R_Wc.first;
                    auto Wc = R_Wc.second;
                    auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                    auto iR = std::distance(Rlist.cbegin(), iteR);
                    sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_ifreq_%d.mtx", Mu, Nu, iR, ifreq);
                    std::string info = "Wc at iR " + std::to_string(iR) + 
                        " ( " + std::to_string(R.x) + " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                        " ) and ifreq " + std::to_string(ifreq) +
                        " ( " + std::to_string(freq) + " a.u. )";
                    print_matrix_mm_file(Wc, Params::output_dir + "/" + fn, info, 1e-10);
                }
            }
        }
        Profiler::stop("write_Wc_freq_R");        
    }

    return Wc_R;
}

void test_libcomm_for_system(const atpair_k_cplx_mat_t &coulmat)
{
    if (mpi_comm_global_h.myid == 0) lib_printf("test_libcomm_for_system Coulumb\n");
    // lib_printf("Calculating EcRPA with BLACS, pid:  %d\n", mpi_comm_global_h.myid);
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    mpi_comm_global_h.barrier();

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about
    // illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf = LIBRPA::utils::get_necessary_IJ_from_block_2D_sy(
        'U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);

    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);

    vector<Vector3_Order<double>> qpts;
    for (const auto &qMuNuchi : irk_weight) qpts.push_back(qMuNuchi.first);

#ifdef LIBRPA_USE_LIBRI
    for (const auto &q : qpts)
    {
        coul_block.zero_out();

        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                coul_libri;
            coul_libri.clear();
            int count_coul = 0;
            for (const auto &Mu_Nu : local_atpair)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "myid " << blacs_ctxt_global_h.myid << "Mu " << Mu << " Nu " << Nu <<
                // endl;
                if (coulmat.count(Mu) == 0 || coulmat.at(Mu).count(Nu) == 0 ||
                    coulmat.at(Mu).at(Nu).count(q) == 0)
                    continue;
                const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                coul_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
                count_coul += 1;
            }
            int count_pair = 0;
            for (auto &Mu : coul_libri)
            {
                for (auto &nu_q : Mu.second)
                {
                    count_pair += 1;
                }
            }
            // printf("Finish RPA blacs 2d  vq arr\n");
            double arr_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            double comm_begin = omp_get_wtime();
            lib_printf(
                "Begin comm_map2_first  myid: %d  q:(%f, %f, %f)  count_coul: %d  count_pair: %d\n",
                mpi_comm_global_h.myid, q.x, q.y, q.z, count_coul, count_pair);
            const auto IJq_coul =
                comm_map2_first(mpi_comm_global_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            mpi_comm_global_h.barrier();
            // printf("End vq comm_map2_first  myid: %d   TIME_USED:
            // %f\n",mpi_comm_global_h.myid,comm_end-comm_begin);
            //  ofs_myid << "IJq_coul" << endl << IJq_coul;
            // printf("Finish RPA blacs 2d  vq 2d\n");
            double block_begin = omp_get_wtime();
            // for (const auto &IJ: set_IJ_nabf_nabf)
            // {
            //     const auto &I = IJ.first;
            //     const auto &J = IJ.second;
            //     // cout << IJq_coul.at(I).at({J, qa});
            //     collect_block_from_IJ_storage_syhe(
            //         coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
            //         IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            //     // lib_printf("myid %d I %d J %d nr %d nc %d\n%s",
            //     //        blacs_ctxt_global_h.myid, I, J,
            //     //        coul_block.nr(), coul_block.nc(),
            //     //        str(coul_block).c_str());
            // }
            collect_block_from_ALL_IJ_Tensor(coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf,
                                             qa, true, CONE, IJq_coul, MAJOR::ROW);
            double block_end = omp_get_wtime();
            // lib_printf("Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f
            // pair_size: %d\n",mpi_comm_global_h.myid,arr_end-vq_begin, comm_end-comm_begin,
            // block_end-block_begin,set_IJ_nabf_nabf.size());
            mpi_comm_global_h.barrier();
            double vq_end = omp_get_wtime();

            if (mpi_comm_global_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",
                           vq_end - vq_begin, comm_begin - vq_begin, block_begin - comm_begin,
                           vq_end - block_begin);
        }
    }
    lib_printf("Success test_libcomm_for_system\n");
#endif
}
