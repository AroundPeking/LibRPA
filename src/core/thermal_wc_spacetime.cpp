#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "../utils/constants.h"
#include "epsilon.h"

namespace librpa_int
{
namespace
{
using WcFrequency = std::map<double, std::map<Vector3_Order<double>, Matz>>;

bool finite(const cplxdb value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

bool finite(const Vector3<double>& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void collective_rethrow(const MpiCommHandler& comm, const std::exception_ptr& error)
{
    int failed = error ? 1 : 0;
    comm.allreduce(MPI_IN_PLACE, &failed, 1, MPI_MAX);
    if (failed)
    {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("thermal Wc spacetime transform failed on another MPI rank");
    }
}

void validate_pbc(const PeriodicBoundaryData& pbc)
{
    const auto& a = pbc.latvec;
    for (double value : {a.e11, a.e12, a.e13, a.e21, a.e22, a.e23, a.e31, a.e32, a.e33})
        if (!std::isfinite(value))
            throw std::invalid_argument("thermal Wc requires a finite lattice");

    std::size_t cells = 1;
    for (int period : {pbc.period.x, pbc.period.y, pbc.period.z})
    {
        if (period <= 0 || cells > std::size_t(std::numeric_limits<int>::max()) / period)
            throw std::invalid_argument("thermal Wc requires a positive representable BvK grid");
        cells *= period;
    }
    if (pbc.Rlist.size() != cells || pbc.klist_full.size() != cells)
        throw std::invalid_argument("thermal Wc requires complete full-q and R grids");

    std::set<Vector3_Order<int>> unique_r;
    for (const auto& r : pbc.Rlist)
    {
        // Use the same canonical BvK cell as construct_R_grid, independent of list order.
        if (r.x < -pbc.period.x / 2 || r.x > (pbc.period.x - 1) / 2 || r.y < -pbc.period.y / 2 ||
            r.y > (pbc.period.y - 1) / 2 || r.z < -pbc.period.z / 2 ||
            r.z > (pbc.period.z - 1) / 2 || !unique_r.insert(r).second)
            throw std::invalid_argument("thermal Wc Rlist must uniquely cover the BvK cell");
    }
    std::set<Vector3_Order<double>> unique_q;
    for (const auto& q : pbc.klist_full)
        if (!finite(q) || !unique_q.insert(q).second)
            throw std::invalid_argument("thermal Wc full-q metadata must be finite and unique");
}

void validate_matrix(const Matz& matrix)
{
    const int rows = matrix.nr(), cols = matrix.nc();
    if (rows < 0 || cols < 0 || (matrix.major() != ROW && matrix.major() != COL) ||
        (cols != 0 && std::size_t(rows) > std::size_t(std::numeric_limits<int>::max()) / cols))
        throw std::invalid_argument("thermal Wc local matrix has invalid dimensions or major");
    const auto size = std::size_t(rows) * cols;
    const auto storage = matrix.sptr();
    if (matrix.size() != size || matrix.ld() != (matrix.major() == ROW ? cols : rows) ||
        (size != 0 && (!storage || storage->size() != size)) ||
        (size == 0 && storage && storage->size() > 1))
        throw std::invalid_argument("thermal Wc local matrix has inconsistent storage");
    // Empty matrix_m objects may have null storage or a dummy entry, neither is a sample.
    for (std::size_t k = 0; k < size; ++k)
        if (!finite(matrix.ptr()[k]))
            throw std::invalid_argument("thermal Wc local matrix contains a nonfinite value");
}

void validate_samples(const WcFrequency& wc, const PeriodicBoundaryData& pbc,
                      const std::vector<double>& frequencies)
{
    if (frequencies.empty() || wc.size() != frequencies.size())
        throw std::invalid_argument("thermal Wc requires the exact signed physical frequency set");
    for (const auto& [frequency, qmap] : wc)
        if (!std::isfinite(frequency))
            throw std::invalid_argument("thermal Wc frequency keys must be finite");
    const Matz* layout = nullptr;
    for (double frequency : frequencies)
    {
        const auto it = wc.find(frequency);
        if (it == wc.end())
            throw std::invalid_argument("thermal Wc is missing an exact signed physical frequency");
        const auto& qmap = it->second;
        if (qmap.size() != pbc.klist_full.size())
            throw std::invalid_argument(
                "thermal Wc requires every full-q matrix at every frequency");
        for (const auto& [q, matrix] : qmap)
            if (!finite(q)) throw std::invalid_argument("thermal Wc q keys must be finite");
        for (const auto& q : pbc.klist_full)
        {
            const auto iq = qmap.find(q);
            if (iq == qmap.end())
                throw std::invalid_argument("thermal Wc is missing an explicit full-q matrix");
            const auto& matrix = iq->second;
            validate_matrix(matrix);
            if (layout && (matrix.nr() != layout->nr() || matrix.nc() != layout->nc() ||
                           matrix.major() != layout->major()))
                throw std::invalid_argument("thermal Wc local shapes and majors must agree");
            layout = &matrix;
        }
    }
}

constexpr std::size_t MAX_DENSE_ENTRIES =
    std::min(std::size_t(std::numeric_limits<int>::max()),
             std::numeric_limits<std::size_t>::max() / sizeof(cplxdb));

void validate_dense_dimensions(std::size_t rows, std::size_t cols)
{
    // ComplexMatrix forms its storage size and all element indices in int arithmetic.
    if (rows == 0 || cols == 0 || rows > MAX_DENSE_ENTRIES || cols > MAX_DENSE_ENTRIES ||
        rows > MAX_DENSE_ENTRIES / cols)
        throw std::overflow_error("thermal Wc dense dimensions must fit positive int storage");
}

struct SpatialBatch
{
    std::size_t r_count;
    std::size_t elements;
};

SpatialBatch choose_spatial_batch(std::size_t nq, std::size_t nr, std::size_t nb, std::size_t nt,
                                  std::size_t elements)
{
    validate_dense_dimensions(nq, 1);
    validate_dense_dimensions(nr, 1);
    validate_dense_dimensions(nb, 1);
    validate_dense_dimensions(nt, 1);
    constexpr std::size_t MAX_R_COUNT = 64;
    constexpr std::size_t TARGET_ENTRIES = 8 * 1024 * 1024 / sizeof(cplxdb);
    const auto max_grid = std::max(nb, nt);
    auto r_count =
        std::min({MAX_R_COUNT, nr, MAX_DENSE_ENTRIES / nq, MAX_DENSE_ENTRIES / max_grid});
    for (;; --r_count)
    {
        const auto phase_entries = r_count * nq;
        // Count phase + qpack + spatial result + B input + B output together.
        // Each product fits int; their sum uses wider arithmetic for large grids.
        const std::uint64_t entries_per_element =
            std::uint64_t(nq) + r_count + r_count * nb + r_count * nt;
        const auto budget_elements =
            phase_entries < TARGET_ENTRIES
                ? std::size_t((TARGET_ENTRIES - phase_entries) / entries_per_element)
                : 0;
        if (budget_elements != 0 || r_count == 1)
        {
            // If a single R/element exceeds 8 MiB, use that unavoidable minimum.
            // The int-storage limits remain mandatory even in this fallback.
            return {r_count, std::min({std::max<std::size_t>(1, elements), MAX_DENSE_ENTRIES / nq,
                                       MAX_DENSE_ENTRIES / (r_count * max_grid),
                                       std::max<std::size_t>(1, budget_elements)})};
        }
    }
}
}  // namespace

std::map<double, std::map<Vector3_Order<int>, Matz>> thermal_Wc_freq_q_to_tau_R(
    const MpiCommHandler& comm_h,
    const std::map<double, std::map<Vector3_Order<double>, Matz>>& Wc_freq_q,
    const PeriodicBoundaryData& pbc, const ThermalGWTransform& transform)
{
    std::exception_ptr error;
    std::vector<double> frequencies;
    try
    {
        validate_pbc(pbc);
        frequencies = transform.get_bosonic_frequencies_ha();
        if (transform.get_times().empty())
            throw std::invalid_argument("thermal Wc requires a nonempty time grid");
        validate_samples(Wc_freq_q, pbc, frequencies);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_rethrow(comm_h, error);

    const auto& layout = Wc_freq_q.begin()->second.begin()->second;
    int major_mask = 1 << layout.major();
    comm_h.allreduce(MPI_IN_PLACE, &major_mask, 1, MPI_BOR);
    if (major_mask != (1 << ROW) && major_mask != (1 << COL))
        throw std::invalid_argument("thermal Wc storage major must agree on every MPI rank");

    std::map<double, std::map<Vector3_Order<int>, Matz>> result;
    try
    {
        const auto& times = transform.get_times();
        const auto& qlist = pbc.klist_full;
        const auto size = layout.size();
        for (double time : times)
            for (const auto& r : pbc.Rlist)
                result[time].emplace(r, Matz(layout.nr(), layout.nc(), layout.major()));

        // Full Wc input/output stays resident. The O(NB*Nq) pointer index is separate
        // metadata, not part of the 8 MiB numerical work-buffer target below.
        std::vector<const Matz*> indexed_samples;
        if (frequencies.size() > std::numeric_limits<std::size_t>::max() / qlist.size() ||
            frequencies.size() * qlist.size() > indexed_samples.max_size())
            throw std::overflow_error("thermal Wc frequency/q pointer index is too large");
        indexed_samples.reserve(frequencies.size() * qlist.size());
        for (double frequency : frequencies)
        {
            const auto& qmap = Wc_freq_q.at(frequency);
            for (const auto& q : qlist) indexed_samples.push_back(&qmap.at(q));
        }

        const auto batch = choose_spatial_batch(qlist.size(), pbc.Rlist.size(), frequencies.size(),
                                                times.size(), size);
        for (std::size_t r_offset = 0; r_offset < pbc.Rlist.size(); r_offset += batch.r_count)
        {
            const auto r_count = std::min(batch.r_count, pbc.Rlist.size() - r_offset);
            validate_dense_dimensions(r_count, qlist.size());
            ComplexMatrix phases(static_cast<int>(r_count), static_cast<int>(qlist.size()));
            for (std::size_t ir = 0; ir < r_count; ++ir)
            {
                const auto cartesian_r = pbc.Rlist[r_offset + ir] * pbc.latvec;
                if (!finite(cartesian_r))
                    throw std::overflow_error("thermal Wc lattice translation is nonfinite");
                for (std::size_t q = 0; q < qlist.size(); ++q)
                {
                    const double angle = -TWO_PI * (qlist[q] * cartesian_r);
                    if (!std::isfinite(angle))
                        throw std::overflow_error("thermal Wc spatial phase is nonfinite");
                    phases(ir, q) = cplxdb(std::cos(angle), std::sin(angle)) / double(qlist.size());
                }
            }
            // Empty owners still validate every R/phase and join the final collective,
            // but never call either spatial GEMM or the B helper with empty storage.
            for (std::size_t offset = 0; offset < size; offset += batch.elements)
            {
                const auto count = std::min(batch.elements, size - offset);
                validate_dense_dimensions(qlist.size(), count);
                validate_dense_dimensions(r_count, count);
                const auto columns = r_count * count;
                validate_dense_dimensions(frequencies.size(), columns);
                validate_dense_dimensions(times.size(), columns);
                ComplexMatrix qpack(static_cast<int>(qlist.size()), static_cast<int>(count));
                ComplexMatrix samples(static_cast<int>(frequencies.size()),
                                      static_cast<int>(columns));
                for (std::size_t m = 0; m < frequencies.size(); ++m)
                {
                    for (std::size_t q = 0; q < qlist.size(); ++q)
                        std::copy_n(indexed_samples[m * qlist.size() + q]->ptr() + offset, count,
                                    qpack.c + q * count);
                    const auto spatial = phases * qpack;
                    // B columns enumerate (R, flat local entry), without changing ROW/COL
                    // semantics or completing any complex entries by conjugation.
                    std::copy_n(spatial.c, columns, samples.c + m * columns);
                }
                const auto transformed = transform.apply_bosonic_frequency_to_time(samples);
                for (std::size_t j = 0; j < times.size(); ++j)
                    for (std::size_t ir = 0; ir < r_count; ++ir)
                        std::copy_n(
                            transformed.c + j * columns + ir * count, count,
                            result.at(times[j]).at(pbc.Rlist[r_offset + ir]).ptr() + offset);
            }
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    // No collectives inside the local batch loops: ranks can have different local sizes,
    // including zero, and any rank-local allocation/arithmetic failure reaches this point.
    collective_rethrow(comm_h, error);
    return result;
}
}  // namespace librpa_int
