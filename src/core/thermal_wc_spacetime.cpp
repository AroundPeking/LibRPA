#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>

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

        // Only one R and a bounded element batch are flattened, never all Wc matrices.
        // The two dense buffers target 1 MiB total (or one column for a larger source grid).
        constexpr std::size_t MAX_BATCH_BYTES = 1024 * 1024;
        constexpr std::size_t MAX_BATCH_ELEMENTS = 4096;
        const auto bytes_per_element = sizeof(cplxdb) * (frequencies.size() + times.size());
        const auto batch_size = std::max<std::size_t>(
            1, std::min({MAX_BATCH_ELEMENTS, MAX_BATCH_BYTES / bytes_per_element,
                         std::size_t(std::numeric_limits<int>::max()) /
                             std::max(frequencies.size(), times.size())}));
        std::vector<cplxdb> phases(qlist.size());
        for (const auto& r : pbc.Rlist)
        {
            const auto cartesian_r = r * pbc.latvec;
            if (!finite(cartesian_r))
                throw std::overflow_error("thermal Wc lattice translation is nonfinite");
            for (std::size_t q = 0; q < qlist.size(); ++q)
            {
                const double angle = -TWO_PI * (qlist[q] * cartesian_r);
                if (!std::isfinite(angle))
                    throw std::overflow_error("thermal Wc spatial phase is nonfinite");
                phases[q] = cplxdb(std::cos(angle), std::sin(angle)) / double(qlist.size());
            }
            // Zero-size ranks retain shaped outputs and enter all collectives, but the
            // transform helper rejects empty batches, so do not call it on those ranks.
            for (std::size_t offset = 0; offset < size; offset += batch_size)
            {
                const int count = static_cast<int>(std::min(batch_size, size - offset));
                ComplexMatrix samples(static_cast<int>(frequencies.size()), count);
                for (std::size_t m = 0; m < frequencies.size(); ++m)
                {
                    const auto& qmap = Wc_freq_q.at(frequencies[m]);
                    for (std::size_t q = 0; q < qlist.size(); ++q)
                    {
                        const auto* values = qmap.at(qlist[q]).ptr() + offset;
                        for (int k = 0; k < count; ++k) samples(m, k) += phases[q] * values[k];
                    }
                }
                const auto transformed = transform.apply_bosonic_frequency_to_time(samples);
                for (std::size_t j = 0; j < times.size(); ++j)
                    std::copy_n(transformed.c + j * count, count,
                                result.at(times[j]).at(r).ptr() + offset);
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
