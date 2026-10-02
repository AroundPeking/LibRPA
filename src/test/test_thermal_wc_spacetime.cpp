#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <valarray>
#include <vector>

#include "../core/epsilon.h"

using namespace librpa_int;

namespace
{
using Complex = std::complex<double>;
using WcFrequency = std::map<double, std::map<Vector3_Order<double>, Matz>>;
using WcTime = std::map<double, std::map<Vector3_Order<int>, Matz>>;
const double PI = std::acos(-1.0);

void require(bool condition, const std::string& message);

double sinc(const double value)
{
    return std::abs(value) < 1.0e-14 ? 1.0 : std::sin(value) / value;
}

double gamma_cell_fourier_moment(const PeriodicBoundaryData& pbc,
                                 const Vector3_Order<int>& r)
{
    return sinc(PI * r.x / pbc.period.x) * sinc(PI * r.y / pbc.period.y) *
           sinc(PI * r.z / pbc.period.z);
}

int wrap_periodic_index(const int index, const int period)
{
    const int remainder = index % period;
    return remainder < 0 ? remainder + period : remainder;
}

std::map<Vector3_Order<int>, Complex> high_order_tetra_q_to_r_reference(
    const PeriodicBoundaryData& pbc, const std::vector<Complex>& nodal_values)
{
    using GridKey = std::array<int, 3>;
    require(nodal_values.size() == pbc.klist_full.size(),
            "high-order tetrahedron reference has inconsistent nodal values");

    std::map<GridKey, std::size_t> q_index;
    for (std::size_t iq = 0; iq < pbc.kfrac_list_full.size(); ++iq)
    {
        const auto& q = pbc.kfrac_list_full[iq];
        const std::array<double, 3> scaled{
            q.x * pbc.period.x, q.y * pbc.period.y, q.z * pbc.period.z};
        GridKey key{};
        for (int axis = 0; axis != 3; ++axis)
        {
            const double nearest = std::round(scaled[axis]);
            require(std::abs(scaled[axis] - nearest) < 1.0e-10,
                    "high-order tetrahedron reference requires a mesh-aligned q point");
            const int period = axis == 0 ? pbc.period.x : axis == 1 ? pbc.period.y : pbc.period.z;
            key[axis] = wrap_periodic_index(static_cast<int>(nearest), period);
        }
        require(q_index.emplace(key, iq).second,
                "high-order tetrahedron reference found duplicate q point");
    }
    require(q_index.size() == nodal_values.size(),
            "high-order tetrahedron reference did not index the complete q grid");

    static constexpr std::array<std::array<std::array<int, 3>, 4>, 6> tetra_vertices{{
        {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}},
        {{{0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 1, 1}}},
        {{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}}},
        {{{0, 0, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1}}},
        {{{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}}},
        {{{0, 0, 0}, {1, 0, 1}, {1, 0, 0}, {1, 1, 1}}}}};
    static constexpr std::array<double, 8> gauss_x{{
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,
        0.7966664774136267,  0.9602898564975363}};
    static constexpr std::array<double, 8> gauss_w{{
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
        0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763}};

    std::map<Vector3_Order<int>, Complex> result;
    for (const auto& r : pbc.Rlist) result.emplace(r, Complex{0.0, 0.0});
    const double cell_jacobian = 1.0 / static_cast<double>(nodal_values.size());
    for (int ix = 0; ix < pbc.period.x; ++ix)
        for (int iy = 0; iy < pbc.period.y; ++iy)
            for (int iz = 0; iz < pbc.period.z; ++iz)
                for (const auto& tetra : tetra_vertices)
                {
                    std::array<std::size_t, 4> vertices{};
                    std::array<Vector3_Order<double>, 4> coordinates{};
                    for (int vertex = 0; vertex != 4; ++vertex)
                    {
                        const auto& offset = tetra[vertex];
                        const GridKey key{wrap_periodic_index(ix + offset[0], pbc.period.x),
                                          wrap_periodic_index(iy + offset[1], pbc.period.y),
                                          wrap_periodic_index(iz + offset[2], pbc.period.z)};
                        vertices[vertex] = q_index.at(key);
                        coordinates[vertex] = {
                            static_cast<double>(ix + offset[0]) / pbc.period.x,
                            static_cast<double>(iy + offset[1]) / pbc.period.y,
                            static_cast<double>(iz + offset[2]) / pbc.period.z};
                    }
                    for (std::size_t iu = 0; iu < gauss_x.size(); ++iu)
                        for (std::size_t iv = 0; iv < gauss_x.size(); ++iv)
                            for (std::size_t iw = 0; iw < gauss_x.size(); ++iw)
                            {
                                const double u = 0.5 * (gauss_x[iu] + 1.0);
                                const double v = 0.5 * (gauss_x[iv] + 1.0);
                                const double w = 0.5 * (gauss_x[iw] + 1.0);
                                const std::array<double, 4> lambda{{
                                    1.0 - u, u * (1.0 - v), u * v * (1.0 - w), u * v * w}};
                                Vector3_Order<double> point{0.0, 0.0, 0.0};
                                Complex value{0.0, 0.0};
                                for (int vertex = 0; vertex != 4; ++vertex)
                                {
                                    point.x += lambda[vertex] * coordinates[vertex].x;
                                    point.y += lambda[vertex] * coordinates[vertex].y;
                                    point.z += lambda[vertex] * coordinates[vertex].z;
                                    value += lambda[vertex] * nodal_values[vertices[vertex]];
                                }
                                const double weight = cell_jacobian * gauss_w[iu] * gauss_w[iv] *
                                                      gauss_w[iw] * u * u * v / 8.0;
                                for (auto& [r, integral] : result)
                                {
                                    const double angle = -2.0 * PI *
                                                         (point.x * r.x + point.y * r.y +
                                                          point.z * r.z);
                                    integral += weight * std::exp(Complex(0.0, angle)) * value;
                                }
                            }
                }
    return result;
}

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(Complex actual, Complex expected)
{
    require(std::isfinite(actual.real()) && std::isfinite(actual.imag()) &&
                std::abs(actual - expected) <= 3e-12 * std::max(1.0, std::abs(expected)),
            "independent full complex spatial/time sum differs");
}

Complex bosonic_coefficient(int j, int m)
{
    return {0.13 + 0.04 * j - 0.07 * m, -0.05 + 0.03 * j * m + 0.02 * m};
}

ThermalGWTransform make_transform()
{
    ComplexMatrix b(3, 5), f(2, 3);
    for (int j = 0; j < b.nr; ++j)
        for (int m = 0; m < b.nc; ++m) b(j, m) = bosonic_coefficient(j, m);
    return ThermalGWTransform(2.7, {0.2, 0.7, 1.6}, {2, 0, -3, -1, 1}, {-2, 0}, b, f);
}

PeriodicBoundaryData make_pbc(int nx, int ny, int nz)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.0, 0.5, 0.25, 0.0, 3.0, 0.75, 0.0, 0.0, 4.0});
    pbc.set_period(nx, ny, nz);
    pbc.klist_full.clear();
    for (int x = 0; x < pbc.period.x; ++x)
        for (int y = 0; y < pbc.period.y; ++y)
            for (int z = 0; z < pbc.period.z; ++z)
            {
                // Solve latvec*q = fractional q explicitly, in internal 2*pi/Bohr units.
                const double qz = double(z) / pbc.period.z / 4.0;
                const double qy = (double(y) / pbc.period.y - 0.75 * qz) / 3.0;
                const double qx = (double(x) / pbc.period.x - 0.5 * qy - 0.25 * qz) / 2.0;
                pbc.klist_full.emplace_back(qx, qy, qz);
            }
    std::reverse(pbc.klist_full.begin(), pbc.klist_full.end());
    std::reverse(pbc.Rlist.begin(), pbc.Rlist.end());
    // Deliberately irrelevant reduced-q metadata: the thermal route must never consult it.
    pbc.klist_coul = {{0.0, 0.0, 0.0}};
    pbc.weight_q = {17.0};
    return pbc;
}

PeriodicBoundaryData make_pbc(bool gamma = false)
{
    return make_pbc(gamma ? 1 : 3, gamma ? 1 : 2, gamma ? 1 : 2);
}

WcFrequency make_samples(const PeriodicBoundaryData& pbc, const ThermalGWTransform& transform,
                         int rank, int rows, int cols, MAJOR major, bool nonseparable = false)
{
    WcFrequency wc;
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    for (std::size_t m = 0; m < frequencies.size(); ++m)
        for (std::size_t q = 0; q < pbc.klist_full.size(); ++q)
        {
            Matz matrix(rows, cols, major);
            for (int row = 0; row < rows; ++row)
                for (int col = 0; col < cols; ++col)
                {
                    matrix(row, col) =
                        Complex(0.4 + 0.7 * m - 0.23 * q + 0.011 * row - 0.019 * col + rank,
                                -0.3 + 0.09 * m * m + 0.17 * q + 0.007 * row + 0.013 * col * col -
                                    0.4 * rank);
                    if (nonseparable)
                    {
                        // Bounded q/mode/element cross terms retain element differences at
                        // nonzero R; purely additive fixture terms can cancel there.
                        const double phase = 0.013 * (q + 1) * (m + 1) * (row + 1) +
                                             0.017 * (q + 2) * (m + 2) * (col + 1);
                        matrix(row, col) +=
                            Complex(0.37 * std::sin(phase),
                                    0.29 * std::cos(0.83 * phase +
                                                    0.011 * (q + 1) * (m + 2) * (row + col + 1)));
                    }
                }
            wc[frequencies[m]].emplace(pbc.klist_full[q], std::move(matrix));
        }
    return wc;
}

struct MatrixSnapshot
{
    double frequency;
    Vector3_Order<double> q;
    int rows;
    int cols;
    MAJOR major;
    Matz::container_t storage;
    std::vector<Complex> values;
};

std::vector<MatrixSnapshot> snapshot(const WcFrequency& wc)
{
    std::vector<MatrixSnapshot> result;
    for (const auto& [frequency, qmap] : wc)
        for (const auto& [q, matrix] : qmap)
        {
            result.push_back(
                {frequency, q, matrix.nr(), matrix.nc(), matrix.major(), matrix.sptr(), {}});
            if (matrix.size())
                result.back().values.assign(matrix.ptr(), matrix.ptr() + matrix.size());
        }
    return result;
}

void unchanged(const WcFrequency& wc, const std::vector<MatrixSnapshot>& before)
{
    const auto after = snapshot(wc);
    require(after.size() == before.size(), "input keys changed");
    for (std::size_t k = 0; k < before.size(); ++k)
    {
        const auto& a = after[k];
        const auto& b = before[k];
        require(a.frequency == b.frequency && a.q == b.q && a.rows == b.rows && a.cols == b.cols &&
                    a.major == b.major && a.storage == b.storage && a.values == b.values,
                "input metadata, storage or values mutated");
    }
}

void check_sum(const WcTime& output, const WcFrequency& input, const PeriodicBoundaryData& pbc,
               const ThermalGWTransform& transform, int rows, int cols, MAJOR major,
               bool quadrature = false, const std::vector<std::size_t>& selected_elements = {})
{
    const auto& times = transform.get_times();
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    auto elements = selected_elements;
    if (elements.empty())
    {
        elements.resize(std::size_t(rows) * cols);
        std::iota(elements.begin(), elements.end(), std::size_t(0));
    }
    require(output.size() == times.size(), "time keys missing (thermal Wc not implemented)");
    for (std::size_t j = 0; j < times.size(); ++j)
    {
        const auto& rmap = output.at(times[j]);
        require(rmap.size() == pbc.Rlist.size(), "R keys missing");
        for (const auto& r : pbc.Rlist)
        {
            const auto& actual = rmap.at(r);
            require(actual.nr() == rows && actual.nc() == cols && actual.major() == major &&
                        actual.size() == std::size_t(rows) * cols,
                    "local dimensions or storage order changed");
            if (actual.size() == 0) continue;
            std::vector<Complex> phases;
            for (const auto& q : pbc.klist_full)
            {
                const auto& a = pbc.latvec;
                const double dot = q.x * (r.x * a.e11 + r.y * a.e21 + r.z * a.e31) +
                                   q.y * (r.x * a.e12 + r.y * a.e22 + r.z * a.e32) +
                                   q.z * (r.x * a.e13 + r.y * a.e23 + r.z * a.e33);
                phases.push_back(std::exp(Complex(0.0, -2.0 * PI * dot)) /
                                 double(pbc.klist_full.size()));
            }
            for (const auto element : elements)
            {
                require(element < actual.size(), "oracle element is outside the local matrix");
                const int row = major == ROW ? element / cols : element % rows;
                const int col = major == ROW ? element % cols : element / rows;
                Complex expected = 0.0;
                // Independent old sum: complete q sum for each caller-ordered signed mode,
                // followed by explicit B arithmetic. No production GEMM or apply helper.
                for (std::size_t m = 0; m < frequencies.size(); ++m)
                {
                    Complex spatial = 0.0;
                    const auto& qmap = input.at(frequencies[m]);
                    for (std::size_t q = 0; q < pbc.klist_full.size(); ++q)
                        spatial += phases[q] * qmap.at(pbc.klist_full[q])(row, col);
                    const auto b = quadrature ? std::exp(Complex(0.0, -frequencies[m] * times[j])) /
                                                    transform.get_beta_ha_inv()
                                              : bosonic_coefficient(j, m);
                    expected += b * spatial;
                }
                close(actual(row, col), expected);
            }
        }
    }
}

void check_success(const MpiCommHandler& comm, MAJOR major, int rows = 2, int cols = 3,
                   const PeriodicBoundaryData& pbc = make_pbc(), bool quadrature = false,
                   const std::vector<std::size_t>& selected_elements = {})
{
    const auto transform =
        quadrature ? ThermalGWTransform::from_quadrature(2.7, {0.2, 0.7, 1.6}, {0.4, 0.8, 1.5},
                                                         {2, 0, -3, -1, 1}, {-2, 0})
                   : make_transform();
    const auto wc =
        make_samples(pbc, transform, comm.myid, rows, cols, major, !selected_elements.empty());
    const auto before = snapshot(wc);
    auto output = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    check_sum(output, wc, pbc, transform, rows, cols, major, quadrature, selected_elements);
    if (!selected_elements.empty() && rows != 0 && cols != 0)
    {
        // Negative control: the oracle must detect broadcasting (0,0) into the final
        // element of the final R tile, even when other selected elements are correct.
        auto& last = output.at(transform.get_times().front()).at(pbc.Rlist.back());
        const auto tail = selected_elements.back();
        const auto saved = last.ptr()[tail];
        last.ptr()[tail] = last(0, 0);
        bool rejected = false;
        try
        {
            check_sum(output, wc, pbc, transform, rows, cols, major, quadrature, {tail});
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        last.ptr()[tail] = saved;
        require(rejected, "boundary oracle did not detect the final R/element broadcast");
    }
    // Output must own its storage, including the singleton-Gamma path.
    for (auto& [time, rmap] : output)
        for (auto& [r, matrix] : rmap)
            if (matrix.size()) matrix(0, 0) += Complex(99.0, -77.0);
    unchanged(wc, before);
}

void check_stream_matches_full(const MpiCommHandler& comm)
{
    const auto pbc = make_pbc(2, 2, 2);
    const auto transform = make_transform();
    const auto wc = make_samples(pbc, transform, comm.myid, 2, 3, COL, true);
    const auto full = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    auto streamed_input = wc;
    std::size_t callbacks = 0;
    thermal_Wc_freq_q_to_tau_R_stream(
        comm, streamed_input, pbc, transform,
        [&](std::size_t index, double time, std::map<Vector3_Order<int>, Matz>&& streamed)
        {
            require(index == callbacks, "stream callback time index is not ordered");
            const auto& expected = full.at(time);
            require(streamed.size() == expected.size(), "stream callback dropped an R block");
            for (const auto& [r, matrix] : streamed)
            {
                const auto& reference = expected.at(r);
                require(matrix.nr() == reference.nr() && matrix.nc() == reference.nc() &&
                            matrix.major() == reference.major(),
                        "stream callback changed an R block shape");
                for (std::size_t element = 0; element < matrix.size(); ++element)
                    close(matrix.ptr()[element], reference.ptr()[element]);
            }
            ++callbacks;
        });
    require(callbacks == transform.get_times().size(), "stream callback missed a time point");
    require(streamed_input.empty(), "stream transform retained consumed q-space matrices");
}

void check_unfolded_rlist_preserves_bvk_values(const MpiCommHandler& comm)
{
    const auto pbc = make_pbc(2, 2, 2);
    const auto transform = make_transform();
    const auto wc = make_samples(pbc, transform, comm.myid, 2, 3, COL, true);
    const auto reference = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    const auto unfolded_grid = build_tetrahedron_unfolded_grid(pbc.period, {3, 3, 3});
    const auto unfolded =
        thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform, unfolded_grid.Rlist);

    for (const auto& time : transform.get_times())
    {
        const auto& expanded_blocks = unfolded.at(time);
        require(expanded_blocks.size() == unfolded_grid.Rlist.size(),
                "unfolded Wc transform dropped a real-space image");
        for (const auto& r : pbc.Rlist)
        {
            const auto& actual = expanded_blocks.at(r);
            const auto& expected = reference.at(time).at(r);
            for (std::size_t element = 0; element < actual.size(); ++element)
                close(actual.ptr()[element], expected.ptr()[element]);
        }
        for (const auto& r : unfolded_grid.Rlist)
        {
            const auto& block = expanded_blocks.at(r);
            for (std::size_t element = 0; element < block.size(); ++element)
                require(std::isfinite(block.ptr()[element].real()) &&
                            std::isfinite(block.ptr()[element].imag()),
                        "unfolded Wc transform produced a nonfinite image");
        }
    }
}

void check_tetra_q2r_diagnostic(const MpiCommHandler& comm)
{
    auto pbc = make_pbc(2, 2, 2);
    for (const auto& q : pbc.klist_full) pbc.kfrac_list_full.push_back(pbc.latvec * q);
    const auto transform = ThermalGWTransform::from_quadrature(
        2.0, {0.4}, {1.0}, {0}, {0});
    WcFrequency wc;
    for (const auto& q : pbc.klist_full)
    {
        const auto qfrac = pbc.latvec * q;
        Matz matrix(1, 1, ROW);
        matrix(0, 0) = 0.7 + 0.31 * qfrac.x * qfrac.x + 0.17 * qfrac.y * qfrac.z;
        wc[0.0].emplace(q, std::move(matrix));
    }

    unsetenv("LIBRPA_TETRA_Q2R_DIAG");
    const auto uniform = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    setenv("LIBRPA_TETRA_Q2R_DIAG", "enabled", 1);
    const auto tetra = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);

    const Vector3_Order<int> gamma_r{0, 0, 0};
    require(std::isfinite(tetra.at(0.4).at(gamma_r)(0, 0).real()) &&
                std::isfinite(tetra.at(0.4).at(gamma_r)(0, 0).imag()),
            "tetra q->R diagnostic produced a nonfinite Gamma-R value");
    const auto nonzero_iter = std::find_if(
        pbc.Rlist.cbegin(), pbc.Rlist.cend(), [](const auto& r) { return r != Vector3_Order<int>{0, 0, 0}; });
    require(nonzero_iter != pbc.Rlist.cend(), "tetra diagnostic needs a nonzero R point");
    const auto& nonzero_r = *nonzero_iter;
    require(std::abs(tetra.at(0.4).at(nonzero_r)(0, 0) - uniform.at(0.4).at(nonzero_r)(0, 0)) >
                1.0e-8,
            "tetra q->R diagnostic did not change the nonzero-R quadrature");

    // A metallic static Gamma sample is a Gamma-cell average, not a nodal value.
    // With all other q samples zero, a corrected transform must keep the average as
    // a single Gamma-cell contribution instead of spreading it through adjacent tetrahedra.
    WcFrequency gamma_cell_average;
    for (const auto& q : pbc.klist_full)
    {
        Matz matrix(1, 1, ROW);
        matrix(0, 0) = q == Vector3_Order<double>{0.0, 0.0, 0.0} ? 1.0 : 0.0;
        gamma_cell_average[0.0].emplace(q, std::move(matrix));
    }
    unsetenv("LIBRPA_TETRA_GAMMA_CELL_AVERAGE");
    const auto naive_gamma = thermal_Wc_freq_q_to_tau_R(comm, gamma_cell_average, pbc, transform);
    setenv("LIBRPA_TETRA_GAMMA_CELL_AVERAGE", "enabled", 1);
    const auto corrected_gamma =
        thermal_Wc_freq_q_to_tau_R(comm, gamma_cell_average, pbc, transform);
    close(corrected_gamma.at(0.4).at(nonzero_r)(0, 0),
          Complex(gamma_cell_fourier_moment(pbc, nonzero_r) /
                      static_cast<double>(pbc.klist_full.size()) /
                      transform.get_beta_ha_inv(),
                  0.0));
    require(std::abs(naive_gamma.at(0.4).at(nonzero_r)(0, 0) -
                    corrected_gamma.at(0.4).at(nonzero_r)(0, 0)) >
                1.0e-8,
            "Gamma-cell average was still treated as an ordinary tetrahedron vertex");

    unsetenv("LIBRPA_TETRA_GAMMA_CELL_AVERAGE");
    auto streamed_input = wc;
    std::map<Vector3_Order<int>, Matz> streamed;
    thermal_Wc_freq_q_to_tau_R_stream(
        comm, streamed_input, pbc, transform,
        [&](std::size_t index, double time, std::map<Vector3_Order<int>, Matz>&& value)
        {
            require(index == 0 && time == 0.4, "tetra stream callback metadata changed");
            streamed = std::move(value);
        });
    close(streamed.at(nonzero_r)(0, 0), tetra.at(0.4).at(nonzero_r)(0, 0));
    require(streamed_input.empty(), "tetra stream did not consume q-space samples");
    unsetenv("LIBRPA_TETRA_Q2R_DIAG");
    unsetenv("LIBRPA_TETRA_GAMMA_CELL_AVERAGE");
}

void check_tetra_q2r_matches_high_order_reference(const MpiCommHandler& comm)
{
    auto pbc = make_pbc(2, 2, 2);
    for (const auto& q : pbc.klist_full) pbc.kfrac_list_full.push_back(pbc.latvec * q);
    const auto transform = ThermalGWTransform::from_quadrature(2.0, {0.4}, {1.0}, {0}, {0});
    WcFrequency wc;
    std::vector<Complex> nodal_values;
    nodal_values.reserve(pbc.klist_full.size());
    for (const auto& q : pbc.klist_full)
    {
        const auto qfrac = pbc.latvec * q;
        const Complex value{0.7 + 0.31 * qfrac.x * qfrac.x + 0.17 * qfrac.y * qfrac.z,
                            -0.23 + 0.29 * qfrac.x - 0.11 * qfrac.y * qfrac.z};
        nodal_values.push_back(value);
        Matz matrix(1, 1, ROW);
        matrix(0, 0) = value;
        wc[0.0].emplace(q, std::move(matrix));
    }
    const auto expected = high_order_tetra_q_to_r_reference(pbc, nodal_values);

    setenv("LIBRPA_TETRA_Q2R_DIAG", "enabled", 1);
    unsetenv("LIBRPA_TETRA_GAMMA_CELL_AVERAGE");
    const auto tetra = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    for (const auto& r : pbc.Rlist)
        close(tetra.at(0.4).at(r)(0, 0), expected.at(r) / transform.get_beta_ha_inv());
    unsetenv("LIBRPA_TETRA_Q2R_DIAG");
}

void check_batch_boundaries(const MpiCommHandler& comm, MAJOR major, bool empty_owner = false)
{
    auto pbc = make_pbc(5, 13, 1);
    // Permute q and R independently, crossing the 64-R tile with a one-R tail.
    std::rotate(pbc.klist_full.begin(), pbc.klist_full.begin() + 7, pbc.klist_full.end());
    std::rotate(pbc.Rlist.begin(), pbc.Rlist.begin() + 19, pbc.Rlist.end());
    // With 65 q, 5 B modes and 3 times, an 8 MiB/64-R slab holds 811 entries.
    // 17*97 = 1649 covers two complete element batches and a 27-entry tail.
    const std::vector<std::size_t> elements{0, 1, 810, 811, 812, 1621, 1622, 1623, 1647, 1648};
    const bool empty = empty_owner && comm.myid == comm.nprocs - 1;
    const int rows = empty && major == ROW ? 0 : 17;
    const int cols = empty && major == COL ? 0 : 97;
    check_success(comm, major, rows, cols, pbc, false, elements);
}

using Mutation = std::function<void(WcFrequency&, PeriodicBoundaryData&)>;

void check_rejection(const MpiCommHandler& comm, const Mutation& mutate, bool empty_rank = false)
{
    auto pbc = make_pbc();
    const auto transform = make_transform();
    const bool bad_rank = comm.myid == comm.nprocs - 1;
    auto wc = make_samples(pbc, transform, comm.myid, bad_rank && empty_rank ? 0 : 2, 3, COL);
    if (bad_rank) mutate(wc, pbc);
    bool rejected = false;
    try
    {
        thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    int count = rejected ? 1 : 0;
    comm.allreduce(MPI_IN_PLACE, &count, 1, MPI_SUM);
    require(count == comm.nprocs, "malformed rank-local input was not rejected on EVERY rank");
}

void check_overflow(const MpiCommHandler& comm, bool spatial, bool empty_rank = false)
{
    auto pbc = make_pbc(!spatial);
    ComplexMatrix b(1, 1), f(1, 1);
    b(0, 0) = spatial ? 1.0 : std::numeric_limits<double>::max();
    const ThermalGWTransform transform(2.0, {1.0}, {0}, {0}, b, f);
    const bool empty = empty_rank && comm.myid == comm.nprocs - 1;
    auto wc = make_samples(pbc, transform, comm.myid, empty ? 0 : 1, 1, ROW);
    for (auto& [frequency, qmap] : wc)
        for (auto& [q, matrix] : qmap)
            if (matrix.size()) matrix(0, 0) = comm.myid == comm.nprocs - 1 ? 2.0 : 0.0;
    if (spatial && comm.myid == comm.nprocs - 1)
    {
        pbc.latvec.e11 = std::numeric_limits<double>::max();
        auto node = wc.begin()->second.extract(pbc.klist_full.front());
        pbc.klist_full.front().x = std::numeric_limits<double>::max();
        node.key() = pbc.klist_full.front();
        wc.begin()->second.insert(std::move(node));
    }
    const auto before = snapshot(wc);
    bool rejected = false;
    try
    {
        thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    int count = rejected ? 1 : 0;
    comm.allreduce(MPI_IN_PLACE, &count, 1, MPI_SUM);
    require(count == comm.nprocs, "rank-local nonfinite arithmetic not rejected collectively");
    unchanged(wc, before);
}
}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const MpiCommHandler comm(MPI_COMM_WORLD, true);
    using Test = std::pair<std::string, std::function<void()>>;
    std::vector<Test> tests{
        {"independent rectangular B full complex sum ROW", [&] { check_success(comm, ROW); }},
        {"independent rectangular B full complex sum COL", [&] { check_success(comm, COL); }},
        {"different local shapes with common storage order",
         [&] { check_success(comm, COL, 2 + comm.myid, 3); }},
        {"singleton Gamma still applies B and owns storage",
         [&] { check_success(comm, COL, 3, 3, make_pbc(true)); }},
        {"quadrature beta normalization and signed phases",
         [&] { check_success(comm, ROW, 2, 3, make_pbc(), true); }},
        {"multiple bounded element batches", [&] { check_success(comm, COL, 65, 129); }},
        {"exactly 64 R with full-element scalar oracle",
         [&] { check_success(comm, ROW, 2, 3, make_pbc(4, 4, 4)); }},
        {"non-Hermitian square matrices across R tiles",
         [&] { check_success(comm, COL, 3, 3, make_pbc(5, 13, 1)); }},
        {"65 R and multiple element tails ROW", [&] { check_batch_boundaries(comm, ROW); }},
        {"65 R and multiple element tails COL", [&] { check_batch_boundaries(comm, COL); }},
        {"tiled transform with zero-row owner", [&] { check_batch_boundaries(comm, ROW, true); }},
        {"tiled transform with zero-column owner",
         [&] { check_batch_boundaries(comm, COL, true); }},
        {"zero local rows retain keys",
         [&] { check_success(comm, ROW, comm.myid == comm.nprocs - 1 ? 0 : 2, 3); }},
        {"zero local columns retain keys",
         [&] { check_success(comm, COL, 2, comm.myid == comm.nprocs - 1 ? 0 : 3); }},
        {"all ranks zero local size", [&] { check_success(comm, COL, 0, 0); }},
        {"streaming time transform matches resident transform",
         [&] { check_stream_matches_full(comm); }},
        {"unfolded R-list preserves BvK Wc values",
         [&] { check_unfolded_rlist_preserves_bvk_values(comm); }},
        {"periodic tetra q-to-R diagnostic",
         [&] { check_tetra_q2r_diagnostic(comm); }},
        {"tetra q-to-R matches independent high-order reference",
         [&] { check_tetra_q2r_matches_high_order_reference(comm); }},
        {"default zero matrix needs no backing storage",
         [&]
         {
             const auto pbc = make_pbc();
             const auto transform = make_transform();
             auto wc = make_samples(pbc, transform, comm.myid, 0, 0, ROW);
             for (auto& [frequency, qmap] : wc)
                 for (auto& [q, matrix] : qmap) matrix = Matz();
             const auto before = snapshot(wc);
             const auto output = thermal_Wc_freq_q_to_tau_R(comm, wc, pbc, transform);
             check_sum(output, wc, pbc, transform, 0, 0, ROW);
             unchanged(wc, before);
         }},
        {"collective rejection of rank-dependent storage order",
         [&]
         {
             if (comm.nprocs > 1)
                 check_rejection(comm,
                                 [](auto& wc, auto&)
                                 {
                                     for (auto& [frequency, qmap] : wc)
                                         for (auto& [q, matrix] : qmap) matrix = Matz(2, 3, ROW);
                                 });
         }},
        {"empty rank participates in collective storage order check",
         [&]
         {
             if (comm.nprocs > 1)
                 check_rejection(
                     comm,
                     [](auto& wc, auto&)
                     {
                         for (auto& [frequency, qmap] : wc)
                             for (auto& [q, matrix] : qmap) matrix = Matz(0, 3, ROW);
                     },
                     true);
         }},
        {"rank-local time arithmetic overflow", [&] { check_overflow(comm, false); }},
        {"rank-local spatial arithmetic overflow", [&] { check_overflow(comm, true); }},
        {"empty owner still rejects spatial arithmetic overflow",
         [&] { check_overflow(comm, true, true); }}};

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::pair<std::string, Mutation>> invalid{
        {"empty frequency map", [](auto& wc, auto&) { wc.clear(); }},
        {"missing negative frequency", [](auto& wc, auto&) { wc.erase(wc.begin()); }},
        {"extra frequency", [](auto& wc, auto&) { wc.emplace(123.0, wc.begin()->second); }},
        {"inexact physical frequency",
         [](auto& wc, auto&)
         {
             auto node = wc.extract(wc.begin());
             node.key() = std::nextafter(node.key(), 0.0);
             wc.insert(std::move(node));
         }},
        {"nonfinite frequency",
         [=](auto& wc, auto&)
         {
             auto node = wc.extract(wc.begin());
             node.key() = inf;
             wc.insert(std::move(node));
         }},
        {"missing full q",
         [](auto& wc, auto&) { wc.begin()->second.erase(wc.begin()->second.begin()); }},
        {"empty q map", [](auto& wc, auto&) { wc.begin()->second.clear(); }},
        {"extra q",
         [](auto& wc, auto&) {
             wc.begin()->second.emplace(Vector3_Order<double>{4.0, 5.0, 6.0}, Matz(2, 3, COL));
         }},
        {"wrong q with unchanged count",
         [](auto& wc, auto&)
         {
             auto node = wc.begin()->second.extract(wc.begin()->second.begin());
             node.key().x += 0.123;
             wc.begin()->second.insert(std::move(node));
         }},
        {"nonfinite q key",
         [=](auto& wc, auto&)
         {
             auto node = wc.begin()->second.extract(wc.begin()->second.begin());
             node.key().x = inf;
             wc.begin()->second.insert(std::move(node));
         }},
        {"shape mismatch",
         [](auto& wc, auto&) { wc.begin()->second.begin()->second = Matz(3, 2, COL); }},
        {"major mismatch",
         [](auto& wc, auto&) { wc.begin()->second.begin()->second = Matz(2, 3, ROW); }},
        {"invalid AUTO storage order",
         [](auto& wc, auto&) { wc.begin()->second.begin()->second = Matz(2, 3, AUTO); }},
        {"null nonempty storage", [](auto& wc, auto&)
         { wc.begin()->second.begin()->second = Matz(2, 3, Matz::container_t{}, COL); }},
        {"short backing storage",
         [](auto& wc, auto&) { wc.begin()->second.begin()->second.sptr()->resize(5); }},
        {"oversized backing storage",
         [](auto& wc, auto&) { wc.begin()->second.begin()->second.sptr()->resize(7); }},
        {"nonfinite real matrix value",
         [=](auto& wc, auto&) { wc.begin()->second.begin()->second(1, 2) = Complex(nan, 0.0); }},
        {"nonfinite imaginary matrix value",
         [=](auto& wc, auto&) { wc.begin()->second.begin()->second(1, 0) = Complex(0.0, inf); }},
        {"nonfinite lattice", [=](auto&, auto& pbc) { pbc.latvec.e23 = nan; }},
        {"empty full q metadata", [](auto&, auto& pbc) { pbc.klist_full.clear(); }},
        {"incomplete full q metadata", [](auto&, auto& pbc) { pbc.klist_full.pop_back(); }},
        {"duplicate full q metadata",
         [](auto&, auto& pbc) { pbc.klist_full[0] = pbc.klist_full[1]; }},
        {"nonfinite full q metadata", [=](auto&, auto& pbc) { pbc.klist_full[0].z = nan; }},
        {"empty R metadata", [](auto&, auto& pbc) { pbc.Rlist.clear(); }},
        {"incomplete R metadata", [](auto&, auto& pbc) { pbc.Rlist.pop_back(); }},
        {"duplicate R metadata", [](auto&, auto& pbc) { pbc.Rlist[0] = pbc.Rlist[1]; }},
        {"out of grid R", [](auto&, auto& pbc) { pbc.Rlist[0].x += pbc.period.x; }},
        {"invalid period", [](auto&, auto& pbc) { pbc.period.x = 0; }},
        {"overflowing period product", [](auto&, auto& pbc) {
             pbc.period = {std::numeric_limits<int>::max(), 2, 2};
         }}};
    for (const auto& [name, mutate] : invalid)
        tests.emplace_back("collective rejection: " + name,
                           [&, mutate] { check_rejection(comm, mutate); });
    tests.emplace_back("zero-local-size rank still rejects missing q",
                       [&]
                       {
                           check_rejection(
                               comm, [](auto& wc, auto&) { wc.begin()->second.clear(); }, true);
                       });
    tests.emplace_back("zero-local-size rank still rejects missing frequency",
                       [&]
                       {
                           check_rejection(
                               comm, [](auto& wc, auto&) { wc.erase(wc.begin()); }, true);
                       });

    int failures = 0;
    for (const auto& [name, test] : tests)
    {
        int failed = 0;
        try
        {
            test();
        }
        catch (const std::exception& error)
        {
            failed = 1;
            std::cerr << "rank " << comm.myid << " FAIL " << name << ": " << error.what() << '\n';
        }
        comm.allreduce(MPI_IN_PLACE, &failed, 1, MPI_MAX);
        failures += failed;
        if (!failed && comm.is_root()) std::cout << "PASS " << name << '\n';
    }
    if (comm.is_root())
        std::cout << tests.size() - failures << '/' << tests.size() << " groups passed on "
                  << comm.nprocs << " rank(s)\n";
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
