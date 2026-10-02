#include "tetrahedron_quadrature.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace librpa_int
{
namespace
{
using GridKey = std::array<int, 3>;

int wrap_index(const int value, const int period)
{
    const int remainder = value % period;
    return remainder < 0 ? remainder + period : remainder;
}

std::map<GridKey, std::size_t> make_k_index(const Vector3_Order<int> &period)
{
    std::map<GridKey, std::size_t> result;
    std::size_t index = 0;
    for (int ix = 0; ix != period.x; ++ix)
        for (int iy = 0; iy != period.y; ++iy)
            for (int iz = 0; iz != period.z; ++iz) result.emplace(GridKey{ix, iy, iz}, index++);
    return result;
}
}  // namespace

std::vector<PeriodicTetrahedronQuadraturePoint> build_periodic_tetrahedron_quadrature(
    const Vector3_Order<int> &period)
{
    if (period.x < 2 || period.y < 2 || period.z < 2)
        throw std::invalid_argument(
            "three-dimensional tetrahedron quadrature requires at least two points per axis");

    static constexpr std::array<std::array<std::array<int, 3>, 4>, 6> tetra_vertices{
        {{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}},
         {{{0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {1, 0, 1}, {1, 0, 0}, {1, 1, 1}}}}};
    constexpr double alpha = 0.5854101966249685;
    constexpr double beta = 0.1381966011250105;
    static constexpr std::array<std::array<double, 4>, 4> barycentric{
        {{{alpha, beta, beta, beta}},
         {{beta, alpha, beta, beta}},
         {{beta, beta, alpha, beta}},
         {{beta, beta, beta, alpha}}}};

    const auto k_index = make_k_index(period);
    const std::size_t nk = static_cast<std::size_t>(period.x) * static_cast<std::size_t>(period.y) *
                           static_cast<std::size_t>(period.z);
    std::vector<PeriodicTetrahedronQuadraturePoint> result;
    result.reserve(nk * tetra_vertices.size() * barycentric.size());
    const double weight = 1.0 / (24.0 * static_cast<double>(nk));
    for (int ix = 0; ix != period.x; ++ix)
        for (int iy = 0; iy != period.y; ++iy)
            for (int iz = 0; iz != period.z; ++iz)
                for (const auto &tetra : tetra_vertices)
                {
                    std::array<std::size_t, 4> vertices{};
                    for (int vertex = 0; vertex != 4; ++vertex)
                    {
                        const auto &offset = tetra[vertex];
                        vertices[vertex] = k_index.at({wrap_index(ix + offset[0], period.x),
                                                       wrap_index(iy + offset[1], period.y),
                                                       wrap_index(iz + offset[2], period.z)});
                    }
                    for (const auto &bary : barycentric)
                    {
                        Vector3_Order<double> coordinate{0.0, 0.0, 0.0};
                        for (int vertex = 0; vertex != 4; ++vertex)
                        {
                            const auto &offset = tetra[vertex];
                            coordinate.x +=
                                bary[vertex] * static_cast<double>(ix + offset[0]) / period.x;
                            coordinate.y +=
                                bary[vertex] * static_cast<double>(iy + offset[1]) / period.y;
                            coordinate.z +=
                                bary[vertex] * static_cast<double>(iz + offset[2]) / period.z;
                        }
                        result.push_back({vertices, bary, coordinate, weight});
                    }
                }
    return result;
}

std::vector<PeriodicTetrahedronQuadraturePoint> build_periodic_tetrahedron_quadrature(
    const Vector3_Order<int> &period, const std::vector<Vector3_Order<double>> &kfrac_list)
{
    const std::size_t expected = static_cast<std::size_t>(period.x) *
                                 static_cast<std::size_t>(period.y) *
                                 static_cast<std::size_t>(period.z);
    if (period.x < 2 || period.y < 2 || period.z < 2 || kfrac_list.size() != expected)
        throw std::invalid_argument(
            "three-dimensional tetrahedron quadrature requires a complete grid with at least two "
            "points per axis");

    auto reduce_fractional = [](double value)
    {
        value -= std::floor(value);
        if (value >= 1.0 - 1.0e-12) value = 0.0;
        if (value < 0.0 && value > -1.0e-12) value = 0.0;
        return value;
    };
    std::array<std::vector<double>, 3> coordinates;
    for (int axis = 0; axis != 3; ++axis)
    {
        auto &values = coordinates[axis];
        values.reserve(kfrac_list.size());
        for (const auto &k : kfrac_list)
            values.push_back(reduce_fractional(axis == 0 ? k.x : axis == 1 ? k.y : k.z));
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(),
                                 [](const double lhs, const double rhs)
                                 { return std::abs(lhs - rhs) < 1.0e-10; }),
                     values.end());
        const int expected_axis = axis == 0 ? period.x : axis == 1 ? period.y : period.z;
        if (static_cast<int>(values.size()) != expected_axis)
            throw std::invalid_argument(
                "tetrahedron quadrature found an incomplete regular coordinate grid");
        for (int index = 0; index != expected_axis; ++index)
        {
            const double next =
                index + 1 == expected_axis ? values.front() + 1.0 : values[index + 1];
            if (std::abs((next - values[index]) - 1.0 / expected_axis) > 1.0e-8)
                throw std::invalid_argument(
                    "tetrahedron quadrature requires equally spaced fractional coordinates");
        }
    }

    std::vector<std::size_t> canonical_to_input(expected, expected);
    for (std::size_t input = 0; input != kfrac_list.size(); ++input)
    {
        const auto &k = kfrac_list[input];
        const std::array<double, 3> values{reduce_fractional(k.x), reduce_fractional(k.y),
                                           reduce_fractional(k.z)};
        std::array<int, 3> indices{};
        for (int axis = 0; axis != 3; ++axis)
        {
            const auto &axis_values = coordinates[axis];
            const auto iter =
                std::lower_bound(axis_values.begin(), axis_values.end(), values[axis]);
            if (iter == axis_values.end() || std::abs(*iter - values[axis]) > 1.0e-10)
                throw std::invalid_argument("tetrahedron quadrature found an off-grid k point");
            indices[axis] = static_cast<int>(iter - axis_values.begin());
        }
        const std::size_t canonical =
            (static_cast<std::size_t>(indices[0]) * static_cast<std::size_t>(period.y) +
             static_cast<std::size_t>(indices[1])) *
                static_cast<std::size_t>(period.z) +
            static_cast<std::size_t>(indices[2]);
        if (canonical_to_input[canonical] != expected)
            throw std::invalid_argument("tetrahedron quadrature found duplicate k points");
        canonical_to_input[canonical] = input;
    }
    for (const auto input : canonical_to_input)
        if (input == expected)
            throw std::invalid_argument("tetrahedron quadrature found a missing k point");

    auto result = build_periodic_tetrahedron_quadrature(period);
    for (auto &point : result)
        for (auto &vertex : point.vertices) vertex = canonical_to_input[vertex];
    return result;
}

std::complex<double> finite_temperature_tetrahedron_kernel(const double energy_n,
                                                           const double energy_m,
                                                           const double frequency,
                                                           const FermiDiracReference &reference)
{
    if (!reference.enabled)
        throw std::invalid_argument("tetrahedron kernel requires FD occupation metadata");
    const double delta = energy_n - energy_m;
    if (std::abs(delta) < 1.0e-14 && std::abs(frequency) < 1.0e-14)
        return {
            fermi_dirac_derivative(0.5 * (energy_n + energy_m) - reference.chemical_potential_ha,
                                   reference.kbt_ha),
            0.0};

    const double occupation_n =
        fermi_dirac_occupation(energy_n - reference.chemical_potential_ha, reference.kbt_ha);
    const double occupation_m =
        fermi_dirac_occupation(energy_m - reference.chemical_potential_ha, reference.kbt_ha);
    return (occupation_n - occupation_m) / std::complex<double>{delta, frequency};
}

}  // namespace librpa_int
