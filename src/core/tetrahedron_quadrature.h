#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "pbc.h"
#include "thermal_occupation.h"

namespace librpa_int
{

struct PeriodicTetrahedronQuadraturePoint
{
    std::array<std::size_t, 4> vertices{};
    std::array<double, 4> barycentric{};
    Vector3_Order<double> fractional_coordinate{0.0, 0.0, 0.0};
    double weight = 0.0;
};

// Build a periodic six-tetrahedron decomposition with a symmetric degree-two
// simplex rule. The weights include the normalized Brillouin-zone volume.
std::vector<PeriodicTetrahedronQuadraturePoint> build_periodic_tetrahedron_quadrature(
    const Vector3_Order<int> &period);

// Same geometry, with vertex indices taken from an explicitly supplied
// complete regular grid.  The input order is not required to be lexicographic.
std::vector<PeriodicTetrahedronQuadraturePoint> build_periodic_tetrahedron_quadrature(
    const Vector3_Order<int> &period, const std::vector<Vector3_Order<double>> &kfrac_list);

// Finite-temperature Adler-Wiser kernel, including its equal-energy static
// limit. The frequency argument is a real Matsubara frequency in Hartree.
std::complex<double> finite_temperature_tetrahedron_kernel(double energy_n, double energy_m,
                                                           double frequency,
                                                           const FermiDiracReference &reference);

}  // namespace librpa_int
