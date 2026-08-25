#ifndef LIBRPA_DRIVER_BZ_SAMPLING_UTILS_H
#define LIBRPA_DRIVER_BZ_SAMPLING_UTILS_H

#include <array>
#include <vector>

namespace driver
{

struct CanonicalBzKvectors
{
    std::vector<double> kvectors;
    double maximum_reported_difference = 0.0;
};

CanonicalBzKvectors canonical_bz_kvectors_from_fractional(
    const std::vector<std::array<double, 3>> &fractional,
    const std::vector<double> &reported_cartesian,
    const std::array<double, 9> &reciprocal_cartesian,
    double consistency_tolerance);

}  // namespace driver

#endif
