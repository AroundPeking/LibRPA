#include "bz_sampling_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace driver
{

CanonicalBzKvectors canonical_bz_kvectors_from_fractional(
    const std::vector<std::array<double, 3>> &fractional,
    const std::vector<double> &reported_cartesian,
    const std::array<double, 9> &reciprocal_cartesian,
    const double consistency_tolerance)
{
    if (reported_cartesian.size() != 3 * fractional.size())
    {
        throw std::runtime_error("BZ sampling fractional and Cartesian sizes do not match");
    }
    if (!std::isfinite(consistency_tolerance) || consistency_tolerance < 0.0)
    {
        throw std::runtime_error("Invalid BZ sampling consistency tolerance");
    }
    for (const double value : reciprocal_cartesian)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("BZ sampling reciprocal lattice contains an invalid number");
        }
    }

    CanonicalBzKvectors result;
    result.kvectors.resize(reported_cartesian.size());
    for (std::size_t ik = 0; ik != fractional.size(); ++ik)
    {
        const auto &kfrac = fractional[ik];
        for (const double value : kfrac)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("BZ sampling fractional coordinate is invalid");
            }
        }
        for (int component = 0; component != 3; ++component)
        {
            const double canonical =
                kfrac[0] * reciprocal_cartesian[component]
                + kfrac[1] * reciprocal_cartesian[3 + component]
                + kfrac[2] * reciprocal_cartesian[6 + component];
            const auto index = 3 * ik + static_cast<std::size_t>(component);
            const double difference = std::abs(canonical - reported_cartesian[index]);
            result.maximum_reported_difference =
                std::max(result.maximum_reported_difference, difference);
            result.kvectors[index] = canonical;
        }
    }
    if (result.maximum_reported_difference > consistency_tolerance)
    {
        throw std::runtime_error(
            "BZ sampling fractional and Cartesian coordinates are inconsistent");
    }
    return result;
}

}  // namespace driver
