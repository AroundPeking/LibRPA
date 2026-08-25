#include "../bz_sampling_utils.h"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

void test_fractional_coordinates_override_consistent_cartesian_roundoff()
{
    const std::array<double, 9> reciprocal{
        -0.9235933162843867, 0.9235933162843867, 0.9235933162843867,
         0.9235933162843867,-0.9235933162843867, 0.9235933162843867,
         0.9235933162843867, 0.9235933162843867,-0.9235933162843867,
    };
    const std::vector<std::array<double, 3>> fractional{{0.25, 0.0, 0.0}};
    const std::vector<double> reported{
        -0.2308970243956508, 0.2308970243956508, 0.2308970243956508,
    };

    const auto result = driver::canonical_bz_kvectors_from_fractional(
        fractional, reported, reciprocal, 1.0e-5);

    assert(std::abs(result.kvectors[0] + 0.2308983290710967) < 1.0e-15);
    assert(std::abs(result.kvectors[1] - 0.2308983290710967) < 1.0e-15);
    assert(std::abs(result.kvectors[2] - 0.2308983290710967) < 1.0e-15);
    assert(std::abs(result.maximum_reported_difference - 1.3046754459e-6) < 1.0e-15);
}

void test_inconsistent_cartesian_column_is_rejected()
{
    const std::array<double, 9> reciprocal{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<std::array<double, 3>> fractional{{0.25, 0.0, 0.0}};
    const std::vector<double> reported{0.20, 0.0, 0.0};

    bool rejected = false;
    try
    {
        driver::canonical_bz_kvectors_from_fractional(
            fractional, reported, reciprocal, 1.0e-5);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main()
{
    test_fractional_coordinates_override_consistent_cartesian_roundoff();
    test_inconsistent_cartesian_column_is_rejected();
}
