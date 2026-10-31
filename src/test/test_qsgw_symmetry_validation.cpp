#include "../core/meanfield_mpi.h"
#include "../core/qsgw/projection_target.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

using namespace librpa_int;
using qsgw::validate_live_symmetry_state;

void require(const bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

struct InversionModel
{
    PeriodicBoundaryData pbc;
    AtomicBasis basis{std::vector<std::size_t>{1, 1}};
    SymmetryContext symmetry;

    InversionModel()
    {
        pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        basis.set_l_shells({{0}, {0}});
        symmetry.set_crystal_structure(
            pbc.latvec, pbc.G, {{0, 0}, {1, 0}},
            {{0, {0.0, 0.0, 0.0}}, {1, {0.25, 0.25, 0.25}}});
        symmetry.basis_convention = {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL,
                                     LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M};
        SymmetryOperation inversion;
        inversion.rotation = {-1, 0, 0, 0, -1, 0, 0, 0, -1};
        inversion.translation = {0.25, 0.25, 0.25};
        inversion.use_row_convention = true;
        symmetry.set_rspace_operations({SpaceGroupSymOp::IDENTITY, inversion});
        symmetry.set_available();
    }

    SymmetryContext explicit_spin_symmetry() const
    {
        auto result = symmetry;
        SymmetrySpinOperation identity, inversion;
        inversion.spatial_id = 1;
        result.set_symmetry_spin_operations({identity, inversion}, false);
        return result;
    }

    MeanField meanfield(const double angle, const int components = 1,
                        const int spins = 1) const
    {
        MeanField result(spins, 1, 2 * components, 2, components);
        const double c = std::cos(angle), s = std::sin(angle);
        for (int spin = 0; spin < spins; ++spin)
        {
            for (int band = 0; band < 2 * components; ++band)
            {
                result.get_eigenvals()[spin](0, band) = band % 2 == 0 ? -1.0 : 1.0;
                result.get_weight()[spin](0, band) = band % 2 == 0 ? 1.0 : 0.0;
            }
            for (int component = 0; component < components; ++component)
            {
                ComplexMatrix coefficients(2 * components, 2);
                coefficients.zero_out();
                coefficients(2 * component, 0) = c;
                coefficients(2 * component, 1) = s;
                coefficients(2 * component + 1, 0) = -s;
                coefficients(2 * component + 1, 1) = c;
                result.get_eigenvectors()[spin][component][0] = std::move(coefficients);
            }
        }
        return result;
    }
};

void test_numerical_frames()
{
    const InversionModel model;
    // A scalar full-grid state need not be inversion covariant. Validation
    // must preserve its eigenvalues, occupations and coefficients.
    const auto state = model.meanfield(0.31);
    const auto before = state;
    validate_live_symmetry_state(state, model.basis, model.pbc, model.symmetry, false);
    for (int band = 0; band < 2; ++band)
    {
        require(state.get_eigenvals()[0](0, band) == before.get_eigenvals()[0](0, band) &&
                    state.get_weight()[0](0, band) == before.get_weight()[0](0, band),
                "validation changed E/f");
        for (int ao = 0; ao < 2; ++ao)
            require((*state.find_wfc(0, 0, 0))(band, ao) ==
                        (*before.find_wfc(0, 0, 0))(band, ao),
                    "validation changed C");
    }

    const auto covariant = model.meanfield(std::acos(-1.0) / 4.0);
    validate_symmetry_band_weights(covariant, model.basis, model.pbc, model.symmetry);
    validate_live_symmetry_state(covariant, model.basis, model.pbc,
                                 model.explicit_spin_symmetry(), false);
    validate_live_symmetry_state(covariant, model.basis, model.pbc, model.symmetry, true);
    const auto covariant_spinor = model.meanfield(std::acos(-1.0) / 4.0, 2);
    validate_live_symmetry_state(covariant_spinor, model.basis, model.pbc, model.symmetry, false);

    // Inversion exchanges the two AOs; their symmetric combination is a
    // closed one-dimensional retained frame.
    MeanField rectangular(1, 1, 1, 2);
    ComplexMatrix coefficients(1, 2);
    coefficients(0, 0) = std::sqrt(0.5);
    coefficients(0, 1) = std::sqrt(0.5);
    rectangular.get_eigenvectors()[0][0][0] = std::move(coefficients);
    rectangular.get_eigenvals()[0](0, 0) = -1.0;
    rectangular.get_weight()[0](0, 0) = 1.0;
    validate_live_symmetry_state(rectangular, model.basis, model.pbc, model.symmetry, false);
}

} // namespace

int main()
{
    try
    {
        test_numerical_frames();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "QSGW live symmetry validation controls passed\n";
    return 0;
}
