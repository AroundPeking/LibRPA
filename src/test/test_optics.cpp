#include "../optics.h"

#include <cassert>
#include <complex>
#include <vector>

#include "testutils.h"

void test_optics_scalar_helpers()
{
    using LIBRPA::optics::build_imaginary_axis_points;
    using LIBRPA::optics::build_real_axis_mesh;
    using LIBRPA::optics::invert_epsinv00;
    using std::complex;
    using std::vector;

    const complex<double> epsinv00{0.25, 0.0};
    const complex<double> epsm = invert_epsinv00(epsinv00);
    assert(fequal(epsm, complex<double>{4.0, 0.0}));

    const vector<double> imag_freqs{0.5, 1.0, 2.0, 4.0};
    const auto imag_axis = build_imaginary_axis_points(imag_freqs);
    assert(imag_axis.size() == imag_freqs.size());
    assert(fequal(imag_axis[0], complex<double>{0.0, 0.5}));
    assert(fequal(imag_axis.back(), complex<double>{0.0, 4.0}));

    const auto real_mesh = build_real_axis_mesh(imag_freqs, 5);
    assert(real_mesh.size() == 5);
    assert(fequal(real_mesh.front(), 0.0));
    assert(fequal(real_mesh.back(), 4.0));
}

void test_optics_pade_continuation()
{
    using LIBRPA::optics::continue_scalar_series_pade;
    using std::complex;
    using std::vector;

    const complex<double> pole{2.0, 2.0};
    vector<complex<double>> imag_axis;
    vector<complex<double>> data;
    for (int i = 1; i <= 6; ++i)
    {
        const complex<double> x{0.0, static_cast<double>(i)};
        imag_axis.push_back(x);
        data.push_back(-1.0 / (x - pole));
    }

    const vector<double> real_mesh{0.0, 1.0, 2.0};
    const auto continued = continue_scalar_series_pade(imag_axis, data, real_mesh, 6);
    assert(continued.size() == real_mesh.size());
    assert(fequal(continued[1], -1.0 / (complex<double>{1.0, 0.0} - pole),
                  complex<double>{1e-10, 0.0}));
}

int main()
{
    test_optics_scalar_helpers();
    test_optics_pade_continuation();
    return 0;
}
