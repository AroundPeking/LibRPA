#include "../optics.h"
#include "../constants.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <fstream>
#include <string>
#include <vector>

#include "testutils.h"

namespace
{

matrix_m<std::complex<double>> make_diag_tensor(const std::complex<double>& xx,
                                                const std::complex<double>& yy,
                                                const std::complex<double>& zz)
{
    matrix_m<std::complex<double>> tensor(3, 3, MAJOR::COL);
    tensor.zero_out();
    tensor(0, 0) = xx;
    tensor(1, 1) = yy;
    tensor(2, 2) = zz;
    return tensor;
}

} // namespace

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

    const auto real_mesh = build_real_axis_mesh(2.0, 0.5);
    assert(real_mesh.size() == 5);
    assert(fequal(real_mesh.front(), 0.0));
    assert(fequal(real_mesh[1], 0.5));
    assert(fequal(real_mesh.back(), 2.0));
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
    const auto continued = continue_scalar_series_pade(imag_axis, data, real_mesh, 6, 0.5);
    assert(continued.size() == real_mesh.size());
    assert(fequal(continued[1], -1.0 / (complex<double>{1.0, 0.5} - pole),
                  complex<double>{1e-10, 0.0}));
}

void test_optics_writers()
{
    using LIBRPA::optics::ImagAxisOpticsData;
    using LIBRPA::optics::RealAxisOpticsData;
    using LIBRPA::optics::read_imag_axis_data;
    using LIBRPA::optics::write_imag_axis_data;
    using LIBRPA::optics::write_real_axis_data;
    using LIBRPA::optics::write_absorption_spectrum;
    using LIBRPA::optics::write_summary;
    using std::complex;
    using std::string;

    ImagAxisOpticsData imag_data;
    imag_data.imag_freqs = {0.5};
    imag_data.epsinv00 = {complex<double>{0.25, 0.0}};
    imag_data.epsm_avg = {complex<double>{4.0, 0.0}};

    matrix_m<complex<double>> imag_tensor(3, 3, MAJOR::COL);
    imag_tensor.zero_out();
    imag_tensor(0, 0) = {1.0, 0.0};
    imag_tensor(1, 1) = {2.0, 0.0};
    imag_tensor(2, 2) = {3.0, 0.0};
    imag_data.epsm_tensor.push_back(imag_tensor);
    imag_data.head_tensor.push_back(imag_tensor);

    RealAxisOpticsData real_data;
    real_data.real_freqs = {0.0};
    real_data.epsm_avg = {complex<double>{12.0, 0.5}};
    real_data.epsm_tensor.push_back(imag_tensor);

    const string imag_file = "/tmp/test_optics_imag_axis.dat";
    const string real_file = "/tmp/test_optics_real_axis.dat";
    const string spectrum_file = "/tmp/test_optics_spectrum.dat";
    const string summary_file = "/tmp/test_optics_summary.txt";

    write_imag_axis_data(imag_file, imag_data);
    write_real_axis_data(real_file, real_data);
    write_absorption_spectrum(spectrum_file, real_data);
    write_summary(summary_file, imag_data, real_data, "pade", 6, 0.1);

    std::ifstream ifs_imag(imag_file);
    std::ifstream ifs_real(real_file);
    std::ifstream ifs_spectrum(spectrum_file);
    std::ifstream ifs_summary(summary_file);
    assert(ifs_imag.good());
    assert(ifs_real.good());
    assert(ifs_spectrum.good());
    assert(ifs_summary.good());

    string imag_header;
    string real_header;
    string spectrum_header;
    string summary_line;
    std::getline(ifs_imag, imag_header);
    std::getline(ifs_real, real_header);
    std::getline(ifs_spectrum, spectrum_header);
    std::getline(ifs_summary, summary_line);

    assert(imag_header.find("epsm_xx_re") != string::npos);
    assert(real_header.find("omega_real_ha") != string::npos);
    assert(spectrum_header.find("alpha_cm_inv") != string::npos);
    assert(summary_line.find("task absorption") != string::npos);

    const auto roundtrip = read_imag_axis_data(imag_file);
    assert(roundtrip.imag_freqs.size() == imag_data.imag_freqs.size());
    assert(fequal(roundtrip.epsinv00[0], imag_data.epsinv00[0]));
    assert(fequal(roundtrip.epsm_avg[0], imag_data.epsm_avg[0]));
    assert(fequal(roundtrip.epsm_tensor[0](1, 1), imag_data.epsm_tensor[0](1, 1)));
    assert(fequal(roundtrip.head_tensor[0](2, 2), imag_data.head_tensor[0](2, 2)));
}

void test_optics_writer_roundtrip_preserves_pade_continuation()
{
    using LIBRPA::optics::ImagAxisOpticsData;
    using LIBRPA::optics::build_real_axis_mesh;
    using LIBRPA::optics::continue_to_real_axis;
    using LIBRPA::optics::read_imag_axis_data;
    using LIBRPA::optics::write_imag_axis_data;
    using std::complex;
    using std::string;

    ImagAxisOpticsData imag_data;
    complex<double> pole_1{3.45, 0.28};
    complex<double> pole_2{4.25, 0.45};
    double freq = 0.02;
    for (int i = 0; i < 18; ++i)
    {
        const complex<double> x{0.0, freq};
        const complex<double> value =
            complex<double>{1.0, 0.0} - 0.8 / (x - pole_1) - 1.2 / (x - pole_2);
        imag_data.imag_freqs.push_back(freq);
        imag_data.epsm_avg.push_back(value);
        imag_data.epsinv00.push_back(1.0 / value);
        imag_data.epsm_tensor.push_back(make_diag_tensor(value, 1.1 * value, 0.9 * value));
        imag_data.head_tensor.push_back(make_diag_tensor(0.1 * value, 0.2 * value, 0.3 * value));
        freq *= 1.45;
    }

    const auto real_mesh = build_real_axis_mesh(8.0 / HA2EV, 0.05 / HA2EV);
    const auto reference = continue_to_real_axis(
        imag_data, real_mesh, static_cast<int>(imag_data.imag_freqs.size()), 0.0);

    const string imag_file = "/tmp/test_optics_imag_axis_roundtrip.dat";
    write_imag_axis_data(imag_file, imag_data);
    const auto roundtrip = read_imag_axis_data(imag_file);
    const auto reloaded = continue_to_real_axis(
        roundtrip, real_mesh, static_cast<int>(roundtrip.imag_freqs.size()), 0.0);

    assert(reference.epsm_avg.size() == reloaded.epsm_avg.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < reference.epsm_avg.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(reference.epsm_avg[i] - reloaded.epsm_avg[i]));
    }
    assert(max_diff < 1e-10);
}

void test_optics_build_imag_axis_data()
{
    using LIBRPA::optics::build_imag_axis_data;
    using std::complex;
    using std::vector;

    vector<double> imag_freqs{1.0, 2.0};
    vector<matrix_m<complex<double>>> epsinv_mats;
    vector<matrix_m<complex<double>>> epsm_tensors;
    vector<matrix_m<complex<double>>> head_tensors;

    matrix_m<complex<double>> epsinv0(2, 2, MAJOR::COL);
    epsinv0.zero_out();
    epsinv0(0, 0) = {0.25, 0.0};
    epsinv_mats.push_back(epsinv0);

    matrix_m<complex<double>> epsinv1(2, 2, MAJOR::COL);
    epsinv1.zero_out();
    epsinv1(0, 0) = {0.5, 0.0};
    epsinv_mats.push_back(epsinv1);

    epsm_tensors.push_back(make_diag_tensor({4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}));
    epsm_tensors.push_back(make_diag_tensor({7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}));
    head_tensors.push_back(make_diag_tensor({1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}));
    head_tensors.push_back(make_diag_tensor({4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}));

    const auto data =
        build_imag_axis_data(imag_freqs, epsinv_mats, epsm_tensors, head_tensors);

    epsm_tensors[0].clear();
    head_tensors[0].clear();

    assert(data.imag_freqs.size() == 2);
    assert(fequal(data.epsinv00[0], complex<double>{0.25, 0.0}));
    assert(fequal(data.epsm_avg[0], complex<double>{4.0, 0.0}));
    assert(data.epsm_tensor[0].nr() == 3 && data.epsm_tensor[0].nc() == 3);
    assert(data.head_tensor[0].nr() == 3 && data.head_tensor[0].nc() == 3);
    assert(fequal(data.epsm_tensor[1](1, 1), complex<double>{8.0, 0.0}));
    assert(fequal(data.head_tensor[1](2, 2), complex<double>{6.0, 0.0}));
}

void test_optics_continue_to_real_axis()
{
    using LIBRPA::optics::ImagAxisOpticsData;
    using LIBRPA::optics::continue_to_real_axis;
    using std::complex;
    using std::vector;

    const complex<double> pole{2.0, 2.0};

    ImagAxisOpticsData imag_data;
    for (int i = 1; i <= 6; ++i)
    {
        const double freq = static_cast<double>(i);
        const complex<double> x{0.0, freq};
        const complex<double> value = -1.0 / (x - pole);

        imag_data.imag_freqs.push_back(freq);
        imag_data.epsinv00.push_back(1.0 / value);
        imag_data.epsm_avg.push_back(value);
        imag_data.epsm_tensor.push_back(
            make_diag_tensor(value, 2.0 * value, 3.0 * value));
        imag_data.head_tensor.push_back(
            make_diag_tensor(4.0 * value, 5.0 * value, 6.0 * value));
    }

    const vector<double> real_mesh{0.0, 1.0, 2.0};
    const auto real_data = continue_to_real_axis(imag_data, real_mesh, 6, 0.5);

    assert(real_data.real_freqs == real_mesh);
    assert(fequal(real_data.epsm_avg[1], -1.0 / (complex<double>{1.0, 0.5} - pole),
                  complex<double>{1e-10, 0.0}));
    assert(fequal(real_data.epsm_tensor[2](1, 1),
                  2.0 * (-1.0 / (complex<double>{2.0, 0.5} - pole)),
                  complex<double>{1e-10, 0.0}));
}

void test_optics_multipole_continuation()
{
    using LIBRPA::optics::ImagAxisOpticsData;
    using LIBRPA::optics::continue_to_real_axis_multipole;
    using std::complex;
    using std::vector;

    constexpr double eps_inf = 1.25;
    constexpr double strength = 6.0;
    constexpr double omega0 = 2.5;
    constexpr double gamma0 = 0.7;

    ImagAxisOpticsData imag_data;
    for (int i = 0; i <= 16; ++i)
    {
        const double u = 0.25 * static_cast<double>(i);
        const double epsm_real = eps_inf + strength / (omega0 * omega0 + gamma0 * u + u * u);
        imag_data.imag_freqs.push_back(u);
        imag_data.epsm_avg.push_back({epsm_real, 0.0});
        imag_data.epsinv00.push_back(1.0 / complex<double>{epsm_real, 0.0});
        imag_data.epsm_tensor.push_back(make_diag_tensor({epsm_real, 0.0}, {0.0, 0.0}, {0.0, 0.0}));
        imag_data.head_tensor.push_back(make_diag_tensor({0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}));
    }

    const vector<double> real_mesh{0.0, 1.0, 2.0, 3.0};
    const double eta = 0.0;
    const auto real_data = continue_to_real_axis_multipole(imag_data, real_mesh, 1, eta);

    assert(real_data.real_freqs == real_mesh);
    const complex<double> z{2.0, eta};
    const complex<double> expected =
        eps_inf + strength / (omega0 * omega0 - z * z - complex<double>{0.0, gamma0} * z);
    assert(std::abs(real_data.epsm_avg[2].imag()) > 1e-3);
    assert(fequal(real_data.epsm_avg[2], expected, complex<double>{5e-3, 0.0}));
}

int main()
{
    test_optics_scalar_helpers();
    test_optics_pade_continuation();
    test_optics_build_imag_axis_data();
    test_optics_continue_to_real_axis();
    test_optics_multipole_continuation();
    test_optics_writers();
    test_optics_writer_roundtrip_preserves_pade_continuation();
    return 0;
}
