#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/thermal_gw_screening.h"

namespace
{
using librpa_int::ComplexMatrix;
using librpa_int::screen_thermal_gw;
using librpa_int::thermal_gw_screening_to_time;
using librpa_int::ThermalGWScreening;
using librpa_int::ThermalGWTransform;
using Complex = std::complex<double>;
using Mat2 = std::array<Complex, 4>;
const Complex I(0.0, 1.0);
const double PI = std::acos(-1.0);
const Mat2 ID{1.0, 0.0, 0.0, 1.0};
const Mat2 S{1.3, Complex(0.2, 0.15), Complex(0.2, -0.15), 0.8};
const Mat2 T{0.9, Complex(-0.1, 0.25), Complex(-0.1, -0.25), 1.4};
const Mat2 CHI{Complex(-0.4, 0.2), Complex(0.3, -0.1), Complex(-0.2, 0.45), Complex(-0.7, -0.15)};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(Complex actual, Complex expected, double tolerance, const std::string& message)
{
    require(std::isfinite(actual.real()) && std::isfinite(actual.imag()), "nonfinite result");
    require(std::abs(actual - expected) <= tolerance, message);
}

template <class Exception = std::invalid_argument>
void reject(const std::function<void()>& operation, const std::string& message = "")
{
    try
    {
        operation();
    }
    catch (const Exception& exception)
    {
        require(std::string(exception.what()).find(message) != std::string::npos,
                "exception lacks expected context: " + message + ": " + exception.what());
        return;
    }
    throw std::runtime_error("operation did not reject invalid input");
}

ComplexMatrix matrix(const Mat2& values)
{
    ComplexMatrix result(2, 2);
    for (int k = 0; k < 4; ++k) result.c[k] = values[k];
    return result;
}

// Independent fixed-size algebra: no library product, inverse or solve in the oracle.
Mat2 product(const Mat2& a, const Mat2& b)
{
    return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3]};
}

Mat2 inverse(const Mat2& a)
{
    const Complex det = a[0] * a[3] - a[1] * a[2];
    require(std::abs(det) > 1e-12, "singular analytic fixture");
    return {a[3] / det, -a[1] / det, -a[2] / det, a[0] / det};
}

Mat2 epsilon(const Mat2& chi, const Mat2& s)
{
    auto result = product(product(s, chi), s);
    for (int k = 0; k < 4; ++k) result[k] = ID[k] - result[k];
    return result;
}

Mat2 reference(const Mat2& chi, const Mat2& s = S, const Mat2& t = T)
{
    auto delta = inverse(epsilon(chi, s));
    for (int k = 0; k < 4; ++k) delta[k] -= ID[k];
    return product(product(t, delta), t);
}

void row_close(const ComplexMatrix& actual, int row, const Mat2& expected, double tolerance = 3e-14)
{
    for (int k = 0; k < 4; ++k)
        close(actual(row, k), expected[k], tolerance, "ordered analytic 2x2 Wc mismatch");
}

void unchanged(const ComplexMatrix& actual, const ComplexMatrix& before)
{
    require(actual.nr == before.nr && actual.nc == before.nc && actual.size == before.size,
            "caller matrix metadata mutated");
    for (int k = 0; k < actual.size; ++k)
        close(actual.c[k], before.c[k], 0.0, "caller matrix data mutated");
}

void check_general_complex_solve()
{
    // Negative frequency is deliberately unrelated to the positive one.
    const Mat2 negative{Complex(-0.1, -0.6), Complex(0.5, 0.3), Complex(-0.4, 0.2),
                        Complex(-0.8, 0.1)};
    std::vector<ComplexMatrix> chi{matrix(CHI), matrix(negative), matrix(Mat2{})};
    const auto before = chi;
    const auto s = matrix(S), t = matrix(T), s_before = s, t_before = t;
    const auto result = screen_thermal_gw(5.0, {3, -3, 0}, chi, s, t);
    require(result.beta_ha_inv == 5.0 && result.bosonic_indices == std::vector<int>({3, -3, 0}),
            "screening changed beta or signed order");
    require(result.wc_frequency.nr == 3 && result.wc_frequency.nc == 4, "packed Wc shape");
    row_close(result.wc_frequency, 0, reference(CHI));
    row_close(result.wc_frequency, 1, reference(negative));
    row_close(result.wc_frequency, 2, Mat2{}, 0.0);
    require(std::abs(product(S, T)[1] - product(T, S)[1]) > 0.1,
            "fixture must use noncommuting roots");
    const auto a = epsilon(CHI, S);
    require(std::abs(a[1] - a[2]) > 0.1 && std::abs(a[1] - std::conj(a[2])) > 0.1,
            "fixture must be nonsymmetric and non-Hermitian");
    // Recover X independently from Wc and check the defining equation, not just its inverse.
    Mat2 wc;
    for (int k = 0; k < 4; ++k) wc[k] = result.wc_frequency(0, k);
    auto x = product(inverse(T), wc);
    for (int k = 0; k < 4; ++k) x[k] += T[k];
    const auto ax = product(a, x);
    for (int k = 0; k < 4; ++k) close(ax[k], T[k], 4e-14, "epsilon X = T residual");
    for (int m = 0; m < 3; ++m) unchanged(chi[m], before[m]);
    unchanged(s, s_before);
    unchanged(t, t_before);
}

void check_pivoted_solve()
{
    const Mat2 a{0.0, Complex(2.0, 1.0), Complex(1.0, -2.0), Complex(3.0, 1.0)};
    Mat2 chi;
    for (int k = 0; k < 4; ++k) chi[k] = ID[k] - a[k];
    const auto result = screen_thermal_gw(2.0, {-2}, {matrix(chi)}, matrix(ID), matrix(T));
    row_close(result.wc_frequency, 0, reference(chi, ID));
    require(result.bosonic_indices == std::vector<int>{-2}, "invented missing positive mode");
}

void check_independent_signed_poles()
{
    const double beta = 6.0;
    const std::vector<int> labels{2, -2, 0, -1, 1};
    const std::array<Complex, 2> v{Complex(0.6, 0.2), Complex(-0.3, 0.5)};
    const std::array<Complex, 2> u{Complex(0.2, -0.4), Complex(0.7, 0.1)};
    std::vector<ComplexMatrix> chi;
    std::vector<Mat2> expected;
    for (int m : labels)
    {
        const Complex z = I * (2.0 * PI * m / beta);
        Mat2 sample;
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                sample[2 * r + c] =
                    v[r] * std::conj(v[c]) / (z - 0.7) - 0.8 * u[r] * std::conj(u[c]) / (z + 1.1);
        chi.push_back(matrix(sample));
        expected.push_back(reference(sample));
    }
    const auto result = screen_thermal_gw(beta, labels, chi, matrix(S), matrix(T));
    for (int m = 0; m < 5; ++m) row_close(result.wc_frequency, m, expected[m]);
    for (const auto& pair : {std::pair<int, int>{0, 1}, {4, 3}})
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                close(result.wc_frequency(pair.second, 2 * r + c),
                      std::conj(result.wc_frequency(pair.first, 2 * c + r)), 3e-14,
                      "independently evaluated signed poles violate adjoint relation");
    require(std::abs(result.wc_frequency(0, 1) - std::conj(result.wc_frequency(0, 2))) > 1e-3,
            "pole model must not be same-frequency Hermitian");
    require(std::abs(result.wc_frequency(1, 1) - std::conj(result.wc_frequency(0, 1))) > 1e-3,
            "elementwise conjugation must differ from adjoint");
}

void check_static_normalization()
{
    const double beta = 7.0;
    const std::vector<int> labels{1, 0, -1};
    const Mat2 static_chi{-0.4, Complex(0.1, 0.2), Complex(0.1, -0.2), -0.6};
    const Mat2 amplitude = reference(static_chi);
    const auto screened = screen_thermal_gw(
        beta, labels, {matrix(Mat2{}), matrix(static_chi), matrix(Mat2{})}, matrix(S), matrix(T));
    const auto transform = ThermalGWTransform::from_quadrature(
        beta, {0.2, 1.1, 2.4, 4.8, 6.7}, {1.0, 1.0, 2.0, 2.0, 1.0}, labels, {-1, 0});
    const auto time = thermal_gw_screening_to_time(screened, transform);
    require(time.nr == 5 && time.nc == 4, "N_tau != N_boson rectangular output");
    for (int j = 0; j < time.nr; ++j)
        for (int k = 0; k < 4; ++k)
            close(time(j, k), amplitude[k] / beta, 5e-15, "static A delta_m0 must give A/beta");
}

void check_external_transform_and_order()
{
    const std::vector<int> labels{4, -2, 0};
    const auto screened = screen_thermal_gw(3.0, labels, {matrix(CHI), matrix(Mat2{}), matrix(ID)},
                                            matrix(S), matrix(T));
    const auto before = screened.wc_frequency;
    ComplexMatrix b(2, 3), f(1, 2);
    for (int k = 0; k < b.size; ++k) b.c[k] = Complex(0.1 * k - 0.2, 0.15 * k + 0.3);
    const ThermalGWTransform transform(3.0, {0.4, 2.2}, labels, {0}, b, f);
    const auto time = thermal_gw_screening_to_time(screened, transform);
    require(time.nr == 2 && time.nc == 4, "external B rectangular result");
    const std::array<Mat2, 3> expected{reference(CHI), Mat2{}, reference(ID)};
    for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 4; ++k)
        {
            Complex sum = 0.0;
            for (int m = 0; m < 3; ++m) sum += b(j, m) * expected[m][k];
            close(time(j, k), sum, 3e-14, "B coefficients must be applied verbatim once");
        }
    unchanged(screened.wc_frequency, before);
    const auto shuffled = screen_thermal_gw(
        3.0, {0, 4, -2}, {matrix(ID), matrix(CHI), matrix(Mat2{})}, matrix(S), matrix(T));
    row_close(shuffled.wc_frequency, 0, expected[2]);
    row_close(shuffled.wc_frequency, 1, expected[0]);
    row_close(shuffled.wc_frequency, 2, expected[1]);
    reject([&] { thermal_gw_screening_to_time(shuffled, transform); });
    auto bad = screened;
    bad.beta_ha_inv = std::nextafter(3.0, 4.0);
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    bad = screened;
    bad.bosonic_indices = {4, -3, 0};
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    bad.bosonic_indices = {4, 0};
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
}

void check_grid_rejections()
{
    const auto s = matrix(S), t = matrix(T);
    const std::vector<ComplexMatrix> chi{matrix(CHI)};
    for (double beta :
         {0.0, -1.0, std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::denorm_min()})
        reject([&] { screen_thermal_gw(beta, {0}, chi, s, t); });
    reject([&] { screen_thermal_gw(2.0, {}, {}, s, t); });
    reject([&] { screen_thermal_gw(2.0, {0}, {}, s, t); });
    reject([&] { screen_thermal_gw(2.0, {0, 1}, chi, s, t); });
    reject([&] { screen_thermal_gw(2.0, {0, 0}, {chi[0], chi[0]}, s, t); });
    reject([&] { screen_thermal_gw(1e-308, {1}, chi, s, t); });
    const std::vector<int> extreme{std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::min()};
    const auto result = screen_thermal_gw(2.0, extreme, {chi[0], chi[0]}, s, t);
    require(result.bosonic_indices == extreme, "extreme signed integer labels changed");
}

void check_matrix_rejections()
{
    const auto s = matrix(S), t = matrix(T);
    const std::vector<ComplexMatrix> good{matrix(CHI)};
    for (const auto& bad : {ComplexMatrix(), ComplexMatrix(2, 3), ComplexMatrix(1, 1)})
    {
        reject([&] { screen_thermal_gw(2.0, {0}, good, bad, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, good, s, bad); });
        reject([&] { screen_thermal_gw(2.0, {0}, {bad}, s, t); });
    }
    for (int corruption = 0; corruption < 5; ++corruption)
    {
        std::vector<ComplexMatrix> chi{matrix(CHI)};
        auto bad_s = s, bad_t = t;
        const auto corrupt = [&](ComplexMatrix& m)
        {
            if (corruption == 0) m.size = 3;
            if (corruption == 1) m.nr = -2;
            if (corruption == 2) m.nc = 0;
            if (corruption == 3) m.nr = m.nc = std::numeric_limits<int>::max();
            if (corruption == 4)
            {
                m = ComplexMatrix();
                m.nr = m.nc = 2;
                m.size = 4;
            }
        };
        corrupt(chi[0]);
        corrupt(bad_s);
        corrupt(bad_t);
        reject([&] { screen_thermal_gw(2.0, {0}, chi, s, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, good, bad_s, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, good, s, bad_t); });
    }
    auto mixed = good;
    mixed.emplace_back(1, 1);
    reject([&] { screen_thermal_gw(2.0, {0, -1}, mixed, s, t); });
}

void check_nonfinite_rejections()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const auto s = matrix(S), t = matrix(T);
    for (Complex value :
         {Complex(nan, 0.0), Complex(0.0, nan), Complex(inf, 0.0), Complex(0.0, -inf)})
    {
        auto bad_s = s, bad_t = t;
        std::vector<ComplexMatrix> chi{matrix(CHI)};
        chi[0](1, 0) = value;
        bad_s(1, 0) = value;
        bad_t(1, 0) = value;
        reject([&] { screen_thermal_gw(2.0, {0}, chi, s, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, bad_s, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, s, bad_t); });
    }
}

void check_hermitian_roots()
{
    const auto s = matrix(S), t = matrix(T);
    for (bool diagonal : {false, true})
    {
        auto bad_s = s, bad_t = t;
        bad_s(0, diagonal ? 0 : 1) += I * 1e-5;
        bad_t(0, diagonal ? 0 : 1) += I * 1e-5;
        reject([&] { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, bad_s, t); });
        reject([&] { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, s, bad_t); });
    }
    auto tiny = s;
    for (int k = 0; k < 4; ++k) tiny.c[k] *= 1e-200;
    tiny(0, 1) += I * 1e-205;
    reject([&] { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, tiny, t); });
    const auto zero_s = screen_thermal_gw(2.0, {0}, {matrix(CHI)}, matrix(Mat2{}), t);
    row_close(zero_s.wc_frequency, 0, Mat2{}, 0.0);
    const Mat2 rank_one{1.0, 0.0, 0.0, 0.0};
    const auto rank_t = screen_thermal_gw(2.0, {0}, {matrix(CHI)}, s, matrix(rank_one));
    row_close(rank_t.wc_frequency, 0, reference(CHI, S, rank_one));
}

void check_singular_and_conditioning()
{
    const auto id = matrix(ID), t = matrix(T);
    std::vector<ComplexMatrix> chi{matrix(ID)};
    const auto before = chi;
    reject<std::runtime_error>([&] { screen_thermal_gw(2.0, {-7}, chi, id, t); }, "INFO");
    unchanged(chi[0], before[0]);
    const auto check_bad = [&](const Mat2& a)
    {
        Mat2 bad_chi;
        for (int k = 0; k < 4; ++k) bad_chi[k] = ID[k] - a[k];
        // Even a zero physical RHS must not hide a singular dielectric.
        for (const auto& root : {t, matrix(Mat2{})})
            reject<std::runtime_error>(
                [&] { screen_thermal_gw(2.0, {-7}, {matrix(bad_chi)}, id, root); }, "rcond");
    };
    check_bad(Mat2{1.0, 0.0, 0.0, 1e-16});
    // Pivots and eigenvalues are all 1, but the infinity condition number is ~1e16.
    check_bad(Mat2{1.0, 1e8, 0.0, 1.0});
    const Mat2 moderate_chi{0.0, -1e5, 0.0, 0.0};
    const auto moderate = screen_thermal_gw(2.0, {0}, {matrix(moderate_chi)}, id, id);
    row_close(moderate.wc_frequency, 0, Mat2{0.0, -1e5, 0.0, 0.0}, 1e-10);
}

void check_scale_and_overflow()
{
    // Extreme uniform scales are not, by themselves, poor conditioning.
    for (double scale : {1e-200, 1e200})
    {
        ComplexMatrix chi(1, 1), s(1, 1), t(1, 1);
        chi(0, 0) = 1.0 - I * scale;
        s(0, 0) = 1.0;
        t(0, 0) = std::sqrt(scale);
        const auto result = screen_thermal_gw(2.0, {0}, {chi}, s, t);
        close(result.wc_frequency(0, 0) / std::max(1.0, scale),
              Complex(-scale, -1.0) / std::max(1.0, scale), 3e-15, "scale-aware solve or residual");
    }
    const auto id = matrix(ID);
    auto enormous_s = id, enormous_t = id;
    enormous_s(0, 0) = 1e200;
    enormous_t(0, 0) = 1e200;
    reject<std::overflow_error>([&]
                                { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, enormous_s, id); });
    reject<std::overflow_error>([&]
                                { screen_thermal_gw(2.0, {0}, {matrix(CHI)}, id, enormous_t); });
}

void check_tiny_response()
{
    const double tiny = 1e-18;
    ComplexMatrix scalar_chi(1, 1), root(1, 1);
    scalar_chi(0, 0) = -tiny;
    root(0, 0) = 1.0;
    const auto scalar = screen_thermal_gw(2.0, {0}, {scalar_chi}, root, root);
    close(scalar.wc_frequency(0, 0) / tiny, -1.0 / (1.0 + tiny), 3e-15,
          "sub-epsilon scalar screening was lost to X-T cancellation");

    auto chi = matrix(CHI);
    for (int k = 0; k < 4; ++k) chi.c[k] *= tiny;
    const auto result = screen_thermal_gw(2.0, {7, -7}, {chi, chi}, matrix(S), matrix(T));
    // Wc = T*S*chi*S*T + O(chi^2). The omitted term is <1e-17 after dividing
    // by tiny here; this independent leading-order oracle never subtracts I.
    const auto expected = product(product(product(product(T, S), CHI), S), T);
    for (int m = 0; m < 2; ++m)
        for (int k = 0; k < 4; ++k)
            close(result.wc_frequency(m, k) / tiny, expected[k], 3e-14,
                  "sub-epsilon noncommuting complex response");
}

void check_underflow_counterexample()
{
    ComplexMatrix s(1, 1), chi(1, 1), t(1, 1);
    s(0, 0) = 1e-200;
    chi(0, 0) = -1.0;
    t(0, 0) = 1e200;
    const auto s_before = s, chi_before = chi, t_before = t;
    // Exact Wc = -1/(1+1e-400) is approximately -1, not the accepted zero
    // obtained when the intermediate S*chi*S underflows before the solve.
    reject<std::underflow_error>([&] { screen_thermal_gw(2.0, {-9}, {chi}, s, t); }, "(S*chi)*S");
    unchanged(s, s_before);
    unchanged(chi, chi_before);
    unchanged(t, t_before);
}

void check_underflow_product_stages()
{
    const auto reject_scalar =
        [](double s_value, double chi_value, double t_value, const std::string& stage)
    {
        ComplexMatrix s(1, 1), chi(1, 1), t(1, 1);
        s(0, 0) = s_value;
        chi(0, 0) = chi_value;
        t(0, 0) = t_value;
        reject<std::underflow_error>([&] { screen_thermal_gw(2.0, {0}, {chi}, s, t); }, stage);
    };
    reject_scalar(1e-200, -1e-200, 1e200, "S*chi:");
    reject_scalar(1e-155, -1.0, 1e155, "(S*chi)*S:");  // Nonzero subnormal product.
    reject_scalar(1.0, -1e-200, 1e-200, "(S*chi*S)*T:");
    reject_scalar(1.0, -1.0, 1e-200, "T*Delta:");

    // A normal complex magnitude must not hide a tiny real/imaginary component.
    ComplexMatrix s(1, 1), chi(1, 1), t(1, 1);
    s(0, 0) = 1e-150;
    chi(0, 0) = Complex(-1.0, 1e-200);
    t(0, 0) = 1.0;
    reject<std::underflow_error>([&] { screen_thermal_gw(2.0, {0}, {chi}, s, t); }, "S*chi:");
    const auto complex_s = matrix(Mat2{1.0, I * 1e-200, -I * 1e-200, 1.0});
    const auto small_chi = matrix(Mat2{-1e-150, 0.0, 0.0, -1e-150});
    reject<std::underflow_error>(
        [&] { screen_thermal_gw(2.0, {0}, {small_chi}, complex_s, matrix(ID)); }, "S*chi:");
}

void check_benign_scaled_products()
{
    // Smallest components at different inner indices never multiply each other.
    const auto s = matrix(Mat2{1e-150, 0.0, 0.0, 1e150});
    const auto chi = matrix(Mat2{-1e300, 0.0, 0.0, -1e-300});
    for (double second_root : {0.0, 1.0})
    {
        const auto t = matrix(Mat2{1.0, 0.0, 0.0, second_root});
        const auto result = screen_thermal_gw(2.0, {0}, {chi}, s, t);
        row_close(result.wc_frequency, 0, Mat2{-0.5, 0.0, 0.0, -0.5 * second_root});
    }
    ComplexMatrix scalar_s(1, 1), scalar_chi(1, 1), scalar_t(1, 1);
    scalar_s(0, 0) = 1e-100;
    scalar_chi(0, 0) = -1.0;
    scalar_t(0, 0) = 1e100;
    const auto scaled = screen_thermal_gw(2.0, {0}, {scalar_chi}, scalar_s, scalar_t);
    close(scaled.wc_frequency(0, 0), -1.0, 3e-15, "representable scaled products rejected");

    const double minimum = std::numeric_limits<double>::min();
    scalar_s(0, 0) = scalar_t(0, 0) = 1.0;
    scalar_chi(0, 0) = -minimum;
    const auto boundary = screen_thermal_gw(2.0, {0}, {scalar_chi}, scalar_s, scalar_t);
    close(boundary.wc_frequency(0, 0) / minimum, -1.0, 3e-15,
          "minimum normal component product must remain admissible");
}

void check_adapter_rejections()
{
    const auto good = screen_thermal_gw(2.0, {0}, {matrix(CHI)}, matrix(S), matrix(T));
    const auto transform =
        ThermalGWTransform::from_quadrature(2.0, {0.5, 1.5}, {1.0, 1.0}, {0}, {-1, 0});
    auto bad = good;
    bad.wc_frequency.size = 3;
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    bad = good;
    bad.wc_frequency(0, 0) = {0.0, std::numeric_limits<double>::infinity()};
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    for (const auto& shape : {std::pair<int, int>{2, 4}, {1, 3}, {1, 0}})
    {
        bad = good;
        bad.wc_frequency = ComplexMatrix(shape.first, shape.second);
        reject([&] { thermal_gw_screening_to_time(bad, transform); });
    }
    bad = good;
    bad.beta_ha_inv = std::numeric_limits<double>::quiet_NaN();
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    bad = good;
    bad.bosonic_indices = {0, 0};
    reject([&] { thermal_gw_screening_to_time(bad, transform); });
    ComplexMatrix b(1, 1), f(1, 1);
    b(0, 0) = std::numeric_limits<double>::max();
    const ThermalGWTransform overflow(2.0, {1.0}, {0}, {0}, b, f);
    bad = good;
    bad.wc_frequency(0, 0) = 2.0;
    reject<std::overflow_error>([&] { thermal_gw_screening_to_time(bad, overflow); });
}
}  // namespace

int main()
{
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"general complex independent inverse and no mutation", check_general_complex_solve},
        {"pivoted multiple-RHS solve", check_pivoted_solve},
        {"independently supplied signed pole matrices", check_independent_signed_poles},
        {"static A/beta and rectangular transform", check_static_normalization},
        {"external B and exact ordered metadata", check_external_transform_and_order},
        {"beta and signed-grid validation", check_grid_rejections},
        {"matrix dimensions and storage validation", check_matrix_rejections},
        {"real and imaginary nonfinite input", check_nonfinite_rejections},
        {"Hermitian and rank-deficient roots", check_hermitian_roots},
        {"singular and nonnormal conditioning policy", check_singular_and_conditioning},
        {"scale-aware solve and arithmetic overflow", check_scale_and_overflow},
        {"independent sub-epsilon weak screening", check_tiny_response},
        {"underflow before large Coulomb amplification", check_underflow_counterexample},
        {"underflow risk in all four complex product stages", check_underflow_product_stages},
        {"benign scaled and rank-deficient products", check_benign_scaled_products},
        {"transform adapter rejects malformed samples", check_adapter_rejections}};
    int failures = 0;
    for (const auto& [name, test] : tests) try
        {
            test();
            std::cout << "PASS " << name << '\n';
        }
        catch (const std::exception& exception)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": " << exception.what() << '\n';
        }
    std::cout << tests.size() - failures << '/' << tests.size() << " groups passed\n";
    return failures == 0 ? 0 : 1;
}
