#include "thermal_gw_screening.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "../interface/blas_lapack.h"

namespace librpa_int
{
namespace
{
using Complex = std::complex<double>;
const double PI = std::acos(-1.0);
const double ROUNDING_FACTOR = 64.0;

bool finite(Complex value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); }

void validate_dimensions(std::size_t rows, std::size_t columns)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (rows == 0 || columns == 0 || rows > maximum || columns > maximum ||
        rows > maximum / columns)
        throw std::invalid_argument(
            "thermal GW screening dimensions must fit positive int storage");
}

void validate_matrix(const ComplexMatrix& matrix, const std::string& name)
{
    if (matrix.nr <= 0 || matrix.nc <= 0)
        throw std::invalid_argument(name + " must have positive dimensions");
    validate_dimensions(matrix.nr, matrix.nc);
    if (matrix.size != matrix.nr * matrix.nc || matrix.c == nullptr)
        throw std::invalid_argument(name + " has inconsistent storage");
    for (int k = 0; k < matrix.size; ++k)
        if (!finite(matrix.c[k])) throw std::invalid_argument(name + " contains a nonfinite value");
}

void validate_grid(double beta, const std::vector<int>& indices)
{
    if (!std::isfinite(beta) || beta <= 0.0 || !std::isfinite(1.0 / beta))
        throw std::invalid_argument(
            "thermal GW screening requires positive finite beta and 1/beta");
    validate_dimensions(indices.size(), 1);
    std::unordered_set<int> unique;
    for (int m : indices)
    {
        if (!unique.insert(m).second)
            throw std::invalid_argument("thermal GW screening bosonic labels must be unique");
        // Match ThermalGWTransform, promoting signed labels before doubling.
        const double nu = 2.0 * static_cast<double>(m) * PI / beta;
        if (!std::isfinite(nu) || (m != 0 && nu == 0.0))
            throw std::invalid_argument("thermal GW screening frequency is not representable");
    }
}

double tolerance(int n) { return ROUNDING_FACTOR * n * std::numeric_limits<double>::epsilon(); }

void validate_root(const ComplexMatrix& root, const char* name)
{
    validate_matrix(root, name);
    if (root.nr != root.nc) throw std::invalid_argument(std::string(name) + " must be square");
    double scale = 0.0;
    for (int k = 0; k < root.size; ++k)
        scale = std::max({scale, std::abs(root.c[k].real()), std::abs(root.c[k].imag())});
    if (scale == 0.0) return;
    // Normalize before subtracting, without an absolute tolerance floor or repair.
    for (int r = 0; r < root.nr; ++r)
        for (int c = 0; c <= r; ++c)
            if (std::abs(root(r, c) / scale - std::conj(root(c, r) / scale)) > tolerance(root.nr))
                throw std::invalid_argument(std::string(name) + " must be Hermitian");
}

void check_finite(const ComplexMatrix& matrix, const std::string& context)
{
    for (int k = 0; k < matrix.size; ++k)
        if (!finite(matrix.c[k])) throw std::overflow_error(context + ": nonfinite arithmetic");
}

ComplexMatrix checked_product(const ComplexMatrix& a, const ComplexMatrix& b,
                              const std::string& context)
{
    const auto minimum_component = [](double minimum, Complex value)
    {
        for (double component : {std::abs(value.real()), std::abs(value.imag())})
            if (component > 0.0) minimum = std::min(minimum, component);
        return minimum;
    };
    // O(N^2) preflight of finite operands: for each inner index, the smallest
    // nonzero components of this column/row bound every real component product.
    // Reject subnormal-product risk even if summation would later cancel it.
    for (int k = 0; k < a.nc; ++k)
    {
        double left = std::numeric_limits<double>::infinity();
        double right = std::numeric_limits<double>::infinity();
        for (int r = 0; r < a.nr; ++r) left = minimum_component(left, a(r, k));
        for (int c = 0; c < b.nc; ++c) right = minimum_component(right, b(k, c));
        if (std::isfinite(left) && std::isfinite(right) &&
            left < std::numeric_limits<double>::min() / right)
            throw std::underflow_error(
                context + ": component product underflow risk at inner index " + std::to_string(k));
    }
    auto result = a * b;
    check_finite(result, context);
    return result;
}

double norm_inf(const ComplexMatrix& matrix, const std::string& context)
{
    double norm = 0.0;
    for (int r = 0; r < matrix.nr; ++r)
    {
        double sum = 0.0;
        for (int c = 0; c < matrix.nc; ++c) sum += std::abs(matrix(r, c));
        if (!std::isfinite(sum))
            throw std::overflow_error(context + ": unrepresentable matrix norm");
        norm = std::max(norm, sum);
    }
    return norm;
}

double backward_error(const ComplexMatrix& a, const ComplexMatrix& x, const ComplexMatrix& b,
                      const std::string& context)
{
    const double an = norm_inf(a, context), xn = norm_inf(x, context), bn = norm_inf(b, context);
    if (xn == 0.0) return bn == 0.0 ? 0.0 : 1.0;

    // Compute ||A X-B||_inf/(||A||_inf ||X||_inf+||B||_inf) without forming
    // an overflowing norm product or residual. Binary scaling avoids log roundoff.
    int ae = 0, xe = 0, be = 0;
    const double am = std::frexp(an, &ae), xm = std::frexp(xn, &xe), bm = std::frexp(bn, &be);
    const int exponent = bn == 0.0 ? ae + xe : std::max(ae + xe, be);
    const double ax_scale = std::scalbn(am * xm, ae + xe - exponent);
    const double b_scale = std::scalbn(bm, be - exponent);
    const double ax_weight = ax_scale / (ax_scale + b_scale);
    const double b_weight = b_scale / (ax_scale + b_scale);
    double residual = 0.0;
    for (int r = 0; r < b.nr; ++r)
    {
        double sum = 0.0;
        for (int c = 0; c < b.nc; ++c)
        {
            Complex ax = 0.0;
            for (int k = 0; k < a.nc; ++k) ax += (a(r, k) / an) * (x(k, c) / xn);
            const Complex scaled_b = bn == 0.0 ? Complex(0.0) : b(r, c) / bn;
            sum += std::abs(ax_weight * ax - b_weight * scaled_b);
        }
        if (!std::isfinite(sum)) throw std::overflow_error(context + ": nonfinite residual");
        residual = std::max(residual, sum);
    }
    return residual;
}

void numerical_failure(const std::string& context, const char* metric, double value, double limit)
{
    std::ostringstream message;
    message << std::setprecision(17) << context << ": " << metric << '=' << value
            << " fails threshold " << limit;
    throw std::runtime_error(message.str());
}

ComplexMatrix solve(const ComplexMatrix& epsilon, const ComplexMatrix& b,
                    const std::string& context)
{
    const int n = epsilon.nr, nrhs = 2 * n;
    std::vector<Complex> lu(epsilon.size), rhs(static_cast<std::size_t>(n) * nrhs, 0.0);
    std::vector<int> pivots(n);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
        {
            // Both A and all RHS columns are Fortran column-major; no conjugation.
            lu[r + c * n] = epsilon(r, c);
            rhs[r + c * n] = b(r, c);
            rhs[r + (c + n) * n] = r == c ? 1.0 : 0.0;
        }
    int info = 0;
    zgesv_(&n, &nrhs, lu.data(), &n, pivots.data(), rhs.data(), &n, &info);
    if (info != 0) throw std::runtime_error(context + ": zgesv INFO=" + std::to_string(info));
    for (const auto& value : lu)
        if (!finite(value)) throw std::overflow_error(context + ": nonfinite LU factors");
    ComplexMatrix x(n, n), inverse(n, n), identity(n, n);
    identity.set_as_identity_matrix();
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
        {
            x(r, c) = rhs[r + c * n];
            inverse(r, c) = rhs[r + (c + n) * n];
        }
    check_finite(x, context);
    check_finite(inverse, context);
    const double an = norm_inf(epsilon, context), invn = norm_inf(inverse, context);
    const double rcond = (1.0 / std::max(an, invn)) / std::min(an, invn);
    if (!std::isfinite(rcond) || rcond <= tolerance(n))
        numerical_failure(context, "rcond_inf", rcond, tolerance(n));
    const double residual = std::max(backward_error(epsilon, x, b, context),
                                     backward_error(epsilon, inverse, identity, context));
    if (!std::isfinite(residual) || residual > tolerance(n))
        numerical_failure(context, "backward residual", residual, tolerance(n));
    return x;
}
}  // namespace

ThermalGWScreening screen_thermal_gw(double beta_ha_inv, const std::vector<int>& bosonic_indices,
                                     const std::vector<ComplexMatrix>& chi, const ComplexMatrix& s,
                                     const ComplexMatrix& t)
{
    validate_grid(beta_ha_inv, bosonic_indices);
    if (chi.size() != bosonic_indices.size())
        throw std::invalid_argument("thermal GW screening chi count does not match bosonic labels");
    validate_root(s, "thermal GW screening S");
    validate_root(t, "thermal GW screening T");
    if (s.nr != t.nr)
        throw std::invalid_argument("thermal GW screening roots must have equal order");
    validate_dimensions(s.nr, 2 * static_cast<std::size_t>(s.nr));
    validate_dimensions(chi.size(), s.size);
    for (std::size_t m = 0; m < chi.size(); ++m)
    {
        validate_matrix(chi[m], "thermal GW chi m=" + std::to_string(bosonic_indices[m]));
        if (chi[m].nr != s.nr || chi[m].nc != s.nc)
            throw std::invalid_argument("thermal GW screening chi shape does not match roots");
    }

    ThermalGWScreening result{beta_ha_inv, bosonic_indices,
                              ComplexMatrix(static_cast<int>(chi.size()), s.size)};
    for (std::size_t m = 0; m < chi.size(); ++m)
    {
        const std::string context = "thermal GW screening m=" + std::to_string(bosonic_indices[m]);
        const auto s_chi = checked_product(s, chi[m], context + " S*chi");
        const auto s_chi_s = checked_product(s_chi, s, context + " (S*chi)*S");
        auto epsilon = s_chi_s;
        for (int r = 0; r < epsilon.nr; ++r)
            for (int c = 0; c < epsilon.nc; ++c)
                epsilon(r, c) = (r == c ? 1.0 : 0.0) - epsilon(r, c);
        check_finite(epsilon, context);
        // epsilon*(X-T) = (S*chi*S)*T: retain weak screening without X-T cancellation.
        const auto rhs = checked_product(s_chi_s, t, context + " (S*chi*S)*T");
        const auto delta = solve(epsilon, rhs, context);
        const auto wc = checked_product(t, delta, context + " T*Delta");
        for (int k = 0; k < wc.size; ++k) result.wc_frequency(static_cast<int>(m), k) = wc.c[k];
    }
    return result;
}

ComplexMatrix thermal_gw_screening_to_time(const ThermalGWScreening& screening,
                                           const ThermalGWTransform& transform)
{
    validate_grid(screening.beta_ha_inv, screening.bosonic_indices);
    if (screening.beta_ha_inv != transform.get_beta_ha_inv() ||
        screening.bosonic_indices != transform.get_bosonic_indices())
        throw std::invalid_argument("thermal GW screening beta/ordered labels do not match B");
    validate_matrix(screening.wc_frequency, "thermal GW screening packed Wc");
    const int order = static_cast<int>(std::sqrt(screening.wc_frequency.nc));
    if (static_cast<std::size_t>(screening.wc_frequency.nr) != screening.bosonic_indices.size() ||
        order * order != screening.wc_frequency.nc)
        throw std::invalid_argument(
            "thermal GW screening packed Wc must have N_boson by N*N shape");
    return transform.apply_bosonic_frequency_to_time(screening.wc_frequency);
}
}  // namespace librpa_int
