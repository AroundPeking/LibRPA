#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/thermal_gw_screening.h"
#include "../core/thermal_gw_transform.h"
#include "../core/thermal_occupation.h"
#ifdef LIBRPA_USE_LIBRI
#include <mpi.h>

#include <map>
#include <utility>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"
#include "RI/physics/GW.h"
#endif

namespace
{
using librpa_int::ComplexMatrix;
using librpa_int::ThermalGWTransform;
using Complex = std::complex<double>;
const double PI = std::acos(-1.0);
const Complex I(0.0, 1.0);
constexpr int CONVOLUTION_CUTOFF = 16384;
// Model acceptance is independent of the sparse basis truncation eps in the fixture.
constexpr double MODEL_TOLERANCE = 5e-9;

// Throwing checks, including the numerical assertions, remain active in Release.
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture
{
    std::string version;
    double beta = 0.0, g_wmax = 0.0, w_wmax = 0.0, sigma_wmax = 0.0, tolerance = 0.0;
    std::vector<double> times;
    std::vector<int> bosons, fermions;
    ComplexMatrix b, f;
};

Fixture read_fixture(const char* path)
{
    // This whitespace format is a test-only exporter fixture, not a public input format.
    std::ifstream input(path);
    require(input.is_open(), std::string("cannot open sparse GW fixture: ") + path);
    const auto expect = [&](const char* expected)
    {
        std::string token;
        require(static_cast<bool>(input >> token) && token == expected,
                std::string("sparse GW fixture: expected ") + expected);
    };
    Fixture fixture;
    expect("LIBRPA_THERMAL_GW_V1");
    expect("sparse-ir");
    require(static_cast<bool>(input >> fixture.version), "missing sparse-ir version");
    expect("Ha");
    expect("Ha^-1");
    expect("B");
    expect("exp_minus");
    expect("inverse");
    expect("F");
    expect("exp_plus");
    expect("integral");
    int ntau = 0, nb = 0, nf = 0;
    require(static_cast<bool>(input >> fixture.beta >> fixture.g_wmax >> fixture.w_wmax >>
                              fixture.sigma_wmax >> fixture.tolerance >> ntau >> nb >> nf),
            "invalid sparse GW fixture metadata");
    for (double value :
         {fixture.beta, fixture.g_wmax, fixture.w_wmax, fixture.sigma_wmax, fixture.tolerance})
        require(std::isfinite(value) && value > 0.0, "fixture scales must be positive and finite");
    require(fixture.tolerance < 1.0, "fixture tolerance must be below one");
    const double summed_wmax = fixture.g_wmax + fixture.w_wmax;
    require(std::isfinite(summed_wmax) &&
                fixture.sigma_wmax >=
                    summed_wmax * (1.0 - 64.0 * std::numeric_limits<double>::epsilon()),
            "Sigma spectral window must cover the G and W windows' sum");
    // Bound allocations without assuming the sparse basis or sampling counts.
    constexpr int MAX_ELEMENTS = 1024 * 1024;
    require(ntau > 0 && nb > 0 && nf > 0 && ntau <= MAX_ELEMENTS && nb <= MAX_ELEMENTS / ntau &&
                nf <= MAX_ELEMENTS / ntau,
            "sparse GW fixture dimensions exceed the bounded test budget");
    fixture.times.resize(ntau);
    fixture.bosons.resize(nb);
    fixture.fermions.resize(nf);
    for (double& tau : fixture.times)
        require(static_cast<bool>(input >> tau) && std::isfinite(tau), "invalid fixture tau");
    for (int& m : fixture.bosons)
        require(static_cast<bool>(input >> m), "invalid signed bosonic label");
    for (int& n : fixture.fermions)
        require(static_cast<bool>(input >> n), "invalid signed fermionic label");
    const auto read_matrix = [&](ComplexMatrix& matrix, int rows, int cols)
    {
        matrix.create(rows, cols);
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
            {
                double real = 0.0, imag = 0.0;
                require(static_cast<bool>(input >> real >> imag) && std::isfinite(real) &&
                            std::isfinite(imag),
                        "invalid or truncated sparse GW operator");
                matrix(row, col) = {real, imag};
            }
    };
    read_matrix(fixture.b, ntau, nb);
    read_matrix(fixture.f, nf, ntau);
    input >> std::ws;
    require(input.eof() && !input.bad(), "trailing tokens in sparse GW fixture");
    require(std::find(fixture.bosons.begin(), fixture.bosons.end(), 0) != fixture.bosons.end(),
            "fixture must contain the static bosonic mode");
    require(*std::min_element(fixture.bosons.begin(), fixture.bosons.end()) < 0 &&
                *std::max_element(fixture.bosons.begin(), fixture.bosons.end()) > 0,
            "fixture must sample both bosonic frequency signs");
    for (int n : {-1, 0})
        require(std::find(fixture.fermions.begin(), fixture.fermions.end(), n) !=
                    fixture.fermions.end(),
                "fixture must contain the two lowest signed fermionic modes");
    require(std::any_of(fixture.fermions.begin(), fixture.fermions.end(),
                        [](int n) { return n < -2 || n > 1; }),
            "fixture must also exercise higher fermionic modes");
    return fixture;
}

enum class ModelKind
{
    PolePair,
    Static
};

struct Model
{
    const char* name;
    ModelKind kind;
    double positive_pole, negative_pole;
    Complex amplitude;
};

Complex w_frequency(const Model& model, double nu, int m)
{
    if (model.kind == ModelKind::Static) return m == 0 ? model.amplitude : Complex{};
    return model.amplitude *
           (1.0 / (I * nu - model.positive_pole) - 1.0 / (I * nu + model.negative_pole));
}

Complex w_time(const Model& model, double tau, double beta)
{
    if (model.kind == ModelKind::Static) return model.amplitude / beta;
    return -model.amplitude *
           (std::exp(-model.positive_pole * tau) / (-std::expm1(-beta * model.positive_pole)) +
            std::exp(-model.negative_pole * (beta - tau)) /
                (-std::expm1(-beta * model.negative_pole)));
}

double green_time(double xi, double tau, double beta)
{
    // Independent positive-time G_lib = -G_standard, with no occupation subtraction.
    if (xi >= 0.0) return std::exp(-xi * tau) / (1.0 + std::exp(-beta * xi));
    return std::exp(xi * (beta - tau)) / (1.0 + std::exp(beta * xi));
}

Complex sigma_closed_form(const Model& model, double omega, double xi, double beta)
{
    if (model.kind == ModelKind::Static) return -model.amplitude / (beta * (I * omega - xi));
    const double occupation = xi >= 0.0 ? std::exp(-beta * xi) / (1.0 + std::exp(-beta * xi))
                                        : 1.0 / (1.0 + std::exp(beta * xi));
    const auto bose = [&](double pole)
    { return std::exp(-beta * pole) / (-std::expm1(-beta * pole)); };
    return model.amplitude *
           ((1.0 + bose(model.positive_pole) - occupation) /
                (I * omega - xi - model.positive_pole) +
            (bose(model.negative_pole) + occupation) / (I * omega - xi + model.negative_pole));
}

Complex signed_convolution(const Model& model, int n, double xi, double beta, int cutoff)
{
    // Direct -sum_m W(i*nu_m) G_standard(i*omega_n-i*nu_m)/beta. No sparse operators,
    // time-domain reference, conjugation, or closed-form Sigma enter this sum.
    using LongComplex = std::complex<long double>;
    const LongComplex imaginary(0.0L, 1.0L);
    const LongComplex amplitude(model.amplitude.real(), model.amplitude.imag());
    const long double period = beta, energy = xi;
    const long double omega = (2.0L * n + 1.0L) * std::acos(-1.0L) / period;
    const long double positive = model.positive_pole, negative = model.negative_pole;
    LongComplex sum = 0.0L;
    if (model.kind == ModelKind::Static) cutoff = 0;
    for (int m = -cutoff; m <= cutoff; ++m)
    {
        const long double nu = 2.0L * m * std::acos(-1.0L) / period;
        // The factored pole expression avoids subtracting almost equal high-frequency terms.
        const LongComplex w = model.kind == ModelKind::Static
                                  ? amplitude
                                  : amplitude * (positive + negative) /
                                        ((imaginary * nu - positive) * (imaginary * nu + negative));
        sum -= w / (imaginary * (omega - nu) - energy) / period;
    }
    return {static_cast<double>(sum.real()), static_cast<double>(sum.imag())};
}

struct ErrorMetric
{
    double absolute = 0.0, scaled = 0.0;

    void add(Complex actual, Complex expected, double scale)
    {
        require(std::isfinite(actual.real()) && std::isfinite(actual.imag()) &&
                    std::isfinite(expected.real()) && std::isfinite(expected.imag()) &&
                    std::isfinite(scale) && scale > 0.0,
                "nonfinite numerical reference or transform result");
        const double error = std::abs(actual - expected);
        absolute = std::max(absolute, error);
        scaled = std::max(scaled, error / std::max(scale, std::abs(expected)));
    }

    void check(double tolerance, const std::string& name) const
    {
        std::cout << "  " << name << ": max_abs=" << absolute << " max_scaled=" << scaled
                  << " limit=" << tolerance << std::endl;
        require(scaled <= tolerance, name + " differs from independent reference");
    }
};

void check_models(const Fixture& fixture, const ThermalGWTransform& transform, bool reordered)
{
    const double wmax = fixture.w_wmax, gmax = fixture.g_wmax, beta = fixture.beta;
    const std::vector<Model> models{
        {"real_even_poles", ModelKind::PolePair, 0.4 * wmax, 0.4 * wmax, 0.15 * wmax * wmax},
        {"complex_unequal_poles", ModelKind::PolePair, 0.35 * wmax, 0.55 * wmax,
         Complex(0.1, -0.075) * wmax * wmax},
        {"static_delta_m0", ModelKind::Static, 0.0, 0.0, Complex(0.7, -0.25) * wmax}};
    const double near_fermi = std::min(0.02 * gmax, 0.05 / beta);
    const std::vector<double> energies{-0.7 * gmax, -near_fermi, 0.0, near_fermi, 0.65 * gmax};
    const auto& bosons = transform.get_bosonic_indices();
    const auto& fermions = transform.get_fermionic_indices();
    const auto& times = transform.get_times();
    const int nmodels = static_cast<int>(models.size()), nxi = static_cast<int>(energies.size());
    const double tolerance = MODEL_TOLERANCE;
    const bool check_direct = !reordered && beta == 8.0 && gmax == 1.0 && wmax == 2.0;
    ComplexMatrix samples(static_cast<int>(bosons.size()), nmodels);
    for (int row = 0; row < samples.nr; ++row)
        for (int col = 0; col < samples.nc; ++col)
            samples(row, col) =
                w_frequency(models[col], 2.0 * PI * bosons[row] / beta, bosons[row]);
    const auto actual_w_time = transform.apply_bosonic_frequency_to_time(samples);
    require(actual_w_time.nr == static_cast<int>(times.size()) && actual_w_time.nc == nmodels,
            "W(tau) batch shape changed");
    ComplexMatrix sigma_time(actual_w_time.nr, nmodels * nxi);
    ComplexMatrix reference_sigma_time(actual_w_time.nr, nmodels * nxi);
    std::vector<ErrorMetric> w_errors(nmodels), time_errors(nmodels), sigma_errors(nmodels),
        f_errors(nmodels), convolution_errors(nmodels), reference_errors(nmodels),
        tail_errors(nmodels);
    ErrorMetric green_error;
    for (int row = 0; row < actual_w_time.nr; ++row)
        for (int model = 0; model < nmodels; ++model)
        {
            const auto expected_w = w_time(models[model], times[row], beta);
            const double time_scale = models[model].kind == ModelKind::Static
                                          ? std::abs(models[model].amplitude) / beta
                                          : std::abs(models[model].amplitude);
            w_errors[model].add(actual_w_time(row, model), expected_w, time_scale);
            for (int energy = 0; energy < nxi; ++energy)
            {
                const int col = model * nxi + energy;
                const double xi = energies[energy];
                const double g_lib =
                    librpa_int::thermal_green_amplitude(xi, times[row], 1.0 / beta);
                const double expected_g = green_time(xi, times[row], beta);
                // For xi<0 near beta, -xi*tau cancels xi/(1/beta) in the log form.
                // Both exponent evaluations round O(beta*|xi|) arithmetic. Since 0<G<1,
                // their absolute roundoff scales as epsilon*(1+beta*|xi|), not epsilon alone.
                green_error.add(g_lib, expected_g, 1.0 + beta * std::abs(xi));
                // G_lib supplies the minus sign already. Apply no additional beta/half weight.
                sigma_time(row, col) = g_lib * actual_w_time(row, model);
                reference_sigma_time(row, col) = expected_g * expected_w;
                time_errors[model].add(sigma_time(row, col), reference_sigma_time(row, col),
                                       time_scale);
            }
        }
    const auto sigma = transform.apply_fermionic_time_to_frequency(sigma_time);
    const auto sigma_from_exact_time =
        transform.apply_fermionic_time_to_frequency(reference_sigma_time);
    require(sigma.nr == static_cast<int>(fermions.size()) && sigma.nc == nmodels * nxi,
            "Sigma(iomega) batch shape changed");
    int low_rows = 0;
    for (int row = 0; row < sigma.nr; ++row)
    {
        const int n = fermions[row];
        // Sparse high labels can be enormous: use the analytic reference there, never a huge sum.
        const bool direct = check_direct && n >= -2 && n <= 1;
        if (direct) ++low_rows;
        const double omega = (2.0 * n + 1.0) * PI / beta;
        for (int model = 0; model < nmodels; ++model)
            for (int energy = 0; energy < nxi; ++energy)
            {
                const int col = model * nxi + energy;
                const double xi = energies[energy];
                const auto expected = sigma_closed_form(models[model], omega, xi, beta);
                sigma_errors[model].add(sigma(row, col), expected, wmax);
                f_errors[model].add(sigma_from_exact_time(row, col), expected, wmax);
                if (direct)
                {
                    const auto convolution =
                        signed_convolution(models[model], n, xi, beta, CONVOLUTION_CUTOFF);
                    const auto shorter =
                        signed_convolution(models[model], n, xi, beta, CONVOLUTION_CUTOFF / 2);
                    convolution_errors[model].add(sigma(row, col), convolution, wmax);
                    reference_errors[model].add(convolution, expected, wmax);
                    tail_errors[model].add(convolution, shorter, wmax);
                }
            }
    }
    std::cout << (reordered ? "Permuted" : "Exported") << " sparse GW ordering; xi=";
    for (double xi : energies) std::cout << ' ' << xi;
    std::cout << "; direct low rows=" << low_rows << "; cutoff=" << CONVOLUTION_CUTOFF << '\n';
    if (!reordered && !check_direct)
        std::cout << "  SKIP direct signed convolution outside beta8/g1/w2: fixed cutoff is not "
                     "certified for a wider window; ALL fermionic rows use closed-form Sigma.\n";
    green_error.check(64.0 * std::numeric_limits<double>::epsilon(),
                      "G_lib(tau), arithmetic scale 1+beta*|xi|");
    for (int model = 0; model < nmodels; ++model)
    {
        const std::string name = models[model].name;
        w_errors[model].check(tolerance, name + " W(tau)");
        time_errors[model].check(tolerance, name + " G_lib*W(tau)");
        f_errors[model].check(tolerance, name + " exact-time -> F -> Sigma");
        sigma_errors[model].check(tolerance, name + " W -> B -> G_lib*W -> F -> Sigma");
        if (check_direct)
        {
            reference_errors[model].check(5e-11, name + " direct vs closed-form Sigma");
            tail_errors[model].check(4e-10, name + " direct cutoff refinement");
            convolution_errors[model].check(tolerance + 5e-11, name + " sparse vs direct Sigma");
        }
    }
    if (check_direct)
        require(low_rows >= 2 && low_rows <= 4, "missing signed low-frequency checks");
    const auto& complex_model = models[1];
    const auto w_plus = w_frequency(complex_model, 2.0 * PI / beta, 1);
    const auto w_minus = w_frequency(complex_model, -2.0 * PI / beta, -1);
    require(std::abs(w_plus - w_minus) > 1e-6 * wmax &&
                std::abs(w_plus - std::conj(w_minus)) > 1e-6 * wmax &&
                std::abs(w_time(complex_model,
                                std::min(0.2 * beta, 0.5 / complex_model.positive_pole), beta)
                             .imag()) > 1e-6 * wmax * wmax,
            "complex pole model must be neither real nor even");
}

void check_screening(const Fixture& fixture, const ThermalGWTransform& transform)
{
    const double beta = fixture.beta, wmax = fixture.w_wmax;
    const double a = 0.2 * wmax, v = 0.4 * wmax, c = 0.15 * wmax;
    const double pole = std::sqrt(a * a + c * v);
    const Model model{"screened_scalar", ModelKind::PolePair, pole, pole, c * v * v / (2.0 * pole)};
    ComplexMatrix root(1, 1);
    root(0, 0) = std::sqrt(v);
    std::vector<ComplexMatrix> chi;
    for (int m : transform.get_bosonic_indices())
    {
        const double nu = 2.0 * PI * m / beta;
        ComplexMatrix value(1, 1);
        value(0, 0) = -c / (nu * nu + a * a);
        chi.push_back(value);
    }
    const auto screening =
        librpa_int::screen_thermal_gw(beta, transform.get_bosonic_indices(), chi, root, root);
    const auto time = librpa_int::thermal_gw_screening_to_time(screening, transform);
    ErrorMetric frequency_error, time_error, sigma_error;
    for (int row = 0; row < screening.wc_frequency.nr; ++row)
    {
        const double nu = 2.0 * PI * transform.get_bosonic_indices()[row] / beta;
        // epsilon=1-v*chi gives Wc=-c*v^2/(nu^2+a^2+c*v), with no instantaneous bare v.
        frequency_error.add(screening.wc_frequency(row, 0), -c * v * v / (nu * nu + a * a + c * v),
                            wmax);
    }
    const double near_fermi = std::min(0.02 * fixture.g_wmax, 0.05 / beta);
    const std::vector<double> energies{-0.7 * fixture.g_wmax, -near_fermi, 0.0, near_fermi,
                                       0.65 * fixture.g_wmax};
    ComplexMatrix sigma_time(time.nr, static_cast<int>(energies.size()));
    for (int row = 0; row < time.nr; ++row)
    {
        const double tau = transform.get_times()[row];
        time_error.add(time(row, 0), w_time(model, tau, beta), std::abs(model.amplitude));
        for (int col = 0; col < sigma_time.nc; ++col)
            sigma_time(row, col) =
                librpa_int::thermal_green_amplitude(energies[col], tau, 1.0 / beta) * time(row, 0);
    }
    const auto sigma = transform.apply_fermionic_time_to_frequency(sigma_time);
    for (int row = 0; row < sigma.nr; ++row)
        for (int col = 0; col < sigma.nc; ++col)
        {
            const double omega = (2.0 * transform.get_fermionic_indices()[row] + 1.0) * PI / beta;
            sigma_error.add(sigma(row, col), sigma_closed_form(model, omega, energies[col], beta),
                            wmax);
        }
    frequency_error.check(128.0 * std::numeric_limits<double>::epsilon(), "scalar LU Wc(i*nu)");
    time_error.check(MODEL_TOLERANCE, "scalar screen -> B -> Wc(tau)");
    sigma_error.check(MODEL_TOLERANCE, "scalar screen -> B -> G_lib*Wc -> F -> Sigma");
}

#ifdef LIBRPA_USE_LIBRI
void check_libri_contraction(const Fixture& fixture, const ThermalGWTransform& transform)
{
    using Tensor = RI::Tensor<Complex>;
    using TensorMap = std::map<int, std::map<std::pair<int, std::array<int, 3>>, Tensor>>;
    const std::array<int, 3> origin{0, 0, 0};
    const std::array<std::array<double, 3>, 3> lattice{
        {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    const double beta = fixture.beta, wmax = fixture.w_wmax;
    const Model model{"LibRI_complex_poles", ModelKind::PolePair, 0.35 * wmax, 0.55 * wmax,
                      Complex(0.1, -0.075) * wmax * wmax};
    const double cs[2][2][2] = {{{0.30, 0.11}, {-0.06, 0.22}}, {{-0.12, 0.09}, {0.18, 0.27}}};
    const Complex auxiliary[2][2] = {{Complex(0.9, 0.1), Complex(0.2, -0.3)},
                                     {Complex(-0.1, 0.25), Complex(0.7, -0.2)}};
    const double cosine = std::cos(0.37), sine = std::sin(0.37);
    const Complex phase = std::polar(1.0, 0.63);
    const Complex u[2][2] = {{cosine, -sine * std::conj(phase)}, {sine * phase, cosine}};
    const double energies[2] = {-0.3 * fixture.g_wmax,
                                std::min(0.02 * fixture.g_wmax, 0.05 / beta)};
    Tensor c_tensor({2, 2, 2});
    for (int mu = 0; mu < 2; ++mu)
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < 2; ++k) c_tensor(mu, i, k) = cs[mu][i][k];
    RI::GW<int, int, 3, Complex> gw;
    gw.set_parallel(MPI_COMM_WORLD, {{0, {0.0, 0.0, 0.0}}}, lattice, {1, 1, 1});
    gw.set_symmetry(false, {});
    gw.set_Cs(TensorMap{{0, {{{0, origin}, c_tensor}}}}, 0.0);
    ComplexMatrix w_samples(static_cast<int>(transform.get_bosonic_indices().size()), 4);
    for (int row = 0; row < w_samples.nr; ++row)
        for (int mu = 0; mu < 2; ++mu)
            for (int nu = 0; nu < 2; ++nu)
            {
                const int m = transform.get_bosonic_indices()[row];
                w_samples(row, 2 * mu + nu) =
                    auxiliary[mu][nu] * w_frequency(model, 2.0 * PI * m / beta, m);
            }
    const auto w_tau = transform.apply_bosonic_frequency_to_time(w_samples);
    ComplexMatrix sigma_tau(w_tau.nr, 4);
    ErrorMetric contraction_error, time_error, frequency_error;
    for (int row = 0; row < w_tau.nr; ++row)
    {
        const double tau = transform.get_times()[row];
        Tensor g({2, 2}), w({2, 2});
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
            {
                w(i, j) = w_tau(row, 2 * i + j);
                for (int band = 0; band < 2; ++band)
                    g(i, j) += u[i][band] * std::conj(u[j][band]) *
                               librpa_int::thermal_green_amplitude(energies[band], tau, 1.0 / beta);
            }
        gw.set_Ws(TensorMap{{0, {{{0, origin}, w}}}}, 0.0);
        gw.set_Gs(TensorMap{{0, {{{0, origin}, g}}}}, 0.0);
        gw.cal_Sigmas();
        require(gw.Sigmas.size() == 1 && gw.Sigmas.at(0).size() == 1,
                "one-atom LibRI contraction returned unexpected blocks");
        const auto& sigma = gw.Sigmas.at(0).at({0, origin});
        require(sigma.shape.size() == 2 && sigma.shape[0] == 2 && sigma.shape[1] == 2,
                "LibRI Sigma must be 2 AO by 2 AO");
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
            {
                Complex explicit_sum = 0.0, exact_time = 0.0;
                for (int mu = 0; mu < 2; ++mu)
                    for (int nu = 0; nu < 2; ++nu)
                        for (int k = 0; k < 2; ++k)
                            for (int l = 0; l < 2; ++l)
                            {
                                // Four same-atom LRI center assignments, with real Cs. Both AO
                                // orderings contribute; no conjugation or extra factor is inserted.
                                const double vertex =
                                    (cs[mu][i][k] + cs[mu][k][i]) * (cs[nu][j][l] + cs[nu][l][j]);
                                explicit_sum += vertex * g(k, l) * w(mu, nu);
                                for (int band = 0; band < 2; ++band)
                                    exact_time += vertex * u[k][band] * std::conj(u[l][band]) *
                                                  green_time(energies[band], tau, beta) *
                                                  auxiliary[mu][nu] * w_time(model, tau, beta);
                            }
                sigma_tau(row, 2 * i + j) = sigma(i, j);
                contraction_error.add(sigma(i, j), explicit_sum, wmax * wmax);
                time_error.add(sigma(i, j), exact_time, wmax * wmax);
            }
    }
    const auto sigma = transform.apply_fermionic_time_to_frequency(sigma_tau);
    ComplexMatrix sigma_expected(sigma.nr, sigma.nc);
    for (int row = 0; row < sigma.nr; ++row)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
            {
                const double omega =
                    (2.0 * transform.get_fermionic_indices()[row] + 1.0) * PI / beta;
                Complex expected = 0.0;
                for (int mu = 0; mu < 2; ++mu)
                    for (int nu = 0; nu < 2; ++nu)
                        for (int k = 0; k < 2; ++k)
                            for (int l = 0; l < 2; ++l)
                                for (int band = 0; band < 2; ++band)
                                    expected +=
                                        (cs[mu][i][k] + cs[mu][k][i]) *
                                        (cs[nu][j][l] + cs[nu][l][j]) * u[k][band] *
                                        std::conj(u[l][band]) * auxiliary[mu][nu] *
                                        sigma_closed_form(model, omega, energies[band], beta);
                sigma_expected(row, 2 * i + j) = expected;
                frequency_error.add(sigma(row, 2 * i + j), expected, wmax);
            }
    std::cout << "LibRI::GW::cal_Sigmas: one atom, 2 AO, 2 auxiliary, " << w_tau.nr
              << " positive sparse times; no production-GW/LibRI workflow claim\n";
    contraction_error.check(512.0 * std::numeric_limits<double>::epsilon(),
                            "LibRI vs explicit Cs*G*W*Cs");
    time_error.check(MODEL_TOLERANCE, "LibRI Sigma(tau) vs exact time");
    frequency_error.check(MODEL_TOLERANCE, "B -> LibRI cal_Sigmas -> F vs closed-form Sigma");

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    librpa_int::PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    const librpa_int::AtomicBasis ao(std::vector<std::size_t>{2});
    const librpa_int::AtomicBasis abf(std::vector<std::size_t>{2});
    librpa_int::MeanField mf(1, 1, 2, 2);
    const double chemical_potential = 0.15 * fixture.g_wmax;
    const auto fd =
        librpa_int::make_fermi_dirac_reference(1.0 / beta, chemical_potential, 2.0, 1e-12);
    mf.get_efermi() = chemical_potential;
    mf.set_fermi_dirac_reference(fd);
    auto& eigenvectors = mf.get_eigenvectors()[0][0][0];
    eigenvectors.create(2, 2);
    for (int band = 0; band < 2; ++band)
    {
        const double energy = energies[band] + chemical_potential;
        mf.get_eigenvals()[0](0, band) = energy;
        mf.get_weight()[0](0, band) = librpa_int::normalized_fermi_dirac_band_weight(energy, fd, 1);
        // MeanField stores (band, AO): transpose u without conjugating it.
        for (int i = 0; i < 2; ++i) eigenvectors(band, i) = u[i][band];
    }
    librpa_int::KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, 1);
    const auto desc = context.create_array_desc(2, 2);
    const librpa_int::TFGrids response_grid;  // No legacy time/frequency grid is initialized.
    const librpa_int::SymmetryContext symmetry;
    const librpa_int::G0W0 production_gw(mf, ao, pbc, symmetry, response_grid, context, context,
                                         desc, false, false);
    librpa_int::Cs_LRI lri_cs;
    lri_cs.use_libri = true;
    if (rank == 0)
    {
        RI::Tensor<double> tensor({2, 2, 2});
        for (int mu = 0; mu < 2; ++mu)
            for (int i = 0; i < 2; ++i)
                for (int k = 0; k < 2; ++k) tensor(mu, i, k) = cs[mu][i][k];
        lri_cs.data_libri[0][{0, origin}] = std::move(tensor);
    }
    std::map<double, std::map<librpa_int::Vector3_Order<double>, librpa_int::Matz>> wc;
    const auto bosonic_frequencies = transform.get_bosonic_frequencies_ha();
    for (int row = 0; row < w_samples.nr; ++row)
    {
        librpa_int::Matz block(desc.m_loc(), desc.n_loc(), librpa_int::MAJOR::ROW);
        for (int i = 0; i < desc.m_loc(); ++i)
            for (int j = 0; j < desc.n_loc(); ++j)
                block(i, j) = w_samples(row, 2 * desc.indx_l2g_r(i) + desc.indx_l2g_c(j));
        wc[bosonic_frequencies[row]][pbc.klist_full.at(0)] = std::move(block);
    }
    const auto result = production_gw.build_thermal_spacetime(abf, lri_cs, wc, desc, transform);
    require(
        result.beta_ha_inv == beta && result.fermionic_indices == transform.get_fermionic_indices(),
        "internal G0W0 changed beta or signed fermionic label order");
    require(!production_gw.is_rspace_built() && production_gw.sigc_kspace_source().empty() &&
                production_gw.sigc_is_ik_f_KS.empty() && production_gw.sigc_diag_is_ik_f_KS.empty(),
            "internal thermal G0W0 changed legacy GW state");
    require(
        result.blocks.size() == 1 && result.blocks.at(0).size() == result.fermionic_indices.size(),
        "internal G0W0 returned unexpected spin/frequency blocks");
    ErrorMetric production_frequency_error;
    const auto fermionic_frequencies = transform.get_fermionic_frequencies_ha();
    for (int row = 0; row < sigma_expected.nr; ++row)
    {
        const auto& pairs = result.blocks.at(0).at(fermionic_frequencies[row]);
        require(pairs.size() == 1 && pairs.at({0, 0}).size() == 1,
                "one-atom internal G0W0 returned unexpected AO pair/R blocks");
        const auto& block = pairs.at({0, 0}).at({0, 0, 0});
        require(block.nr() == 2 && block.nc() == 2, "internal G0W0 Sigma must be 2 AO by 2 AO");
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                production_frequency_error.add(block(i, j), sigma_expected(row, 2 * i + j), wmax);
    }
    production_frequency_error.check(
        MODEL_TOLERANCE, "internal G0W0 W(q,i*nu) -> B -> Green/LibRI -> F vs closed-form Sigma");
}
#endif

ThermalGWTransform reordered_transform(const Fixture& fixture)
{
    std::vector<int> b_order(fixture.bosons.size()), f_order(fixture.fermions.size());
    std::iota(b_order.begin(), b_order.end(), 0);
    std::iota(f_order.begin(), f_order.end(), 0);
    std::reverse(b_order.begin(), b_order.end());
    std::rotate(f_order.begin(), f_order.begin() + 1, f_order.end());
    auto bosons = fixture.bosons, fermions = fixture.fermions;
    ComplexMatrix b(fixture.b.nr, fixture.b.nc), f(fixture.f.nr, fixture.f.nc);
    for (int col = 0; col < b.nc; ++col)
    {
        bosons[col] = fixture.bosons[b_order[col]];
        for (int row = 0; row < b.nr; ++row) b(row, col) = fixture.b(row, b_order[col]);
    }
    for (int row = 0; row < f.nr; ++row)
    {
        fermions[row] = fixture.fermions[f_order[row]];
        for (int col = 0; col < f.nc; ++col) f(row, col) = fixture.f(f_order[row], col);
    }
    return ThermalGWTransform(fixture.beta, fixture.times, bosons, fermions, b, f);
}
}  // namespace

int main(int argc, char** argv)
{
#ifdef LIBRPA_USE_LIBRI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif
    int status = 0;
    try
    {
#ifdef LIBRPA_USE_LIBRI
        librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
        librpa_int::global::init_global_io();
        int ranks = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        require(ranks == 1, "bounded external GW integration test requires one MPI rank");
#endif
        require(argc <= 2, "usage: test_external_thermal_gw [exported-fixture.dat]");
        const char* path = argc == 2 ? argv[1] : LIBRPA_SPARSE_GW_FIXTURE;
        const auto fixture = read_fixture(path);
        std::cout << std::setprecision(12) << "Sparse GW fixture: " << path << "; sparse-ir "
                  << fixture.version << "; beta=" << fixture.beta << "; windows=" << fixture.g_wmax
                  << ',' << fixture.w_wmax << ',' << fixture.sigma_wmax
                  << "; eps=" << fixture.tolerance << "; ntau/nb/nf=" << fixture.times.size() << '/'
                  << fixture.bosons.size() << '/' << fixture.fermions.size() << '\n';
        const ThermalGWTransform transform(fixture.beta, fixture.times, fixture.bosons,
                                           fixture.fermions, fixture.b, fixture.f);
        check_models(fixture, transform, false);
        check_screening(fixture, transform);
        const auto reordered = reordered_transform(fixture);
        check_models(fixture, reordered, true);
        check_screening(fixture, reordered);
#ifdef LIBRPA_USE_LIBRI
        check_libri_contraction(fixture, reordered);
#else
        std::cout << "SKIP LibRI contraction: LIBRPA_USE_LIBRI is disabled\n";
#endif
        std::cout << "External sparse-ir thermal GW model integration checks passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_external_thermal_gw: " << error.what() << '\n';
        status = 1;
    }
#ifdef LIBRPA_USE_LIBRI
    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
#endif
    return status;
}
