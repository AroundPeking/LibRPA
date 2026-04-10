#include "optics.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "constants.h"
#include "fitting.h"

namespace LIBRPA::optics
{

namespace
{

void configure_numeric_output(std::ostream& os)
{
    os.setf(std::ios::scientific);
    os.precision(std::numeric_limits<double>::max_digits10);
}

} // namespace

std::complex<double> invert_epsinv00(const std::complex<double>& epsinv00)
{
    if (std::abs(epsinv00) == 0.0)
    {
        throw std::runtime_error("epsinv00 is zero; cannot invert macroscopic dielectric scalar");
    }
    return 1.0 / epsinv00;
}

std::vector<std::complex<double>> build_imaginary_axis_points(
    const std::vector<double>& imag_freqs)
{
    std::vector<std::complex<double>> imag_axis;
    imag_axis.reserve(imag_freqs.size());
    for (const double freq : imag_freqs)
    {
        imag_axis.emplace_back(0.0, freq);
    }
    return imag_axis;
}

std::vector<double> build_real_axis_mesh(const std::vector<double>& imag_freqs, const int npts)
{
    if (imag_freqs.empty() || npts <= 0)
    {
        return {};
    }

    std::vector<double> mesh(static_cast<std::size_t>(npts), 0.0);
    if (npts == 1)
    {
        return mesh;
    }

    const double max_freq = imag_freqs.back();
    const double step = max_freq / static_cast<double>(npts - 1);
    for (int i = 0; i != npts; ++i)
    {
        mesh[static_cast<std::size_t>(i)] = step * static_cast<double>(i);
    }
    return mesh;
}

std::vector<double> build_real_axis_mesh(const double max_freq, const double step)
{
    if (max_freq < 0.0 || step <= 0.0)
    {
        return {};
    }

    std::vector<double> mesh;
    mesh.push_back(0.0);
    if (max_freq == 0.0)
    {
        return mesh;
    }

    const double tol = step * 1e-8;
    for (double omega = step; omega < max_freq - tol; omega += step)
    {
        mesh.push_back(omega);
    }
    if (std::abs(mesh.back() - max_freq) > tol)
    {
        mesh.push_back(max_freq);
    }
    return mesh;
}

std::vector<std::complex<double>> continue_scalar_series_pade(
    const std::vector<std::complex<double>>& imag_axis,
    const std::vector<std::complex<double>>& data,
    const std::vector<double>& real_mesh,
    const int n_pade,
    const double eta)
{
    const int n_data = static_cast<int>(data.size());
    assert(n_pade > 0);
    if (imag_axis.size() != data.size())
    {
        throw std::runtime_error("imag_axis and data size mismatch in continue_scalar_series_pade");
    }
    if (n_data == 0)
    {
        return {};
    }

    int n_pars = n_pade;
    std::vector<std::complex<double>> par_x;
    std::vector<std::complex<double>> data_npar;

    if (n_data <= n_pars)
    {
        n_pars = n_data;
        par_x = imag_axis;
        data_npar = data;
    }
    else
    {
        par_x.resize(n_pars);
        data_npar.resize(n_pars);
        const int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ++ipar)
        {
            par_x[static_cast<std::size_t>(ipar)] = imag_axis[static_cast<std::size_t>(ipar * step)];
            data_npar[static_cast<std::size_t>(ipar)] = data[static_cast<std::size_t>(ipar * step)];
        }
        par_x[static_cast<std::size_t>(n_pars - 1)] = imag_axis.back();
        data_npar[static_cast<std::size_t>(n_pars - 1)] = data.back();
    }

    std::vector<std::vector<std::complex<double>>> g(
        static_cast<std::size_t>(n_pars),
        std::vector<std::complex<double>>(static_cast<std::size_t>(n_pars), {0.0, 0.0}));
    for (int i_par = 0; i_par < n_pars; ++i_par)
    {
        g[static_cast<std::size_t>(i_par)][0] = data_npar[static_cast<std::size_t>(i_par)];
    }
    for (int i_par = 1; i_par < n_pars; ++i_par)
    {
        for (int i = i_par; i < n_pars; ++i)
        {
            g[static_cast<std::size_t>(i)][static_cast<std::size_t>(i_par)] =
                (g[static_cast<std::size_t>(i_par - 1)][static_cast<std::size_t>(i_par - 1)]
                 - g[static_cast<std::size_t>(i)][static_cast<std::size_t>(i_par - 1)])
                / ((par_x[static_cast<std::size_t>(i)]
                    - par_x[static_cast<std::size_t>(i_par - 1)])
                   * g[static_cast<std::size_t>(i)][static_cast<std::size_t>(i_par - 1)]);
        }
    }

    std::vector<std::complex<double>> par_y(static_cast<std::size_t>(n_pars), {0.0, 0.0});
    for (int i_par = 0; i_par < n_pars; ++i_par)
    {
        par_y[static_cast<std::size_t>(i_par)] =
            g[static_cast<std::size_t>(i_par)][static_cast<std::size_t>(i_par)];
    }

    std::vector<std::complex<double>> continued;
    continued.reserve(real_mesh.size());
    for (const double omega : real_mesh)
    {
        std::complex<double> tmp{1.0, 0.0};
        const std::complex<double> x{omega, eta};
        for (int i_par = n_pars - 1; i_par > 0; --i_par)
        {
            tmp = 1.0
                  + par_y[static_cast<std::size_t>(i_par)]
                        * (x - par_x[static_cast<std::size_t>(i_par - 1)]) / tmp;
        }
        continued.push_back(par_y[0] / tmp);
    }
    return continued;
}

namespace
{

constexpr double d_multipole_freq_floor = 1e-6;
constexpr double d_multipole_log_min = -20.0;
constexpr double d_multipole_log_max = 10.0;

double clamp_log_par(const double par)
{
    return std::max(d_multipole_log_min, std::min(d_multipole_log_max, par));
}

double positive_frequency(const double par)
{
    return std::exp(clamp_log_par(par)) + d_multipole_freq_floor;
}

double positive_strength(const double par)
{
    return std::exp(clamp_log_par(par)) + d_multipole_freq_floor;
}

double positive_damping(const double par)
{
    return std::exp(clamp_log_par(par)) + d_multipole_freq_floor;
}

void validate_tensor_shape(const matrix_m<std::complex<double>>& tensor, const char* name)
{
    if (tensor.nr() != 3 || tensor.nc() != 3)
    {
        throw std::runtime_error(std::string(name) + " must be a 3x3 tensor");
    }
}

template <typename T>
void write_tensor_row(std::ofstream& ofs, const matrix_m<std::complex<T>>& tensor)
{
    for (int row = 0; row != 3; ++row)
    {
        for (int col = 0; col != 3; ++col)
        {
            const auto value = tensor(row, col);
            ofs << ' ' << value.real() << ' ' << value.imag();
        }
    }
}

matrix_m<std::complex<double>> unpack_tensor_row(const std::vector<double>& values,
                                                 const std::size_t offset)
{
    matrix_m<std::complex<double>> tensor(3, 3, MAJOR::COL);
    tensor.zero_out();
    for (int row = 0; row != 3; ++row)
    {
        for (int col = 0; col != 3; ++col)
        {
            const std::size_t iv = offset + static_cast<std::size_t>(2 * (3 * row + col));
            tensor(row, col) = {values.at(iv), values.at(iv + 1)};
        }
    }
    return tensor;
}

double eval_multipole_imag_model(const double u, const std::vector<double>& pars)
{
    double value = pars.at(0);
    const int n_poles = static_cast<int>((pars.size() - 1) / 3);
    for (int ipole = 0; ipole != n_poles; ++ipole)
    {
        const std::size_t istr = static_cast<std::size_t>(1 + 3 * ipole);
        const std::size_t iom = istr + 1;
        const std::size_t igam = istr + 2;
        const double strength = positive_strength(pars.at(istr));
        const double omega = positive_frequency(pars.at(iom));
        const double gamma = positive_damping(pars.at(igam));
        const double den = omega * omega + gamma * u + u * u;
        value += strength / den;
    }
    return value;
}

void eval_multipole_imag_grad(std::vector<double>& grads, const double u,
                              const std::vector<double>& pars)
{
    grads.assign(pars.size(), 0.0);
    grads.at(0) = 1.0;
    const int n_poles = static_cast<int>((pars.size() - 1) / 3);
    for (int ipole = 0; ipole != n_poles; ++ipole)
    {
        const std::size_t istr = static_cast<std::size_t>(1 + 3 * ipole);
        const std::size_t iom = istr + 1;
        const std::size_t igam = istr + 2;
        const double strength = positive_strength(pars.at(istr));
        const double omega = positive_frequency(pars.at(iom));
        const double gamma = positive_damping(pars.at(igam));
        const double den = omega * omega + gamma * u + u * u;
        grads.at(istr) = strength / den;
        grads.at(iom) = -2.0 * strength * omega * omega / (den * den);
        grads.at(igam) = -strength * gamma * u / (den * den);
    }
}

std::vector<double> make_initial_multipole_guess(const std::vector<double>& imag_freqs,
                                                 const std::vector<double>& epsm_real,
                                                 const std::vector<double>& real_mesh,
                                                 const int n_poles,
                                                 const double min_probe_scale,
                                                 const double max_probe_scale,
                                                 const double damping_scale)
{
    std::vector<double> pars(static_cast<std::size_t>(1 + 3 * n_poles), 0.0);
    const double eps_inf = epsm_real.empty() ? 1.0 : epsm_real.back();
    pars.at(0) = eps_inf;

    const double base_max_probe =
        real_mesh.empty() ? (imag_freqs.empty() ? 1.0 : imag_freqs.back())
                          : std::max(real_mesh.back(), d_multipole_freq_floor * 10.0);
    const double base_min_probe =
        real_mesh.size() > 1
            ? std::max(real_mesh[1], base_max_probe * 0.05)
            : std::max(base_max_probe * 0.1, d_multipole_freq_floor * 10.0);
    const double max_probe = std::max(base_max_probe * max_probe_scale, base_min_probe * 1.5);
    const double min_probe =
        std::max(base_min_probe * min_probe_scale, d_multipole_freq_floor * 10.0);
    const double spectral_weight = epsm_real.empty() ? 1.0 : std::max(epsm_real.front() - eps_inf, 1e-4);
    const double ratio =
        n_poles == 1 ? 1.0 : std::pow(std::max(max_probe / min_probe, 1.0 + 1e-8),
                                      1.0 / static_cast<double>(n_poles - 1));
    for (int ipole = 0; ipole != n_poles; ++ipole)
    {
        const double omega = n_poles == 1
                                 ? std::max(0.5 * max_probe, d_multipole_freq_floor * 10.0)
                                 : min_probe * std::pow(ratio, static_cast<double>(ipole));
        const double gamma =
            std::max(damping_scale * omega,
                     0.5 * (real_mesh.size() > 1 ? real_mesh[1] : max_probe * 0.05));
        const std::size_t istr = static_cast<std::size_t>(1 + 3 * ipole);
        pars.at(istr) = std::log(spectral_weight * omega * omega / static_cast<double>(n_poles));
        pars.at(istr + 1) = std::log(omega);
        pars.at(istr + 2) = std::log(gamma);
    }
    return pars;
}

double eval_multipole_fit_error(const std::vector<double>& fit_freqs,
                                const std::vector<double>& fit_real,
                                const std::vector<double>& pars)
{
    double err = 0.0;
    for (std::size_t i = 0; i != fit_freqs.size(); ++i)
    {
        const double diff = eval_multipole_imag_model(fit_freqs[i], pars) - fit_real[i];
        err += diff * diff;
    }
    return err;
}

std::complex<double> eval_multipole_real_model(const double omega, const double eta,
                                               const std::vector<double>& pars)
{
    const std::complex<double> z{omega, eta};
    std::complex<double> value{pars.at(0), 0.0};
    const int n_poles = static_cast<int>((pars.size() - 1) / 3);
    for (int ipole = 0; ipole != n_poles; ++ipole)
    {
        const std::size_t istr = static_cast<std::size_t>(1 + 3 * ipole);
        const double strength = positive_strength(pars.at(istr));
        const double omega = positive_frequency(pars.at(istr + 1));
        const double gamma = positive_damping(pars.at(istr + 2));
        value += strength / (omega * omega - z * z - std::complex<double>{0.0, gamma} * z);
    }
    return value;
}

} // namespace

std::vector<std::complex<double>> continue_scalar_series_multipole(
    const std::vector<double>& imag_freqs,
    const std::vector<std::complex<double>>& data,
    const std::vector<double>& real_mesh,
    const int n_poles,
    const double eta)
{
    if (imag_freqs.size() != data.size())
    {
        throw std::runtime_error("imag_freqs and data size mismatch in continue_scalar_series_multipole");
    }
    if (n_poles <= 0)
    {
        throw std::runtime_error("continue_scalar_series_multipole requires positive n_poles");
    }
    if (data.empty())
    {
        return {};
    }

    std::vector<double> ys_real;
    ys_real.reserve(data.size());
    for (const auto& value : data)
    {
        ys_real.push_back(value.real());
    }

    std::vector<double> fit_freqs;
    std::vector<double> fit_real;
    const double fit_cutoff =
        real_mesh.empty() ? imag_freqs.back()
                          : std::max(3.0 * real_mesh.back(), imag_freqs.front());
    const std::size_t min_fit_points =
        std::min(imag_freqs.size(), static_cast<std::size_t>(std::max(2 * n_poles + 2, 6)));
    for (std::size_t i = 0; i != imag_freqs.size(); ++i)
    {
        if (imag_freqs[i] <= fit_cutoff || fit_freqs.size() < min_fit_points)
        {
            fit_freqs.push_back(imag_freqs[i]);
            fit_real.push_back(ys_real[i]);
        }
    }
    if (fit_freqs.empty() || fit_freqs.back() != imag_freqs.back())
    {
        fit_freqs.push_back(imag_freqs.back());
        fit_real.push_back(ys_real.back());
    }

    LIBRPA::utils::LevMarqFitting fitter;
    fitter.d_maxiter = 4000;
    fitter.d_target_derr = 1e-10;

    const std::vector<double> min_probe_scales{0.5, 1.0, 2.0};
    const std::vector<double> max_probe_scales{0.75, 1.0};
    const std::vector<double> damping_scales{0.05, 0.15, 0.35};

    std::vector<double> best_pars;
    double best_err = std::numeric_limits<double>::infinity();
    for (const double min_probe_scale : min_probe_scales)
    {
        for (const double max_probe_scale : max_probe_scales)
        {
            for (const double damping_scale : damping_scales)
            {
                auto pars = make_initial_multipole_guess(imag_freqs, ys_real, real_mesh, n_poles,
                                                         min_probe_scale, max_probe_scale,
                                                         damping_scale);
                fitter.fit(pars, fit_freqs, fit_real, eval_multipole_imag_model,
                           eval_multipole_imag_grad);
                const double err = eval_multipole_fit_error(fit_freqs, fit_real, pars);
                if (err < best_err)
                {
                    best_err = err;
                    best_pars = pars;
                }
            }
        }
    }

    auto pars = best_pars;

    std::vector<std::complex<double>> continued;
    continued.reserve(real_mesh.size());
    for (const double omega : real_mesh)
    {
        continued.push_back(eval_multipole_real_model(omega, eta, pars));
    }
    return continued;
}

ImagAxisOpticsData build_imag_axis_data(
    const std::vector<double>& imag_freqs,
    const std::vector<matrix_m<std::complex<double>>>& epsinv_matrices,
    const std::vector<matrix_m<std::complex<double>>>& epsm_tensors,
    const std::vector<matrix_m<std::complex<double>>>& head_tensors)
{
    const auto n_freq = imag_freqs.size();
    if (epsinv_matrices.size() != n_freq || epsm_tensors.size() != n_freq
        || head_tensors.size() != n_freq)
    {
        throw std::runtime_error("imag-axis optics inputs have inconsistent sizes");
    }

    ImagAxisOpticsData data;
    data.imag_freqs = imag_freqs;
    data.epsinv00.reserve(n_freq);
    data.epsm_avg.reserve(n_freq);
    data.epsm_tensor.reserve(n_freq);
    data.head_tensor.reserve(n_freq);

    for (std::size_t i = 0; i != n_freq; ++i)
    {
        if (epsinv_matrices[i].nr() == 0 || epsinv_matrices[i].nc() == 0)
        {
            throw std::runtime_error("average inverse dielectric matrix must contain (0,0)");
        }
        validate_tensor_shape(epsm_tensors[i], "epsm_tensor");
        validate_tensor_shape(head_tensors[i], "head_tensor");

        const auto epsinv00 = epsinv_matrices[i](0, 0);
        data.epsinv00.push_back(epsinv00);
        data.epsm_avg.push_back(invert_epsinv00(epsinv00));
        data.epsm_tensor.push_back(epsm_tensors[i].copy());
        data.head_tensor.push_back(head_tensors[i].copy());
    }

    return data;
}

ImagAxisOpticsData read_imag_axis_data(const std::string& file_path)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("failed to open imaginary-axis optics file: " + file_path);
    }

    ImagAxisOpticsData data;
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty() || line.front() == '#')
        {
            continue;
        }

        std::stringstream iss(line);
        std::vector<double> values;
        double value = 0.0;
        while (iss >> value)
        {
            values.push_back(value);
        }
        if (values.empty())
        {
            continue;
        }
        if (values.size() != 5 && values.size() != 23 && values.size() != 41)
        {
            throw std::runtime_error("unexpected column count in imaginary-axis optics file: "
                                     + std::to_string(values.size()));
        }

        data.imag_freqs.push_back(values.at(0));
        data.epsinv00.emplace_back(values.at(1), values.at(2));
        data.epsm_avg.emplace_back(values.at(3), values.at(4));

        if (values.size() >= 23)
        {
            data.epsm_tensor.push_back(unpack_tensor_row(values, 5));
        }
        if (values.size() == 41)
        {
            data.head_tensor.push_back(unpack_tensor_row(values, 23));
        }
    }
    return data;
}

RealAxisOpticsData continue_to_real_axis(const ImagAxisOpticsData& imag_data,
                                         const std::vector<double>& real_mesh, const int n_pade,
                                         const double eta)
{
    const auto n_freq = imag_data.imag_freqs.size();
    if (imag_data.epsm_avg.size() != n_freq || imag_data.epsm_tensor.size() != n_freq
        || imag_data.epsinv00.size() != n_freq || imag_data.head_tensor.size() != n_freq)
    {
        throw std::runtime_error("imag-axis optics data is internally inconsistent");
    }

    RealAxisOpticsData real_data;
    real_data.real_freqs = real_mesh;
    if (n_freq == 0 || real_mesh.empty())
    {
        return real_data;
    }

    for (const auto& tensor : imag_data.epsm_tensor)
    {
        validate_tensor_shape(tensor, "epsm_tensor");
    }

    const auto imag_axis = build_imaginary_axis_points(imag_data.imag_freqs);
    real_data.epsm_avg = continue_scalar_series_pade(imag_axis, imag_data.epsm_avg, real_mesh,
                                                     n_pade, eta);

    real_data.epsm_tensor.resize(real_mesh.size());
    for (auto& tensor : real_data.epsm_tensor)
    {
        tensor.resize(3, 3);
        tensor.zero_out();
    }

    for (int row = 0; row != 3; ++row)
    {
        for (int col = 0; col != 3; ++col)
        {
            std::vector<std::complex<double>> series;
            series.reserve(n_freq);
            for (std::size_t i = 0; i != n_freq; ++i)
            {
                series.push_back(imag_data.epsm_tensor[i](row, col));
            }

            const auto continued = continue_scalar_series_pade(imag_axis, series, real_mesh,
                                                               n_pade, eta);
            for (std::size_t i = 0; i != real_mesh.size(); ++i)
            {
                real_data.epsm_tensor[i](row, col) = continued[i];
            }
        }
    }

    return real_data;
}

RealAxisOpticsData continue_to_real_axis_multipole(const ImagAxisOpticsData& imag_data,
                                                   const std::vector<double>& real_mesh,
                                                   const int n_poles,
                                                   const double eta)
{
    if (imag_data.epsm_avg.size() != imag_data.imag_freqs.size())
    {
        throw std::runtime_error("imag-axis epsilon_M data is internally inconsistent");
    }

    RealAxisOpticsData real_data;
    real_data.real_freqs = real_mesh;
    if (imag_data.imag_freqs.empty() || real_mesh.empty())
    {
        return real_data;
    }

    real_data.epsm_avg = continue_scalar_series_multipole(
        imag_data.imag_freqs, imag_data.epsm_avg, real_mesh, n_poles, eta);
    return real_data;
}

void write_imag_axis_data(const std::string& file_path, const ImagAxisOpticsData& data)
{
    std::ofstream ofs(file_path);
    configure_numeric_output(ofs);
    ofs << "# omega_imag_ha epsinv00_re epsinv00_im epsm_avg_re epsm_avg_im"
        << " epsm_xx_re epsm_xx_im epsm_xy_re epsm_xy_im epsm_xz_re epsm_xz_im"
        << " epsm_yx_re epsm_yx_im epsm_yy_re epsm_yy_im epsm_yz_re epsm_yz_im"
        << " epsm_zx_re epsm_zx_im epsm_zy_re epsm_zy_im epsm_zz_re epsm_zz_im"
        << " head_xx_re head_xx_im head_xy_re head_xy_im head_xz_re head_xz_im"
        << " head_yx_re head_yx_im head_yy_re head_yy_im head_yz_re head_yz_im"
        << " head_zx_re head_zx_im head_zy_re head_zy_im head_zz_re head_zz_im\n";

    for (std::size_t i = 0; i != data.imag_freqs.size(); ++i)
    {
        ofs << data.imag_freqs[i]
            << ' ' << data.epsinv00[i].real() << ' ' << data.epsinv00[i].imag()
            << ' ' << data.epsm_avg[i].real() << ' ' << data.epsm_avg[i].imag();
        write_tensor_row(ofs, data.epsm_tensor[i]);
        write_tensor_row(ofs, data.head_tensor[i]);
        ofs << '\n';
    }
}

void write_real_axis_data(const std::string& file_path, const RealAxisOpticsData& data)
{
    std::ofstream ofs(file_path);
    configure_numeric_output(ofs);
    const bool has_tensors = data.epsm_tensor.size() == data.real_freqs.size();
    if (!data.epsm_tensor.empty() && !has_tensors)
    {
        throw std::runtime_error("real-axis tensor data is internally inconsistent");
    }

    ofs << "# omega_real_ha epsm_avg_re epsm_avg_im";
    if (has_tensors)
    {
        ofs << " epsm_xx_re epsm_xx_im epsm_xy_re epsm_xy_im epsm_xz_re epsm_xz_im"
            << " epsm_yx_re epsm_yx_im epsm_yy_re epsm_yy_im epsm_yz_re epsm_yz_im"
            << " epsm_zx_re epsm_zx_im epsm_zy_re epsm_zy_im epsm_zz_re epsm_zz_im";
    }
    ofs << '\n';

    for (std::size_t i = 0; i != data.real_freqs.size(); ++i)
    {
        ofs << data.real_freqs[i]
            << ' ' << data.epsm_avg[i].real() << ' ' << data.epsm_avg[i].imag();
        if (has_tensors)
        {
            write_tensor_row(ofs, data.epsm_tensor[i]);
        }
        ofs << '\n';
    }
}

void write_absorption_spectrum(const std::string& file_path, const RealAxisOpticsData& data)
{
    constexpr double c_light_au = 137.035999084;
    const double bohr_inv_to_cm_inv = ANG2BOHR * 1.0e8;

    std::ofstream ofs(file_path);
    configure_numeric_output(ofs);
    ofs << "# omega_real_ha omega_real_ev epsm_avg_re epsm_avg_im loss_fn refr_index"
        << " ext_coeff alpha_bohr_inv alpha_cm_inv\n";

    for (std::size_t i = 0; i != data.real_freqs.size(); ++i)
    {
        const auto eps = data.epsm_avg[i];
        const double eps_abs = std::abs(eps);
        const double refr_index = std::sqrt(std::max(0.0, 0.5 * (eps_abs + eps.real())));
        const double ext_coeff = std::sqrt(std::max(0.0, 0.5 * (eps_abs - eps.real())));
        const double loss_fn = (eps_abs == 0.0) ? 0.0 : std::imag(-1.0 / eps);
        const double alpha_bohr_inv = 2.0 * data.real_freqs[i] * ext_coeff / c_light_au;
        const double alpha_cm_inv = alpha_bohr_inv * bohr_inv_to_cm_inv;

        ofs << data.real_freqs[i]
            << ' ' << data.real_freqs[i] * HA2EV
            << ' ' << eps.real()
            << ' ' << eps.imag()
            << ' ' << loss_fn
            << ' ' << refr_index
            << ' ' << ext_coeff
            << ' ' << alpha_bohr_inv
            << ' ' << alpha_cm_inv
            << '\n';
    }
}

void write_summary(const std::string& file_path, const ImagAxisOpticsData& imag_data,
                   const RealAxisOpticsData& real_data,
                   const std::string& continuation_method,
                   const int continuation_order, const double eta)
{
    std::ofstream ofs(file_path);
    configure_numeric_output(ofs);
    ofs << "task absorption\n";
    ofs << "n_imag_freq " << imag_data.imag_freqs.size() << '\n';
    ofs << "n_real_freq " << real_data.real_freqs.size() << '\n';
    ofs << "continuation_method " << continuation_method << '\n';
    ofs << "continuation_order " << continuation_order << '\n';
    ofs << "eta_ha " << eta << '\n';
    ofs << "eta_ev " << eta * HA2EV << '\n';
    ofs << "definition epsm_avg = 1 / epsinv00\n";
}

} // namespace LIBRPA::optics
