#include "optics.h"

#include <cassert>
#include <stdexcept>

namespace LIBRPA::optics
{

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

std::vector<std::complex<double>> continue_scalar_series_pade(
    const std::vector<std::complex<double>>& imag_axis,
    const std::vector<std::complex<double>>& data,
    const std::vector<double>& real_mesh,
    const int n_pade)
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
        const std::complex<double> x{omega, 0.0};
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

void write_imag_axis_data(const std::string&, const ImagAxisOpticsData&)
{
}

void write_real_axis_data(const std::string&, const RealAxisOpticsData&)
{
}

void write_summary(const std::string&, const ImagAxisOpticsData&, const RealAxisOpticsData&,
                   const int)
{
}

} // namespace LIBRPA::optics
