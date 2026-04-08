#pragma once

#include <complex>
#include <string>
#include <vector>

#include "matrix_m.h"

namespace LIBRPA::optics
{

struct ImagAxisOpticsData
{
    std::vector<double> imag_freqs;
    std::vector<std::complex<double>> epsinv00;
    std::vector<std::complex<double>> epsm_avg;
    std::vector<matrix_m<std::complex<double>>> epsm_tensor;
    std::vector<matrix_m<std::complex<double>>> head_tensor;
};

struct RealAxisOpticsData
{
    std::vector<double> real_freqs;
    std::vector<std::complex<double>> epsm_avg;
    std::vector<matrix_m<std::complex<double>>> epsm_tensor;
};

std::complex<double> invert_epsinv00(const std::complex<double>& epsinv00);

std::vector<std::complex<double>> build_imaginary_axis_points(
    const std::vector<double>& imag_freqs);

std::vector<double> build_real_axis_mesh(const std::vector<double>& imag_freqs, int npts);

std::vector<std::complex<double>> continue_scalar_series_pade(
    const std::vector<std::complex<double>>& imag_axis,
    const std::vector<std::complex<double>>& data,
    const std::vector<double>& real_mesh,
    int n_pade);

void write_imag_axis_data(const std::string& file_path, const ImagAxisOpticsData& data);

void write_real_axis_data(const std::string& file_path, const RealAxisOpticsData& data);

void write_summary(const std::string& file_path, const ImagAxisOpticsData& imag_data,
                   const RealAxisOpticsData& real_data, int n_pade);

} // namespace LIBRPA::optics
