#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/tetrahedron_quadrature.h"
#include "../core/thermal_occupation.h"
#include "../core/timefreq.h"
#ifdef LIBRPA_USE_LIBRI
#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "librpa.hpp"
#endif

namespace
{
using namespace librpa_int;

struct Fixture
{
    double beta, wmax, tolerance;
    std::vector<double> times;
    ComplexMatrix transform;
};

Fixture read_fixture()
{
    std::ifstream input(LIBRPA_SPARSE_TAU_FIXTURE);
    std::string magic, method, version;
    Fixture fixture;
    int ntau = 0, nfreq = 0;
    input >> magic >> method >> version >> fixture.beta >> fixture.wmax >> fixture.tolerance >>
        ntau >> nfreq;
    if (!input || magic != "LIBRPA_SPARSE_TAU_TEST_V1" || method != "sparse-ir" || ntau <= 0 ||
        nfreq != 16)
        throw std::runtime_error("invalid sparse-time test fixture header");
    fixture.times.resize(ntau);
    for (auto &time : fixture.times) input >> time;
    fixture.transform.create(nfreq, ntau);
    for (int row = 0; row < nfreq; ++row)
        for (int col = 0; col < ntau; ++col)
        {
            double real = 0.0, imag = 0.0;
            input >> real >> imag;
            fixture.transform(row, col) = {real, imag};
        }
    if (!input) throw std::runtime_error("truncated sparse-time test fixture");
    return fixture;
}

void require_close(std::complex<double> actual, std::complex<double> expected, double tolerance)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > tolerance)
        throw std::runtime_error("external thermal transform differs from its reference");
}

void check_green_pairs(const Fixture &fixture)
{
    TFGrids grid(fixture.transform.nr);
    grid.set_finite_beta_time_grid(fixture.times, fixture.beta, fixture.transform);
    if (grid.get_n_grids() <= grid.get_n_time_grids() / 2 + 1)
        throw std::runtime_error("fixture does not exercise independent time/frequency counts");
    const double kbt = 1.0 / fixture.beta;
    for (double gap : {0.0, 1e-7, 0.01, 0.3, 1.9})
    {
        const double en = 0.13 - gap / 2.0, em = 0.13 + gap / 2.0;
        const double fn = fermi_dirac_occupation(en, kbt);
        const double fm = fermi_dirac_occupation(em, kbt);
        for (std::size_t l = 0; l < grid.get_n_grids(); ++l)
        {
            std::complex<double> actual = 0.0;
            for (std::size_t j = 0; j < fixture.times.size(); ++j)
                actual -= grid.get_time_to_frequency_factor(l, j) *
                          thermal_green_amplitude(em, fixture.times[j], kbt) *
                          thermal_green_amplitude(en, -fixture.times[j], kbt);
            const std::complex<double> denominator(-gap, grid.get_freq_nodes()[l]);
            const auto expected =
                gap == 0.0 ? std::complex<double>(l == 0 ? fermi_dirac_derivative(en, kbt) : 0.0)
                           : (fn - fm) / denominator;
            require_close(actual, expected, 5e-9);
            require_close(grid.find_correlation_frequency_weight(grid.get_freq_nodes()[l]),
                          (l == 0 ? 0.5 : 1.0) / fixture.beta, 1e-14);
        }
    }
}

void check_validation(const Fixture &fixture)
{
    TFGrids grid(fixture.transform.nr);
    grid.set_finite_beta_time_grid(fixture.times, fixture.beta, fixture.transform);
    auto reject = [&](const std::vector<double> &times, double beta, const ComplexMatrix &matrix)
    {
        bool rejected = false;
        try
        {
            grid.set_finite_beta_time_grid(times, beta, matrix);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        if (!rejected || grid.get_time_nodes() != fixture.times)
            throw std::runtime_error(
                "invalid external grid was accepted or changed the active grid");
        require_close(grid.get_time_to_frequency_factor(0, 0), fixture.transform(0, 0), 0.0);
    };
    reject({}, fixture.beta, fixture.transform);
    reject(fixture.times, -fixture.beta, fixture.transform);
    reject(fixture.times, std::numeric_limits<double>::infinity(), fixture.transform);
    auto times = fixture.times;
    times[0] = 0.0;
    reject(times, fixture.beta, fixture.transform);
    times = fixture.times;
    times[1] = times[0];
    reject(times, fixture.beta, fixture.transform);
    times = fixture.times;
    times.back() = fixture.beta;
    reject(times, fixture.beta, fixture.transform);
    times[0] = std::numeric_limits<double>::quiet_NaN();
    reject(times, fixture.beta, fixture.transform);
    ComplexMatrix bad = fixture.transform;
    bad(1, 1) = std::numeric_limits<double>::quiet_NaN();
    reject(fixture.times, fixture.beta, bad);
    bad = fixture.transform;
    bad(0, 0) += std::complex<double>(0.0, 1.0);
    reject(fixture.times, fixture.beta, bad);
    ComplexMatrix wrong_shape(1, 1);
    reject(fixture.times, fixture.beta, wrong_shape);
    grid.set_finite_beta_time_grid(grid.get_time_nodes(), fixture.beta, grid.get_fourier_t2f());
    require_close(grid.get_time_to_frequency_factor(0, 0), fixture.transform(0, 0), 0.0);
    grid.generate_finite_beta_matsubara(64, fixture.beta);
    require_close(grid.get_time_nodes()[0], fixture.beta / 128.0, 1e-15);
}

std::vector<PeriodicTetrahedronQuadraturePoint> build_high_order_periodic_tetrahedron_quadrature(
    const Vector3_Order<int> &period, const std::vector<Vector3_Order<double>> &kfrac)
{
    static constexpr std::array<std::array<std::array<int, 3>, 4>, 6> tetra_vertices{
        {{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}},
         {{{0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}}},
         {{{0, 0, 0}, {1, 0, 1}, {1, 0, 0}, {1, 1, 1}}}}};
    // Eight Gauss-Legendre nodes on [0,1].  The Duffy Jacobian integrates the
    // non-polynomial LRI phase factors in the nonlocal-C control accurately.
    static constexpr std::array<double, 8> nodes{
        0.0198550717512319, 0.1016667612931870, 0.2372337950418355, 0.4082826787521751,
        0.5917173212478249, 0.7627662049581645, 0.8983332387068130, 0.9801449282487681};
    static constexpr std::array<double, 8> weights{
        0.0506142681451881, 0.1111905172266872, 0.1568533229389436, 0.1813418916891809,
        0.1813418916891809, 0.1568533229389436, 0.1111905172266872, 0.0506142681451881};

    const std::size_t nk = static_cast<std::size_t>(period.x) * period.y * period.z;
    if (period.x < 2 || period.y < 2 || period.z < 2 || kfrac.size() != nk)
        throw std::runtime_error("high-order tetrahedron control requires a complete 3D grid");

    const auto reduce_fractional = [](double value)
    {
        value -= std::floor(value);
        return value >= 1.0 - 1.0e-12 ? 0.0 : value;
    };
    std::array<std::vector<double>, 3> coordinates;
    for (int axis = 0; axis != 3; ++axis)
    {
        auto &axis_coordinates = coordinates[axis];
        axis_coordinates.reserve(nk);
        for (const auto &point : kfrac)
            axis_coordinates.push_back(reduce_fractional(axis == 0   ? point.x
                                                         : axis == 1 ? point.y
                                                                     : point.z));
        std::sort(axis_coordinates.begin(), axis_coordinates.end());
        axis_coordinates.erase(std::unique(axis_coordinates.begin(), axis_coordinates.end(),
                                           [](const double lhs, const double rhs)
                                           { return std::abs(lhs - rhs) < 1e-10; }),
                               axis_coordinates.end());
        const int axis_period = axis == 0 ? period.x : axis == 1 ? period.y : period.z;
        if (static_cast<int>(axis_coordinates.size()) != axis_period)
            throw std::runtime_error("high-order tetrahedron control found an incomplete grid");
    }

    std::vector<std::size_t> canonical_to_input(nk, nk);
    for (std::size_t input = 0; input != nk; ++input)
    {
        const auto &point = kfrac[input];
        std::array<int, 3> index{};
        for (int axis = 0; axis != 3; ++axis)
        {
            const double value = reduce_fractional(axis == 0   ? point.x
                                                   : axis == 1 ? point.y
                                                               : point.z);
            const auto iter =
                std::lower_bound(coordinates[axis].begin(), coordinates[axis].end(), value);
            if (iter == coordinates[axis].end() || std::abs(*iter - value) > 1e-10)
                throw std::runtime_error("high-order tetrahedron control found an off-grid point");
            index[axis] = static_cast<int>(iter - coordinates[axis].begin());
        }
        const std::size_t canonical =
            (static_cast<std::size_t>(index[0]) * period.y + static_cast<std::size_t>(index[1])) *
                period.z +
            static_cast<std::size_t>(index[2]);
        if (canonical_to_input[canonical] != nk)
            throw std::runtime_error("high-order tetrahedron control found a duplicate k point");
        canonical_to_input[canonical] = input;
    }

    const auto wrap_index = [](int value, int axis_period)
    {
        value %= axis_period;
        return value < 0 ? value + axis_period : value;
    };
    std::vector<PeriodicTetrahedronQuadraturePoint> result;
    result.reserve(nk * tetra_vertices.size() * nodes.size() * nodes.size() * nodes.size());
    for (int ix = 0; ix != period.x; ++ix)
        for (int iy = 0; iy != period.y; ++iy)
            for (int iz = 0; iz != period.z; ++iz)
                for (const auto &tetrahedron : tetra_vertices)
                {
                    std::array<std::size_t, 4> vertices{};
                    std::array<Vector3_Order<double>, 4> vertex_coordinates{};
                    for (int vertex = 0; vertex != 4; ++vertex)
                    {
                        const auto &offset = tetrahedron[vertex];
                        const std::array<int, 3> wrapped{wrap_index(ix + offset[0], period.x),
                                                         wrap_index(iy + offset[1], period.y),
                                                         wrap_index(iz + offset[2], period.z)};
                        const std::size_t canonical =
                            (static_cast<std::size_t>(wrapped[0]) * period.y + wrapped[1]) *
                                period.z +
                            wrapped[2];
                        vertices[vertex] = canonical_to_input[canonical];
                        vertex_coordinates[vertex] = {
                            static_cast<double>(ix + offset[0]) / period.x,
                            static_cast<double>(iy + offset[1]) / period.y,
                            static_cast<double>(iz + offset[2]) / period.z};
                    }
                    for (int iu = 0; iu != static_cast<int>(nodes.size()); ++iu)
                        for (int iv = 0; iv != static_cast<int>(nodes.size()); ++iv)
                            for (int iw = 0; iw != static_cast<int>(nodes.size()); ++iw)
                            {
                                const double u = nodes[iu];
                                const double v = nodes[iv];
                                const double w = nodes[iw];
                                const std::array<double, 4> barycentric{
                                    1.0 - u, u * (1.0 - v), u * v * (1.0 - w), u * v * w};
                                Vector3_Order<double> coordinate{0.0, 0.0, 0.0};
                                for (int vertex = 0; vertex != 4; ++vertex)
                                {
                                    coordinate.x +=
                                        barycentric[vertex] * vertex_coordinates[vertex].x;
                                    coordinate.y +=
                                        barycentric[vertex] * vertex_coordinates[vertex].y;
                                    coordinate.z +=
                                        barycentric[vertex] * vertex_coordinates[vertex].z;
                                }
                                result.push_back({vertices, barycentric, coordinate,
                                                  weights[iu] * weights[iv] * weights[iw] * u * u *
                                                      v / static_cast<double>(nk)});
                            }
                }
    return result;
}

#ifdef LIBRPA_USE_LIBRI
void check_full_libri_response(const Fixture &fixture, bool nonlocal_coefficients,
                               bool sparse_frequencies = false)
{
    constexpr int nk = 3;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double kbt = 1.0 / fixture.beta;
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, nk, 2, 2);
    auto ds = librpa_int::api::get_dataset_instance(handler);
    auto &mf = ds->mf;
    mf.get_efermi() = 0.0;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(kbt, 0.0, 2.0, 1e-12));
    ds->basis_wfc.set({2});
    ds->basis_aux.set({2});
    ds->pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    std::vector<double> kvecs;
    for (int ik = 0; ik < nk; ++ik)
    {
        const double k = two_pi * ik / nk;
        kvecs.insert(kvecs.end(), {k, 0.0, 0.0});
        mf.get_eigenvals()[0](ik, 0) = -0.4 + 0.15 * std::cos(k);
        mf.get_eigenvals()[0](ik, 1) = 0.5 + 0.1 * std::cos(k);
        auto &u = mf.get_eigenvectors()[0][0][ik];
        u.create(2, 2);
        u(0, 0) = std::cos(0.37);
        u(0, 1) = std::sin(0.37) * std::polar(1.0, k);
        u(1, 0) = -std::sin(0.37) * std::polar(1.0, -k);
        u(1, 1) = std::cos(0.37);
        for (int n = 0; n < 2; ++n)
            mf.get_weight()[0](ik, n) =
                2.0 * fermi_dirac_occupation(mf.get_eigenvals()[0](ik, n), kbt) / nk;
    }
    ds->pbc.set_kgrids_kvec(nk, 1, 1, kvecs);
    std::vector<double> energies(mf.get_eigenvals()[0].c, mf.get_eigenvals()[0].c + 2 * nk);
    std::vector<double> occupations(2 * nk);
    for (int i = 0; i < 2 * nk; ++i) occupations[i] = mf.get_weight()[0].c[i] * nk;
    handler.set_wg_ekb_efermi(1, nk, 2, occupations.data(), energies.data(), 0.0);
    std::vector<double> transform_real(fixture.transform.size),
        transform_imag(fixture.transform.size);
    for (int i = 0; i < fixture.transform.size; ++i)
    {
        transform_real[i] = fixture.transform.c[i].real();
        transform_imag[i] = fixture.transform.c[i].imag();
    }
    int nfreq = fixture.transform.nr;
    if (sparse_frequencies)
    {
        const std::vector<int> modes{0, 1, 3, 7, 15};
        const std::vector<double> weights{0.5 / fixture.beta, 0.2, 0.4, -0.1, 0.6};
        nfreq = modes.size();
        std::vector<double> real(nfreq * fixture.transform.nc), imag(real.size());
        for (int row = 0; row < nfreq; ++row)
            for (int col = 0; col < fixture.transform.nc; ++col)
            {
                real[row * fixture.transform.nc + col] = fixture.transform(modes[row], col).real();
                imag[row * fixture.transform.nc + col] = fixture.transform(modes[row], col).imag();
            }
        handler.set_external_thermal_rpa_grid(fixture.beta, fixture.wmax, 2 * fixture.wmax,
                                              fixture.tolerance, nfreq, fixture.transform.nc,
                                              fixture.times.data(), modes.data(), weights.data(),
                                              real.data(), imag.data());
    }
    else
        handler.set_external_thermal_time_grid(fixture.beta, fixture.wmax, fixture.tolerance,
                                               fixture.transform.nr, fixture.transform.nc,
                                               fixture.times.data(), transform_real.data(),
                                               transform_imag.data());
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = nfreq;
    options.ntau = fixture.transform.nc;
    initialize_ds_tfgrids(*ds, options);
    const auto &grid = ds->tfg;

    // For an onsite pair the two atom-centered LRI contributions are each P_mu/2.
    const double vertices[2][2][2] = {{{1.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.7}, {0.7, 0.2}}};
    const double neighbor_coefficients[2][2][2][2] = {
        {{{0.08, 0.03}, {-0.02, 0.05}}, {{-0.03, 0.07}, {0.04, 0.02}}},
        {{{-0.02, 0.06}, {0.01, 0.04}}, {{0.05, -0.03}, {0.02, -0.01}}}};
    Cs_LRI cs;
    cs.use_libri = true;
    if (ds->comm_h.is_root())
    {
        auto coefficients = std::make_shared<std::valarray<double>>(8);
        for (int mu = 0; mu < 2; ++mu)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    (*coefficients)[(mu * 2 + i) * 2 + j] = 0.5 * vertices[mu][i][j];
        cs.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 2, 2}, coefficients);
        if (nonlocal_coefficients)
            for (int side = 0; side < 2; ++side)
            {
                auto neighbor = std::make_shared<std::valarray<double>>(8);
                for (int mu = 0; mu < 2; ++mu)
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            (*neighbor)[(mu * 2 + i) * 2 + j] =
                                neighbor_coefficients[side][mu][i][j];
                cs.data_libri[0][{0, {side == 0 ? 1 : -1, 0, 0}}] =
                    RI::Tensor<double>({2, 2, 2}, neighbor);
            }
    }
    Chi0 chi(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, grid,
             ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi.gf_threshold = 0.0;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    chi.build(LIBRPA_ROUTING_LIBRI, cs, {{0, 0}}, ds->basis_aux, no_shrink, ds->blacs_h);

    double max_error = 0.0;
    int checked = 0;
    for (const auto &[frequency, qblocks] : chi.get_chi0_q())
        for (const auto &[q, blocks] : qblocks)
        {
            const int iq = (static_cast<int>(std::lround(q.x * nk)) % nk + nk) % nk;
            ComplexMatrix expected(2, 2);
            for (int ik = 0; ik < nk; ++ik)
                for (int n = 0; n < 2; ++n)
                    for (int m = 0; m < 2; ++m)
                    {
                        const int ikq = (ik + iq) % nk;
                        const double en = mf.get_eigenvals()[0](ik, n);
                        const double em = mf.get_eigenvals()[0](ikq, m);
                        const auto &un = mf.get_eigenvectors()[0][0][ik];
                        const auto &um = mf.get_eigenvectors()[0][0][ikq];
                        std::array<std::complex<double>, 2> vertex{};
                        for (int mu = 0; mu < 2; ++mu)
                            for (int i = 0; i < 2; ++i)
                                for (int j = 0; j < 2; ++j)
                                {
                                    std::complex<double> pair_vertex = vertices[mu][i][j];
                                    if (nonlocal_coefficients)
                                        for (int side = 0; side < 2; ++side)
                                        {
                                            const int r = side == 0 ? 1 : -1;
                                            // The two LRI centers contribute C_ij(R)e^(ik'R)
                                            // and C_ji(R)e^(-ikR), respectively.
                                            pair_vertex +=
                                                neighbor_coefficients[side][mu][i][j] *
                                                    std::polar(1.0, two_pi * ikq * r / nk) +
                                                neighbor_coefficients[side][mu][j][i] *
                                                    std::polar(1.0, -two_pi * ik * r / nk);
                                        }
                                    vertex[mu] += std::conj(un(n, i)) * pair_vertex * um(m, j);
                                }
                        const std::complex<double> denominator(en - em, frequency);
                        const std::complex<double> bubble =
                            std::abs(denominator) < 1e-14 ? fermi_dirac_derivative(en, kbt)
                                                          : (fermi_dirac_occupation(en, kbt) -
                                                             fermi_dirac_occupation(em, kbt)) /
                                                                denominator;
                        for (int mu = 0; mu < 2; ++mu)
                            for (int nu = 0; nu < 2; ++nu)
                                expected(mu, nu) +=
                                    2.0 / nk * bubble * vertex[mu] * std::conj(vertex[nu]);
                    }
            for (const auto &[atom_i, row] : blocks)
                for (const auto &[atom_j, actual] : row)
                    for (int mu = 0; mu < 2; ++mu)
                        for (int nu = 0; nu < 2; ++nu)
                        {
                            max_error =
                                std::max(max_error, std::abs(actual(mu, nu) - expected(mu, nu)));
                            ++checked;
                        }
        }
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_error, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &checked, 1, MPI_SUM);
    if (ds->comm_h.is_root())
        std::cout << "Sparse LibRI/Adler-Wiser max error: " << max_error
                  << "; matrix elements checked: " << checked
                  << "; nonlocal coefficients: " << nonlocal_coefficients << std::endl;
    if (checked < nfreq * nk * 4 || max_error > 5e-9)
        throw std::runtime_error("sparse full LibRI response differs from Adler-Wiser");

    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(2.0 * kbt, 0.0, 2.0, 1e-12));
    bool rejected = false;
    try
    {
        Chi0 mismatched(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, grid,
                        ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("chi0 accepted a grid at the wrong FD temperature");
}

void check_direct_bandpair_tetrahedron_reference(const Fixture &fixture)
{
    constexpr int nk1 = 2;
    constexpr int nk2 = 2;
    constexpr int nk3 = 2;
    constexpr int nk = nk1 * nk2 * nk3;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double kbt = 1.0 / fixture.beta;

    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, nk, 2, 2);
    auto ds = librpa_int::api::get_dataset_instance(handler);
    auto &mf = ds->mf;
    mf.get_efermi() = 0.0;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(kbt, 0.0, 2.0, 1e-12));
    ds->basis_wfc.set({2});
    ds->basis_aux.set({2});
    ds->pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    std::vector<double> kvecs;
    const std::array<int, nk> k_order{7, 2, 5, 0, 6, 3, 1, 4};
    for (int ik = 0; ik != nk; ++ik)
    {
        const int canonical = k_order[ik];
        const int ix = canonical / (nk2 * nk3);
        const int iy = (canonical / nk3) % nk2;
        const int iz = canonical % nk3;
        const double kx = static_cast<double>(ix) / nk1;
        const double ky = static_cast<double>(iy) / nk2;
        const double kz = static_cast<double>(iz) / nk3;
        kvecs.insert(kvecs.end(), {two_pi * kx, two_pi * ky, two_pi * kz});
        mf.get_eigenvals()[0](ik, 0) = -0.21 + 0.04 * std::cos(two_pi * kx) -
                                       0.03 * std::cos(two_pi * ky) + 0.02 * std::cos(two_pi * kz);
        mf.get_eigenvals()[0](ik, 1) = 0.19 + 0.05 * std::cos(two_pi * kx) +
                                       0.02 * std::cos(two_pi * ky) - 0.04 * std::cos(two_pi * kz);
        auto &u = mf.get_eigenvectors()[0][0][ik];
        u.create(2, 2);
        u.zero_out();
        const double theta = 0.19 + 0.07 * kx - 0.05 * ky + 0.03 * kz;
        const auto phase = std::polar(1.0, two_pi * (kx + 2.0 * ky - kz));
        u(0, 0) = std::cos(theta);
        u(0, 1) = std::sin(theta) * phase;
        u(1, 0) = -std::sin(theta) * std::conj(phase);
        u(1, 1) = std::cos(theta);
        for (int band = 0; band != 2; ++band)
            mf.get_weight()[0](ik, band) =
                2.0 * fermi_dirac_occupation(mf.get_eigenvals()[0](ik, band), kbt) / nk;
    }
    ds->pbc.set_kgrids_kvec(nk1, nk2, nk3, kvecs);
    std::vector<double> energies(mf.get_eigenvals()[0].c, mf.get_eigenvals()[0].c + 2 * nk);
    std::vector<double> occupations(2 * nk);
    for (int index = 0; index != 2 * nk; ++index)
        occupations[index] = mf.get_weight()[0].c[index] * nk;
    handler.set_wg_ekb_efermi(1, nk, 2, occupations.data(), energies.data(), 0.0);

    std::vector<double> transform_real(fixture.transform.size),
        transform_imag(fixture.transform.size);
    for (int index = 0; index != fixture.transform.size; ++index)
    {
        transform_real[index] = fixture.transform.c[index].real();
        transform_imag[index] = fixture.transform.c[index].imag();
    }
    handler.set_external_thermal_time_grid(
        fixture.beta, fixture.wmax, fixture.tolerance, fixture.transform.nr, fixture.transform.nc,
        fixture.times.data(), transform_real.data(), transform_imag.data());
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = fixture.transform.nr;
    options.ntau = fixture.transform.nc;
    initialize_ds_tfgrids(*ds, options);

    constexpr double lri_vertex[2][2][2] = {
        {{1.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.6}, {0.6, 0.3}},
    };
    constexpr double nonlocal_lri_vertex[2][2][2] = {
        {{0.08, -0.03}, {0.04, 0.02}},
        {{-0.02, 0.05}, {0.03, -0.06}},
    };
    Cs_LRI cs;
    cs.use_libri = true;
    if (ds->comm_h.is_root())
    {
        auto coefficients = std::make_shared<std::valarray<double>>(8);
        for (int mu = 0; mu != 2; ++mu)
            for (int row = 0; row != 2; ++row)
                for (int col = 0; col != 2; ++col)
                    (*coefficients)[(mu * 2 + row) * 2 + col] = 0.5 * lri_vertex[mu][row][col];
        cs.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 2, 2}, coefficients);
        auto nonlocal_coefficients = std::make_shared<std::valarray<double>>(8);
        for (int mu = 0; mu != 2; ++mu)
            for (int row = 0; row != 2; ++row)
                for (int col = 0; col != 2; ++col)
                    (*nonlocal_coefficients)[(mu * 2 + row) * 2 + col] =
                        nonlocal_lri_vertex[mu][row][col];
        cs.data_libri[0][{0, {1, 0, 0}}] = RI::Tensor<double>({2, 2, 2}, nonlocal_coefficients);
    }

    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    Chi0 chi_uniform(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, ds->tfg,
                     ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi_uniform.gf_threshold = 0.0;
    chi_uniform.build(LIBRPA_ROUTING_LIBRI, cs, {{0, 0}}, ds->basis_aux, no_shrink, ds->blacs_h);

    Chi0 chi(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, ds->tfg,
             ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi.gf_threshold = 0.0;
    setenv("LIBRPA_DIRECT_CHI0_BANDPAIR_TETRA_REFERENCE", "enabled", 1);
    chi.build(LIBRPA_ROUTING_LIBRI, cs, {{0, 0}}, ds->basis_aux, no_shrink, ds->blacs_h);
    unsetenv("LIBRPA_DIRECT_CHI0_BANDPAIR_TETRA_REFERENCE");

    const auto quadrature =
        build_periodic_tetrahedron_quadrature(ds->pbc.period, ds->pbc.kfrac_list);
    const auto &reference = mf.get_fermi_dirac_reference();
    const auto &frequencies = ds->tfg.get_freq_nodes();
    double max_error = 0.0;
    double max_uniform_difference = 0.0;
    int checked = 0;
    for (const auto &[frequency, qblocks] : chi.get_chi0_q())
    {
        const auto ifrequency = ds->tfg.get_freq_index(frequency);
        if (ifrequency < 0 || frequencies.at(static_cast<std::size_t>(ifrequency)) != frequency)
            throw std::runtime_error(
                "tetrahedron reference returned an unknown Matsubara frequency");
        for (const auto &[q, blocks] : qblocks)
        {
            std::vector<int> kplusq(nk, -1);
            const auto qfrac = ds->pbc.latvec * q;
            for (int source = 0; source != nk; ++source)
                for (int target = 0; target != nk; ++target)
                    if (nearly_integer_vector(
                            ds->pbc.kfrac_list[static_cast<std::size_t>(source)] + qfrac -
                                ds->pbc.kfrac_list[static_cast<std::size_t>(target)],
                            1e-10))
                    {
                        kplusq[static_cast<std::size_t>(source)] = target;
                        break;
                    }
            for (const int target : kplusq)
                if (target < 0)
                    throw std::runtime_error("tetrahedron reference could not map k plus q");

            const auto lri_matrix = [&](const int mu, const int source)
            {
                ComplexMatrix result(2, 2, true);
                const double kx = ds->pbc.kfrac_list[static_cast<std::size_t>(source)].x;
                const auto phase = std::polar(1.0, two_pi * kx);
                for (int row = 0; row != 2; ++row)
                    for (int col = 0; col != 2; ++col)
                        result(row, col) = 0.5 * lri_vertex[mu][row][col] +
                                           phase * nonlocal_lri_vertex[mu][row][col];
                return result;
            };
            const auto density_vertex =
                [&](const int mu, const int source, const int n, const int m)
            {
                const int target = kplusq[static_cast<std::size_t>(source)];
                const auto c_source = lri_matrix(mu, source);
                const auto c_target = lri_matrix(mu, target);
                const auto &u_source = mf.get_eigenvectors().at(0).at(0).at(source);
                const auto &u_target = mf.get_eigenvectors().at(0).at(0).at(target);
                std::complex<double> value = 0.0;
                for (int row = 0; row != 2; ++row)
                    for (int col = 0; col != 2; ++col)
                        value += std::conj(u_source(n, row)) *
                                 (c_target(row, col) + std::conj(c_source(col, row))) *
                                 u_target(m, col);
                return value;
            };

            ComplexMatrix expected(2, 2, true);
            for (const auto &point : quadrature)
                for (int n = 0; n != 2; ++n)
                {
                    double energy_n = 0.0;
                    for (int vertex = 0; vertex != 4; ++vertex)
                        energy_n +=
                            point.barycentric[vertex] *
                            mf.get_eigenvals().at(0)(static_cast<int>(point.vertices[vertex]), n);
                    for (int m = 0; m != 2; ++m)
                    {
                        double energy_m = 0.0;
                        for (int vertex = 0; vertex != 4; ++vertex)
                        {
                            const auto source = point.vertices[vertex];
                            energy_m += point.barycentric[vertex] *
                                        mf.get_eigenvals().at(0)(kplusq[source], m);
                        }
                        const auto kernel = finite_temperature_tetrahedron_kernel(
                            energy_n, energy_m, frequency, reference);
                        for (int mu = 0; mu != 2; ++mu)
                            for (int nu = 0; nu != 2; ++nu)
                            {
                                std::complex<double> matrix_element = 0.0;
                                for (int vertex = 0; vertex != 4; ++vertex)
                                {
                                    const int source = static_cast<int>(point.vertices[vertex]);
                                    matrix_element += point.barycentric[vertex] *
                                                      density_vertex(mu, source, n, m) *
                                                      std::conj(density_vertex(nu, source, n, m));
                                }
                                expected(mu, nu) += 2.0 * point.weight * kernel * matrix_element;
                            }
                    }
                }
            for (const auto &[atom_i, row] : blocks)
                for (const auto &[atom_j, actual] : row)
                {
                    if (atom_i != 0 || atom_j != 0)
                        throw std::runtime_error(
                            "tetrahedron reference returned an unexpected atom block");
                    for (int mu = 0; mu != 2; ++mu)
                        for (int nu = 0; nu != 2; ++nu)
                        {
                            max_error =
                                std::max(max_error, std::abs(actual(mu, nu) - expected(mu, nu)));
                            const auto &uniform =
                                chi_uniform.get_chi0_q().at(frequency).at(q).at(atom_i).at(atom_j);
                            max_uniform_difference =
                                std::max(max_uniform_difference,
                                         std::abs(uniform(mu, nu) - expected(mu, nu)));
                            ++checked;
                        }
                }
        }
    }
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_error, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_uniform_difference, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &checked, 1, MPI_SUM);
    if (ds->comm_h.is_root())
        std::cout << "Direct band-pair tetrahedron max residual: " << max_error
                  << "; uniform-grid/reference max difference: " << max_uniform_difference
                  << "; matrix elements checked: " << checked << std::endl;
    if (checked < fixture.transform.nr * nk * 4 || max_error > 5e-12 ||
        max_uniform_difference < 1e-8)
        throw std::runtime_error(
            "direct band-pair tetrahedron test did not distinguish the uniform response");
}

// This compares the expanded-R LibRI CGGC path against the reciprocal-space
// integral of exactly the same piecewise-linear AO Green's functions.  It is
// deliberately simpler than the Adler-Wiser band-pair oracle above: the two
// references answer different interpolation questions.
void check_realspace_tetrahedron_green_function_reference(const Fixture &fixture,
                                                          bool nonlocal_coefficient)
{
    constexpr int nk1 = 2;
    constexpr int nk2 = 2;
    constexpr int nk3 = 2;
    constexpr int nk = nk1 * nk2 * nk3;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double kbt = 1.0 / fixture.beta;

    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, nk, 1, 1);
    auto ds = librpa_int::api::get_dataset_instance(handler);
    auto &mf = ds->mf;
    mf.get_efermi() = 0.0;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(kbt, 0.0, 2.0, 1e-12));
    ds->basis_wfc.set({1});
    ds->basis_aux.set({1});
    ds->pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    std::vector<double> kvecs;
    const std::array<int, nk> k_order{7, 2, 5, 0, 6, 3, 1, 4};
    for (int ik = 0; ik != nk; ++ik)
    {
        const int canonical = k_order[ik];
        const int ix = canonical / (nk2 * nk3);
        const int iy = (canonical / nk3) % nk2;
        const int iz = canonical % nk3;
        const double kx = static_cast<double>(ix) / nk1;
        const double ky = static_cast<double>(iy) / nk2;
        const double kz = static_cast<double>(iz) / nk3;
        kvecs.insert(kvecs.end(), {two_pi * kx, two_pi * ky, two_pi * kz});
        mf.get_eigenvals()[0](ik, 0) = -0.08 + 0.13 * std::cos(two_pi * kx) -
                                       0.07 * std::cos(two_pi * ky) + 0.05 * std::cos(two_pi * kz);
        mf.get_eigenvectors()[0][0][ik].create(1, 1);
        mf.get_eigenvectors()[0][0][ik](0, 0) = 1.0;
        mf.get_weight()[0](ik, 0) =
            2.0 * fermi_dirac_occupation(mf.get_eigenvals()[0](ik, 0), kbt) / nk;
    }
    ds->pbc.set_kgrids_kvec(nk1, nk2, nk3, kvecs);
    std::vector<double> energies(mf.get_eigenvals()[0].c, mf.get_eigenvals()[0].c + nk);
    std::vector<double> occupations(nk);
    for (int index = 0; index != nk; ++index) occupations[index] = mf.get_weight()[0].c[index] * nk;
    handler.set_wg_ekb_efermi(1, nk, 1, occupations.data(), energies.data(), 0.0);

    std::vector<double> transform_real(fixture.transform.size),
        transform_imag(fixture.transform.size);
    for (int index = 0; index != fixture.transform.size; ++index)
    {
        transform_real[index] = fixture.transform.c[index].real();
        transform_imag[index] = fixture.transform.c[index].imag();
    }
    handler.set_external_thermal_time_grid(
        fixture.beta, fixture.wmax, fixture.tolerance, fixture.transform.nr, fixture.transform.nc,
        fixture.times.data(), transform_real.data(), transform_imag.data());
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = fixture.transform.nr;
    options.ntau = fixture.transform.nc;
    initialize_ds_tfgrids(*ds, options);

    Cs_LRI cs;
    cs.use_libri = true;
    constexpr double neighbor_coefficient = 0.13;
    if (ds->comm_h.is_root())
    {
        auto coefficient = std::make_shared<std::valarray<double>>(1);
        (*coefficient)[0] = 0.5;
        cs.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1, 1, 1}, coefficient);
        if (nonlocal_coefficient)
        {
            auto neighbor = std::make_shared<std::valarray<double>>(1);
            (*neighbor)[0] = neighbor_coefficient;
            cs.data_libri[0][{0, {1, 0, 0}}] = RI::Tensor<double>({1, 1, 1}, neighbor);
        }
    }

    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    Chi0 chi_uniform(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, ds->tfg,
                     ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi_uniform.gf_threshold = 0.0;
    chi_uniform.build(LIBRPA_ROUTING_LIBRI, cs, {{0, 0}}, ds->basis_aux, no_shrink, ds->blacs_h);
    Chi0 chi(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, ds->tfg,
             ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi.gf_threshold = 0.0;
    const char *image_factor = std::getenv("LIBRPA_TETRA_TEST_IMAGE_FACTOR");
    setenv("LIBRPA_TETRA_FULL_GW", "enabled", 1);
    setenv("LIBRPA_TETRA_IMAGE_FACTOR", image_factor == nullptr ? "5" : image_factor, 1);
    chi.build(LIBRPA_ROUTING_LIBRI, cs, {{0, 0}}, ds->basis_aux, no_shrink, ds->blacs_h);
    unsetenv("LIBRPA_TETRA_IMAGE_FACTOR");
    unsetenv("LIBRPA_TETRA_FULL_GW");

    const auto quadrature =
        build_high_order_periodic_tetrahedron_quadrature(ds->pbc.period, ds->pbc.kfrac_list);
    double max_error = 0.0;
    std::complex<double> max_actual = 0.0;
    std::complex<double> max_expected = 0.0;
    double max_frequency = 0.0;
    Vector3_Order<double> max_q{0.0, 0.0, 0.0};
    double max_uniform_error = 0.0;
    int checked = 0;
    for (const auto &[frequency, qblocks] : chi.get_chi0_q())
        for (const auto &[q, blocks] : qblocks)
        {
            std::vector<int> kplusq(nk, -1);
            const auto qfrac = ds->pbc.latvec * q;
            for (int source = 0; source != nk; ++source)
                for (int target = 0; target != nk; ++target)
                    if (nearly_integer_vector(
                            ds->pbc.kfrac_list[static_cast<std::size_t>(source)] + qfrac -
                                ds->pbc.kfrac_list[static_cast<std::size_t>(target)],
                            1e-10))
                    {
                        kplusq[static_cast<std::size_t>(source)] = target;
                        break;
                    }
            for (const int target : kplusq)
                if (target < 0)
                    throw std::runtime_error(
                        "real-space tetrahedron reference could not map k plus q");

            const auto density_vertex = [&](const Vector3_Order<double> &kfrac)
            {
                std::complex<double> vertex = 1.0;
                if (nonlocal_coefficient)
                {
                    const auto phase_source = std::polar(1.0, two_pi * kfrac.x);
                    const auto phase_target = std::polar(1.0, two_pi * (kfrac.x + qfrac.x));
                    vertex += neighbor_coefficient * (phase_target + std::conj(phase_source));
                }
                return vertex;
            };
            std::complex<double> expected = 0.0;
            std::complex<double> expected_uniform = 0.0;
            for (int itime = 0; itime != ds->tfg.get_n_time_grids(); ++itime)
            {
                std::complex<double> response_tau = 0.0;
                std::complex<double> response_uniform_tau = 0.0;
                for (int source = 0; source != nk; ++source)
                {
                    const int target = kplusq[static_cast<std::size_t>(source)];
                    const double green_positive = thermal_green_amplitude(
                        mf.get_eigenvals()[0](target, 0), fixture.times[itime], kbt);
                    const double green_negative = -thermal_green_amplitude(
                        mf.get_eigenvals()[0](source, 0), -fixture.times[itime], kbt);
                    const auto vertex =
                        density_vertex(ds->pbc.kfrac_list[static_cast<std::size_t>(source)]);
                    response_uniform_tau += green_positive * green_negative * vertex *
                                            std::conj(vertex) / static_cast<double>(nk);
                }
                for (const auto &point : quadrature)
                {
                    double green_positive = 0.0;
                    double green_negative = 0.0;
                    for (int vertex = 0; vertex != 4; ++vertex)
                    {
                        const int source = static_cast<int>(point.vertices[vertex]);
                        const int target = kplusq[static_cast<std::size_t>(source)];
                        const double bary = point.barycentric[vertex];
                        green_positive +=
                            bary * thermal_green_amplitude(mf.get_eigenvals()[0](target, 0),
                                                           fixture.times[itime], kbt);
                        green_negative -=
                            bary * thermal_green_amplitude(mf.get_eigenvals()[0](source, 0),
                                                           -fixture.times[itime], kbt);
                    }
                    const auto vertex = density_vertex(point.fractional_coordinate);
                    response_tau +=
                        point.weight * green_positive * green_negative * vertex * std::conj(vertex);
                }
                const auto time_to_frequency = ds->tfg.get_time_to_frequency_factor(
                    static_cast<std::size_t>(ds->tfg.get_freq_index(frequency)),
                    static_cast<std::size_t>(itime));
                expected += 2.0 * time_to_frequency * response_tau;
                expected_uniform += 2.0 * time_to_frequency * response_uniform_tau;
            }
            const auto &actual = blocks.at(0).at(0);
            const auto &actual_uniform = chi_uniform.get_chi0_q().at(frequency).at(q).at(0).at(0);
            const double error = std::abs(actual(0, 0) - expected);
            max_uniform_error =
                std::max(max_uniform_error, std::abs(actual_uniform(0, 0) - expected_uniform));
            if (error > max_error)
            {
                max_error = error;
                max_actual = actual(0, 0);
                max_expected = expected;
                max_frequency = frequency;
                max_q = q;
            }
            ++checked;
        }
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_error, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_uniform_error, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &checked, 1, MPI_SUM);
    if (ds->comm_h.is_root())
        std::cout << "Real-space tetrahedron Green-function max residual: " << max_error
                  << "; actual = " << max_actual << "; expected = " << max_expected
                  << "; q = " << max_q << "; frequency = " << max_frequency
                  << "; image factor = " << (image_factor == nullptr ? "5" : image_factor)
                  << "; nonlocal coefficient = " << nonlocal_coefficient
                  << "; uniform LRI residual = " << max_uniform_error
                  << "; matrix elements checked: " << checked << std::endl;
    // A finite image tile truncates the algebraic Fourier tail of the
    // piecewise-linear interpolant.  The 5x5x5 test tile is an explicit
    // convergence control rather than a material-production cutoff.
    if (checked < fixture.transform.nr * nk || max_uniform_error > 5e-9 || max_error > 5e-4)
        throw std::runtime_error(
            "expanded-R LibRI response differs from its reciprocal tetrahedron Green-function "
            "reference");
}

void check_direct_bandpair_tetrahedron_multiatom_reference(const Fixture &fixture)
{
    constexpr int nk1 = 2;
    constexpr int nk2 = 2;
    constexpr int nk3 = 2;
    constexpr int nk = nk1 * nk2 * nk3;
    constexpr int nao = 3;
    const double two_pi = 2.0 * std::acos(-1.0);
    const double kbt = 1.0 / fixture.beta;

    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, nk, nao, nao);
    auto ds = librpa_int::api::get_dataset_instance(handler);
    auto &mf = ds->mf;
    mf.get_efermi() = 0.0;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(kbt, 0.0, 2.0, 1e-12));
    ds->basis_wfc.set(std::vector<std::size_t>{1, 2});
    ds->basis_aux.set(std::vector<std::size_t>{2, 1});
    const AtomicBasis full_aux(std::vector<std::size_t>{3, 2});
    ds->pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});

    std::vector<double> kvecs;
    const std::array<int, nk> k_order{3, 6, 0, 5, 2, 7, 1, 4};
    for (int ik = 0; ik != nk; ++ik)
    {
        const int canonical = k_order[ik];
        const int ix = canonical / (nk2 * nk3);
        const int iy = (canonical / nk3) % nk2;
        const int iz = canonical % nk3;
        const double kx = static_cast<double>(ix) / nk1;
        const double ky = static_cast<double>(iy) / nk2;
        const double kz = static_cast<double>(iz) / nk3;
        kvecs.insert(kvecs.end(), {two_pi * kx, two_pi * ky, two_pi * kz});
        for (int band = 0; band != nao; ++band)
        {
            mf.get_eigenvals()[0](ik, band) = -0.27 + 0.24 * band + 0.05 * std::cos(two_pi * kx) -
                                              0.03 * std::cos(two_pi * ky) +
                                              0.02 * (band + 1) * std::cos(two_pi * kz);
            mf.get_weight()[0](ik, band) =
                2.0 * fermi_dirac_occupation(mf.get_eigenvals()[0](ik, band), kbt) / nk;
        }
        auto &u = mf.get_eigenvectors()[0][0][ik];
        u.create(nao, nao);
        u.zero_out();
        const double theta = 0.16 + 0.09 * kx - 0.04 * ky + 0.03 * kz;
        const auto phase = std::polar(1.0, two_pi * (kx - 2.0 * ky + kz));
        u(0, 0) = std::cos(theta);
        u(0, 1) = std::sin(theta) * phase;
        u(1, 0) = -std::sin(theta) * std::conj(phase);
        u(1, 1) = std::cos(theta);
        u(2, 2) = std::polar(1.0, two_pi * (2.0 * kx + ky - kz));
    }
    ds->pbc.set_kgrids_kvec(nk1, nk2, nk3, kvecs);
    std::vector<double> energies(mf.get_eigenvals()[0].c, mf.get_eigenvals()[0].c + nao * nk);
    std::vector<double> occupations(nao * nk);
    for (int index = 0; index != nao * nk; ++index)
        occupations[index] = mf.get_weight()[0].c[index] * nk;
    handler.set_wg_ekb_efermi(1, nk, nao, occupations.data(), energies.data(), 0.0);

    std::vector<double> transform_real(fixture.transform.size),
        transform_imag(fixture.transform.size);
    for (int index = 0; index != fixture.transform.size; ++index)
    {
        transform_real[index] = fixture.transform.c[index].real();
        transform_imag[index] = fixture.transform.c[index].imag();
    }
    handler.set_external_thermal_time_grid(
        fixture.beta, fixture.wmax, fixture.tolerance, fixture.transform.nr, fixture.transform.nc,
        fixture.times.data(), transform_real.data(), transform_imag.data());
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = fixture.transform.nr;
    options.ntau = fixture.transform.nc;
    initialize_ds_tfgrids(*ds, options);

    const auto coefficient = [](const int i_atom, const int j_atom, const int cell_x,
                                const int aux_local, const int i_ao, const int j_ao)
    {
        return 0.018 * std::sin(0.31 + 0.73 * i_atom + 0.41 * j_atom + 0.29 * cell_x +
                                0.53 * aux_local + 0.67 * i_ao + 0.37 * j_ao) +
               0.011 * std::cos(0.19 + 0.47 * i_atom - 0.23 * j_atom + 0.17 * cell_x +
                                0.31 * aux_local - 0.13 * i_ao + 0.59 * j_ao);
    };
    Cs_LRI cs;
    cs.use_libri = true;
    if (ds->comm_h.is_root())
        for (int i_atom = 0; i_atom != 2; ++i_atom)
            for (int j_atom = 0; j_atom != 2; ++j_atom)
                for (int cell_x = -1; cell_x != 2; ++cell_x)
                {
                    RI::Tensor<double> block({full_aux.get_atom_nb(i_atom),
                                              ds->basis_wfc.get_atom_nb(i_atom),
                                              ds->basis_wfc.get_atom_nb(j_atom)});
                    for (int aux_local = 0;
                         aux_local != static_cast<int>(full_aux.get_atom_nb(i_atom)); ++aux_local)
                        for (int i_ao = 0;
                             i_ao != static_cast<int>(ds->basis_wfc.get_atom_nb(i_atom)); ++i_ao)
                            for (int j_ao = 0;
                                 j_ao != static_cast<int>(ds->basis_wfc.get_atom_nb(j_atom));
                                 ++j_ao)
                                block(aux_local, i_ao, j_ao) =
                                    coefficient(i_atom, j_atom, cell_x, aux_local, i_ao, j_ao);
                    cs.data_libri[i_atom][{j_atom, {cell_x, 0, 0}}] = std::move(block);
                }

    const std::vector<atpair_t> atpairs{{0, 0}, {0, 1}, {1, 1}};
    std::map<Vector3_Order<double>, ComplexMatrix> shrink_transforms;
    for (const auto &q : ds->pbc.klist_full)
    {
        ComplexMatrix transform(ds->basis_aux.nb_total, full_aux.nb_total, true);
        for (int small = 0; small != static_cast<int>(ds->basis_aux.nb_total); ++small)
            for (int large = 0; large != static_cast<int>(full_aux.nb_total); ++large)
                transform(small, large) =
                    0.07 * std::sin(0.23 + 0.41 * small + 0.17 * large + q.x) +
                    (small == large % static_cast<int>(ds->basis_aux.nb_total) ? 0.6 : 0.0);
        shrink_transforms.emplace(q, std::move(transform));
    }
    Chi0 chi(mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context, ds->tfg,
             ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, false);
    chi.gf_threshold = 0.0;
    setenv("LIBRPA_DIRECT_CHI0_BANDPAIR_TETRA_REFERENCE", "enabled", 1);
    chi.build(LIBRPA_ROUTING_LIBRI, cs, atpairs, full_aux, shrink_transforms, ds->blacs_h);
    unsetenv("LIBRPA_DIRECT_CHI0_BANDPAIR_TETRA_REFERENCE");

    const auto quadrature =
        build_periodic_tetrahedron_quadrature(ds->pbc.period, ds->pbc.kfrac_list);
    const auto &reference = mf.get_fermi_dirac_reference();
    double max_error = 0.0;
    int checked = 0;
    for (const auto &[frequency, qblocks] : chi.get_chi0_q())
        for (const auto &[q, blocks] : qblocks)
        {
            std::vector<int> kplusq(nk, -1);
            const auto qfrac = ds->pbc.latvec * q;
            for (int source = 0; source != nk; ++source)
                for (int target = 0; target != nk; ++target)
                    if (nearly_integer_vector(
                            ds->pbc.kfrac_list[static_cast<std::size_t>(source)] + qfrac -
                                ds->pbc.kfrac_list[static_cast<std::size_t>(target)],
                            1e-10))
                    {
                        kplusq[static_cast<std::size_t>(source)] = target;
                        break;
                    }
            for (const int target : kplusq)
                if (target < 0)
                    throw std::runtime_error(
                        "multiatom tetrahedron reference could not map k plus q");

            const auto transform_for_q = shrink_transforms.at(q);
            const auto lri_matrix = [&](const int aux_global, const int source)
            {
                ComplexMatrix result(nao, nao, true);
                const double kx = ds->pbc.kfrac_list[static_cast<std::size_t>(source)].x;
                for (int large_aux_atom = 0; large_aux_atom != 2; ++large_aux_atom)
                    for (int large_aux_local = 0;
                         large_aux_local != static_cast<int>(full_aux.get_atom_nb(large_aux_atom));
                         ++large_aux_local)
                    {
                        const int large_aux_global =
                            full_aux.get_global_index(large_aux_atom, large_aux_local);
                        const auto transform_coefficient =
                            transform_for_q(aux_global, large_aux_global);
                        for (int j_atom = 0; j_atom != 2; ++j_atom)
                            for (int cell_x = -1; cell_x != 2; ++cell_x)
                            {
                                const auto phase = std::polar(1.0, two_pi * kx * cell_x);
                                for (int i_ao = 0;
                                     i_ao !=
                                     static_cast<int>(ds->basis_wfc.get_atom_nb(large_aux_atom));
                                     ++i_ao)
                                    for (int j_ao = 0;
                                         j_ao !=
                                         static_cast<int>(ds->basis_wfc.get_atom_nb(j_atom));
                                         ++j_ao)
                                        result(ds->basis_wfc.get_global_index(large_aux_atom, i_ao),
                                               ds->basis_wfc.get_global_index(j_atom, j_ao)) +=
                                            transform_coefficient * phase *
                                            coefficient(large_aux_atom, j_atom, cell_x,
                                                        large_aux_local, i_ao, j_ao);
                            }
                    }
                return result;
            };
            const auto density_vertex =
                [&](const int aux_global, const int source, const int n, const int m)
            {
                const int target = kplusq[static_cast<std::size_t>(source)];
                const auto c_source = lri_matrix(aux_global, source);
                const auto c_target = lri_matrix(aux_global, target);
                const auto &u_source = mf.get_eigenvectors().at(0).at(0).at(source);
                const auto &u_target = mf.get_eigenvectors().at(0).at(0).at(target);
                std::complex<double> value = 0.0;
                for (int row = 0; row != nao; ++row)
                    for (int col = 0; col != nao; ++col)
                        value += std::conj(u_source(n, row)) *
                                 (c_target(row, col) + std::conj(c_source(col, row))) *
                                 u_target(m, col);
                return value;
            };

            for (const auto &[mu_atom, row] : blocks)
                for (const auto &[nu_atom, actual] : row)
                {
                    ComplexMatrix expected(ds->basis_aux.get_atom_nb(mu_atom),
                                           ds->basis_aux.get_atom_nb(nu_atom), true);
                    for (const auto &point : quadrature)
                        for (int n = 0; n != nao; ++n)
                        {
                            double energy_n = 0.0;
                            for (int vertex = 0; vertex != 4; ++vertex)
                                energy_n += point.barycentric[vertex] *
                                            mf.get_eigenvals().at(0)(
                                                static_cast<int>(point.vertices[vertex]), n);
                            for (int m = 0; m != nao; ++m)
                            {
                                double energy_m = 0.0;
                                for (int vertex = 0; vertex != 4; ++vertex)
                                {
                                    const auto source = point.vertices[vertex];
                                    energy_m += point.barycentric[vertex] *
                                                mf.get_eigenvals().at(0)(kplusq[source], m);
                                }
                                const auto kernel = finite_temperature_tetrahedron_kernel(
                                    energy_n, energy_m, frequency, reference);
                                for (int mu_local = 0;
                                     mu_local !=
                                     static_cast<int>(ds->basis_aux.get_atom_nb(mu_atom));
                                     ++mu_local)
                                    for (int nu_local = 0;
                                         nu_local !=
                                         static_cast<int>(ds->basis_aux.get_atom_nb(nu_atom));
                                         ++nu_local)
                                    {
                                        const int mu_global =
                                            ds->basis_aux.get_global_index(mu_atom, mu_local);
                                        const int nu_global =
                                            ds->basis_aux.get_global_index(nu_atom, nu_local);
                                        std::complex<double> matrix_element = 0.0;
                                        for (int vertex = 0; vertex != 4; ++vertex)
                                        {
                                            const int source =
                                                static_cast<int>(point.vertices[vertex]);
                                            matrix_element +=
                                                point.barycentric[vertex] *
                                                density_vertex(mu_global, source, n, m) *
                                                std::conj(density_vertex(nu_global, source, n, m));
                                        }
                                        expected(mu_local, nu_local) +=
                                            2.0 * point.weight * kernel * matrix_element;
                                    }
                            }
                        }
                    for (int mu_local = 0; mu_local != expected.nr; ++mu_local)
                        for (int nu_local = 0; nu_local != expected.nc; ++nu_local)
                        {
                            max_error = std::max(max_error, std::abs(actual(mu_local, nu_local) -
                                                                     expected(mu_local, nu_local)));
                            ++checked;
                        }
                }
        }
    ds->comm_h.allreduce(MPI_IN_PLACE, &max_error, 1, MPI_MAX);
    ds->comm_h.allreduce(MPI_IN_PLACE, &checked, 1, MPI_SUM);
    if (ds->comm_h.is_root())
        std::cout << "Direct multiatom band-pair tetrahedron max residual: " << max_error
                  << "; matrix elements checked: " << checked << std::endl;
    if (checked < fixture.transform.nr * nk * 7 || max_error > 5e-12)
        throw std::runtime_error("multiatom direct band-pair tetrahedron reference failed");
}
#endif
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    const auto fixture = read_fixture();
    check_green_pairs(fixture);
    check_validation(fixture);
#ifdef LIBRPA_USE_LIBRI
    check_full_libri_response(fixture, false);
    check_full_libri_response(fixture, true);
    check_full_libri_response(fixture, false, true);
    check_full_libri_response(fixture, true, true);
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size == 1)
    {
        check_direct_bandpair_tetrahedron_reference(fixture);
        check_realspace_tetrahedron_green_function_reference(fixture, false);
        check_realspace_tetrahedron_green_function_reference(fixture, true);
        check_direct_bandpair_tetrahedron_multiatom_reference(fixture);
    }
#endif
    MPI_Finalize();
}
