#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
#endif
    MPI_Finalize();
}
