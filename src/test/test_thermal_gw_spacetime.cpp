#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;
namespace
{
using Z = std::complex<double>;
const double PI = std::acos(-1.0);
void require(bool test, const char *message)
{
    if (!test) throw std::runtime_error(message);
}

void check(MAJOR major, int nk, int nabf = 2)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr double beta = 8;
    const std::vector<double> times{0.3, 1.2, 4.3, 7.7}, weights{0.6, 1.8, 3.2, 2.4};
    const std::vector<int> bosons{1, -1, 0}, fermions{2, -1, 0, -3, 4, -5};
    auto transform = ThermalGWTransform::from_quadrature(beta, times, weights, bosons, fermions);
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2, 0, 0, 0, 3, 0, 0, 0, 4});
    std::vector<double> kvec;
    for (int k = 0; k < nk; ++k)
    {
        // Public PBC setter takes physical reciprocal coordinates, including 2*pi.
        kvec.push_back(2 * PI * k / (2 * nk));
        kvec.push_back(0);
        kvec.push_back(0);
    }
    pbc.set_kgrids_kvec(nk, 1, 1, kvec);
    AtomicBasis ao(std::vector<std::size_t>{2}), abf(std::vector<std::size_t>{std::size_t(nabf)});
    MeanField mf(1, nk, 2, 2);
    mf.get_efermi() = 0.15;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / beta, 0.15, 2, 1e-12));
    for (int k = 0; k < nk; ++k)
    {
        mf.get_eigenvectors()[0][0][k].create(2, 2);
        for (int a = 0; a < 2; ++a)
        {
            const double xi = -0.3 + 0.4 * a + 0.07 * k;
            mf.get_eigenvals()[0](k, a) = xi + 0.15;
            mf.get_weight()[0](k, a) = 2.0 / (nk * (1 + std::exp(beta * xi)));
            mf.get_eigenvectors()[0][0][k](a, a) = 1;
        }
    }
    KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, nk);
    const auto desc = context.create_array_desc(nabf, nabf);
    TFGrids response_grid;  // Deliberately uninitialized: no legacy grid may be consulted.
    SymmetryContext symmetry;
    G0W0 gw(mf, ao, pbc, symmetry, response_grid, context, context, desc, false, false);
    Matz cached(1, 1, MAJOR::ROW);
    cached(0, 0) = Z(3, -7);
    gw.sigc_is_ik_f_KS[7][5][123] = cached;
    Cs_LRI cs;
    cs.use_libri = true;
    const double c[2][2][2] = {{{0.3, 0.11}, {-0.06, 0.22}}, {{-0.12, 0.09}, {0.18, 0.27}}};
    if (rank == 0)
    {
        RI::Tensor<double> tensor({std::size_t(nabf), 2, 2});
        for (int u = 0; u < nabf; ++u)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) tensor(u, a, b) = c[u][a][b];
        cs.data_libri[0][{0, {0, 0, 0}}] = std::move(tensor);
    }
    const auto sample = [](int m, int q, int u, int v)
    { return Z(0.02 * (1 + m + 2 * q + 3 * u - v), 0.03 * (2 - m + q - u + 2 * v)); };
    std::map<double, std::map<Vector3_Order<double>, Matz>> wc;
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    for (std::size_t m = 0; m < bosons.size(); ++m)
        for (int q = 0; q < nk; ++q)
        {
            Matz mat(desc.m_loc(), desc.n_loc(), major);
            for (int i = 0; i < desc.m_loc(); ++i)
                for (int j = 0; j < desc.n_loc(); ++j)
                    mat(i, j) = sample(bosons[m], q, desc.indx_l2g_r(i), desc.indx_l2g_c(j));
            wc[frequencies[m]][pbc.klist_full[q]] = std::move(mat);
        }
    const auto rejects_collectively = [&]()
    {
        int caught = 0;
        try
        {
            (void)gw.build_thermal_spacetime(abf, cs, wc, desc, transform);
        }
        catch (const std::exception &)
        {
            caught = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, &caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        require(caught == size, "invalid rank-local thermal input did not fail on every rank");
    };
    if (rank == size - 1)
        mf.set_fermi_dirac_reference(make_fermi_dirac_reference(2 / beta, 0.15, 2, 1e-12));
    rejects_collectively();
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / beta, 0.15, 2, 1e-12));
    const auto energy = mf.get_eigenvals()[0](0, 0);
    if (rank == size - 1) mf.get_eigenvals()[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    rejects_collectively();
    mf.get_eigenvals()[0](0, 0) = energy;
    if (rank == size - 1) gw.libri_threshold_G = -1;
    rejects_collectively();
    gw.libri_threshold_G = 0;
    auto saved_cs = cs.data_libri;
    if (rank == 0) cs.data_libri.at(0).begin()->second.data.reset();
    rejects_collectively();
    cs.data_libri = saved_cs;
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    const auto wfc_size = wfc.size;
    if (rank == size - 1) wfc.size = 0;
    rejects_collectively();
    wfc.size = wfc_size;
    const auto eigenvalues = mf.get_eigenvals()[0];
    if (rank == size - 1) mf.get_eigenvals()[0] = matrix();
    rejects_collectively();
    mf.get_eigenvals()[0] = eigenvalues;
    const auto result = gw.build_thermal_spacetime(abf, cs, wc, desc, transform);
    require(result.beta_ha_inv == beta && result.fermionic_indices == fermions,
            "thermal Sigma lost beta or signed label order");
    require(wc.size() == bosons.size() && !gw.is_rspace_built(),
            "thermal reference mutated input or legacy GW state");
    require(gw.sigc_is_ik_f_KS.size() == 1 && gw.sigc_is_ik_f_KS.at(7).at(5).at(123)(0, 0) == Z(3, -7),
            "thermal reference altered existing legacy KS data");
    for (std::size_t m = 0; m < bosons.size(); ++m)
        for (int q = 0; q < nk; ++q)
            for (int a = 0; a < desc.m_loc(); ++a)
                for (int b = 0; b < desc.n_loc(); ++b)
                    require(wc.at(frequencies[m]).at(pbc.klist_full[q])(a, b) ==
                                sample(bosons[m], q, desc.indx_l2g_r(a), desc.indx_l2g_c(b)),
                            "thermal reference changed a W input element");
    if (rank == 0)
        for (int u = 0; u < nabf; ++u)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    require(cs.data_libri.at(0).begin()->second(u, a, b) == c[u][a][b],
                            "thermal reference changed an LRI coefficient");
    // Destroy caller payloads before reading the returned matrices.
    wc.clear();
    cs.data_libri.clear();
    mf.get_eigenvectors().clear();
    double largest_error = 0;
    const auto ff = transform.get_fermionic_frequencies_ha();
    for (const auto &r : pbc.Rlist)
        for (std::size_t n = 0; n < fermions.size(); ++n)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                {
                    Z actual = 0;
                    if (result.blocks.count(0) && result.blocks.at(0).count(ff[n]))
                    {
                        const auto &pairs = result.blocks.at(0).at(ff[n]);
                        if (pairs.count({0, 0}) && pairs.at({0, 0}).count(r))
                            actual = pairs.at({0, 0}).at(r)(a, b);
                    }
                    MPI_Allreduce(MPI_IN_PLACE, &actual, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                                  MPI_COMM_WORLD);
                    Z expected = 0;
                    for (std::size_t t = 0; t < times.size(); ++t)
                        for (int u = 0; u < nabf; ++u)
                            for (int v = 0; v < nabf; ++v)
                                for (int band = 0; band < 2; ++band)
                                {
                                    Z g = 0, w = 0;
                                    for (int k = 0; k < nk; ++k)
                                    {
                                        const double xi = -0.3 + 0.4 * band + 0.07 * k;
                                        g += std::exp(Z(0, -2 * PI * k * r.x / nk)) / double(nk) *
                                             std::exp(-xi * times[t]) / (1 + std::exp(-beta * xi));
                                    }
                                    for (int q = 0; q < nk; ++q)
                                        for (int m : bosons)
                                            w += sample(m, q, u, v) *
                                                 std::exp(Z(0, -2 * PI *
                                                                   (q * r.x / double(nk) +
                                                                    m * times[t] / beta))) /
                                                 (beta * nk);
                                    expected += weights[t] * std::exp(Z(0, ff[n] * times[t])) *
                                                (c[u][a][band] + c[u][band][a]) *
                                                (c[v][b][band] + c[v][band][b]) * g * w;
                                }
                    largest_error = std::max(largest_error, std::abs(actual - expected));
                }
    require(largest_error < 2e-12, "production-shaped W/B/LibRI/F differs from explicit sums");
    if (rank == 0)
        std::cout << "thermal G0W0 data flow nk=" << nk << " max_abs=" << largest_error << '\n';
}
}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    int status = 0;
    try
    {
        check(MAJOR::ROW, 1);
        check(MAJOR::COL, 3);
        check(MAJOR::ROW, 3, 1);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return status;
}
