#include <mpi.h>

#include <algorithm>
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
    require(result.fermionic_grid.get_beta_ha_inv() == beta &&
                result.fermionic_grid.get_indices() == fermions &&
                result.fermionic_grid.get_frequencies_ha() == transform.get_fermionic_frequencies_ha(),
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
                    if (result.blocks.count(0) && result.blocks.at(0).count(fermions[n]))
                    {
                        const auto &pairs = result.blocks.at(0).at(fermions[n]);
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

void check_multiatom(MAJOR major, bool distribute_cs)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto require_all = [&](bool condition, const char *message)
    {
        int failed = !condition;
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(failed == 0, message);
    };
    constexpr int NK = 3, NSPIN = 2, NAO = 3, NABF = 3;
    constexpr int NAO_SUPER = NK * NAO, NABF_SUPER = NK * NABF;
    constexpr double BETA = 8.0, MU = 0.15;
    const std::vector<double> times{0.3, 1.2, 4.3, 7.7}, weights{0.6, 1.8, 3.2, 2.4};
    const std::vector<int> bosons{1, -2, 0, 2, -1}, fermions{2, -1, 0, -3, 4, -5};
    const auto transform =
        ThermalGWTransform::from_quadrature(BETA, times, weights, bosons, fermions);
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2.5, 0, 0, 0.6, 3, 0, 0.2, 0.4, 4});
    std::vector<double> kvec;
    for (int k = 0; k < NK; ++k)
    {
        const double factor = 2 * PI * k / NK;
        kvec.push_back(factor * pbc.G.e11);
        kvec.push_back(factor * pbc.G.e12);
        kvec.push_back(factor * pbc.G.e13);
    }
    pbc.set_kgrids_kvec(NK, 1, 1, kvec);
    AtomicBasis ao(std::vector<std::size_t>{1, 2}), abf(std::vector<std::size_t>{2, 1});
    const auto xi = [](int spin, int k, int band)
    { return -0.31 + 0.27 * band + 0.08 * k + 0.04 * spin + 0.015 * band * k; };
    const auto orbital = [](int spin, int k, int band, int a)
    {
        return std::polar(1 / std::sqrt(3.0), 2 * PI * band * a / 3 + 0.13 * a * (k + 1) +
                                                  0.07 * spin * (a + 1) + 0.03 * k * a * a);
    };
    const auto coefficient = [](int i, int j, int r, int u, int a, int b)
    {
        return 0.035 *
                   std::sin(0.4 + 0.7 * i + 1.1 * j + 0.23 * r + 0.31 * u + 0.43 * a + 0.61 * b) +
               0.017 * (i - j + r);
    };
    const auto sample_w = [](int m, int q, int u, int v)
    {
        return Z(0.02 * (1 + m + 2 * q + 3 * u - v) + 0.007 * std::cos(0.3 * m * (u + 1)),
                 0.014 * (2 - m + q - 2 * u + 3 * v));
    };
    MeanField mf(NSPIN, NK, NAO, NAO);
    mf.get_efermi() = MU;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / BETA, MU, 1, 1e-12));
    for (int spin = 0; spin < NSPIN; ++spin)
        for (int k = 0; k < NK; ++k)
        {
            auto &wfc = mf.get_eigenvectors()[spin][0][k];
            wfc.create(NAO, NAO);
            for (int band = 0; band < NAO; ++band)
            {
                mf.get_eigenvals()[spin](k, band) = xi(spin, k, band) + MU;
                mf.get_weight()[spin](k, band) =
                    1 / (NK * (1 + std::exp(BETA * xi(spin, k, band))));
                for (int a = 0; a < NAO; ++a) wfc(band, a) = orbital(spin, k, band, a);
            }
        }
    KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, NK);
    const auto desc = context.create_array_desc(NABF, NABF);
    TFGrids legacy_grid;
    SymmetryContext symmetry;
    G0W0 gw(mf, ao, pbc, symmetry, legacy_grid, context, context, desc, false, false);
    Cs_LRI cs;
    cs.use_libri = true;
    int block_index = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int r = -1; r <= 1; ++r)
            {
                const int owner = distribute_cs ? block_index % size : 0;
                ++block_index;
                if (rank != owner) continue;
                RI::Tensor<double> block(
                    {abf.get_atom_nb(i), ao.get_atom_nb(i), ao.get_atom_nb(j)});
                for (int u = 0; u < int(abf.get_atom_nb(i)); ++u)
                    for (int a = 0; a < int(ao.get_atom_nb(i)); ++a)
                        for (int b = 0; b < int(ao.get_atom_nb(j)); ++b)
                            block(u, a, b) = coefficient(i, j, r, u, a, b);
                cs.data_libri[i][{j, {r, 0, 0}}] = std::move(block);
            }
    std::map<double, std::map<Vector3_Order<double>, Matz>> wc;
    const auto bosonic_frequencies = transform.get_bosonic_frequencies_ha();
    for (std::size_t m = 0; m < bosons.size(); ++m)
        for (int q = 0; q < NK; ++q)
        {
            Matz block(desc.m_loc(), desc.n_loc(), major);
            for (int u = 0; u < block.nr(); ++u)
                for (int v = 0; v < block.nc(); ++v)
                    block(u, v) = sample_w(bosons[m], q, desc.indx_l2g_r(u), desc.indx_l2g_c(v));
            wc[bosonic_frequencies[m]][pbc.klist_full[q]] = std::move(block);
        }

    // Cs_LRI stores C[I][J,R](u,a,b): ABF u and AO a on I, AO b on J+R.
    // Both ordered pairs are supplied independently, with real first-center C.
    // GW.hpp contracts W(a0,b0) with G(a1,b1), G(a1,b2), G(a2,b1), G(a2,b2).
    // Equivalently, translate C onto the finite periodic supercell and define
    // D(U;x,p) = C(U;x,p) + C(U;p,x), with C zero unless U and its first AO share a site.
    // Then Sigma(x,y;t) = sum_{p,q,U,V} D(U;x,p) G_lib(p,q;t) W(U,V;t) D(V;y,q).
    // This counts both attachments even onsite; no extra 1/2, spin factor or conjugation of C.
    // G_lib is the positive-tau amplitude used by GW, i.e. minus the physical G(tau>0).
    std::vector<double> vertices(NABF_SUPER * NAO_SUPER * NAO_SUPER, 0.0);
    const auto vertex = [&](int u, int a, int b) -> double &
    { return vertices[(u * NAO_SUPER + a) * NAO_SUPER + b]; };
    const auto cell = [](int r) { return (r % NK + NK) % NK; };
    for (int origin = 0; origin < NK; ++origin)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int r = -1; r <= 1; ++r)
                    for (int u = 0; u < int(abf.get_atom_nb(i)); ++u)
                        for (int a = 0; a < int(ao.get_atom_nb(i)); ++a)
                            for (int b = 0; b < int(ao.get_atom_nb(j)); ++b)
                            {
                                const int su = origin * NABF + abf.get_global_index(i, u);
                                const int sa = origin * NAO + ao.get_global_index(i, a);
                                const int sb = cell(origin + r) * NAO + ao.get_global_index(j, b);
                                const double c = coefficient(i, j, r, u, a, b);
                                vertex(su, sa, sb) += c;
                                vertex(su, sb, sa) += c;
                            }
    const int nfreq = fermions.size();
    const auto output_index = [=](int spin, int n, int r, int a, int b)
    { return (((spin * nfreq + n) * NK + cell(r)) * NAO + a) * NAO + b; };
    std::vector<Z> expected(NSPIN * nfreq * NK * NAO * NAO, Z(0));
    double transpose_gap = 0, adjoint_gap = 0;
    for (std::size_t t = 0; t < times.size(); ++t)
    {
        ComplexMatrix w(NABF_SUPER, NABF_SUPER);
        for (int u = 0; u < NABF_SUPER; ++u)
            for (int v = 0; v < NABF_SUPER; ++v)
                for (int q = 0; q < NK; ++q)
                    for (int m : bosons)
                    {
                        const int delta = v / NABF - u / NABF;
                        w(u, v) +=
                            sample_w(m, q, u % NABF, v % NABF) *
                            std::exp(
                                Z(0, -2 * PI * (q * delta / double(NK) + m * times[t] / BETA))) /
                            (BETA * NK);
                    }
        for (int spin = 0; spin < NSPIN; ++spin)
        {
            ComplexMatrix g(NAO_SUPER, NAO_SUPER);
            for (int p = 0; p < NAO_SUPER; ++p)
                for (int q = 0; q < NAO_SUPER; ++q)
                    for (int k = 0; k < NK; ++k)
                        for (int band = 0; band < NAO; ++band)
                        {
                            const int delta = q / NAO - p / NAO;
                            const double energy = xi(spin, k, band);
                            g(p, q) += orbital(spin, k, band, p % NAO) *
                                       std::conj(orbital(spin, k, band, q % NAO)) *
                                       std::exp(Z(0, -2 * PI * k * delta / NK)) *
                                       std::exp(-energy * times[t]) /
                                       (NK * (1 + std::exp(-BETA * energy)));
                        }
            for (int a = 0; a < NAO; ++a)
                for (int b = 0; b < NAO; ++b)
                {
                    transpose_gap =
                        std::max(transpose_gap, std::abs(g(a, NAO + b) - g(b, NAO + a)));
                    adjoint_gap =
                        std::max(adjoint_gap, std::abs(g(a, NAO + b) - std::conj(g(b, NAO + a))));
                }
            for (int x = 0; x < NAO; ++x)
                for (int y = 0; y < NAO_SUPER; ++y)
                {
                    Z sigma_tau = 0;
                    for (int p = 0; p < NAO_SUPER; ++p)
                        for (int q = 0; q < NAO_SUPER; ++q)
                        {
                            Z interaction = 0;
                            for (int u = 0; u < NABF_SUPER; ++u)
                            {
                                const double left = vertex(u, x, p);
                                if (left == 0) continue;
                                for (int v = 0; v < NABF_SUPER; ++v)
                                    interaction += left * w(u, v) * vertex(v, y, q);
                            }
                            sigma_tau += g(p, q) * interaction;
                        }
                    for (int n = 0; n < nfreq; ++n)
                    {
                        const double omega = (2.0 * fermions[n] + 1.0) * PI / BETA;
                        expected[output_index(spin, n, y / NAO, x, y % NAO)] +=
                            weights[t] * std::exp(Z(0, omega * times[t])) * sigma_tau;
                    }
                }
        }
    }
    require_all(transpose_gap > 1e-5 && adjoint_gap > 1e-5,
                "multiatom fixture does not distinguish transpose/adjoint at fixed R");
    const auto result = gw.build_thermal_spacetime(abf, cs, wc, desc, transform);
    require_all(result.fermionic_grid.get_beta_ha_inv() == BETA && result.chemical_potential_ha == MU &&
                    result.fermionic_grid.get_indices() == fermions &&
                    result.fermionic_grid.get_frequencies_ha() == transform.get_fermionic_frequencies_ha() &&
                    !gw.is_rspace_built() &&
                    legacy_grid.get_n_grids() == 0,
                "multiatom builder changed metadata or legacy GW state");
    std::vector<Z> actual(expected.size(), Z(0));
    bool valid_blocks = true;
    for (const auto &[spin, frequencies] : result.blocks)
        for (const auto &[label, pairs] : frequencies)
            for (const auto &[ij, cells] : pairs)
                for (const auto &[r, block] : cells)
                {
                    const auto frequency = std::find(fermions.begin(), fermions.end(), label);
                    if (spin < 0 || spin >= NSPIN || frequency == fermions.end() ||
                        ij.first >= 2 || ij.second >= 2 || pbc.get_R_index(r) < 0 ||
                        block.nr() != ao.get_atom_nb(ij.first) ||
                        block.nc() != ao.get_atom_nb(ij.second))
                    {
                        valid_blocks = false;
                        continue;
                    }
                    const int n = std::distance(fermions.begin(), frequency);
                    for (int a = 0; a < block.nr(); ++a)
                        for (int b = 0; b < block.nc(); ++b)
                        {
                            const Z value = block(a, b);
                            valid_blocks = valid_blocks && std::isfinite(value.real()) &&
                                           std::isfinite(value.imag());
                            actual[output_index(spin, n, r.x, ao.get_global_index(ij.first, a),
                                                ao.get_global_index(ij.second, b))] += value;
                        }
                }
    require_all(valid_blocks, "multiatom Sigma returned invalid ordered blocks");
    MPI_Allreduce(MPI_IN_PLACE, actual.data(), actual.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
    double error = 0, magnitude = 0, imaginary = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        error = std::max(error, std::abs(actual[i] - expected[i]));
        magnitude = std::max(magnitude, std::abs(expected[i]));
        imaginary = std::max(imaginary, std::abs(expected[i].imag()));
    }
    require_all(magnitude > 1e-6 && imaginary > 1e-6,
                "multiatom reference is vacuous or real-only");
    if (rank == 0)
        std::cout << "thermal multiatom AO=1+2 ABF=2+1 nk=" << NK << " spins=" << NSPIN
                  << " major=" << int(major) << " distributed_C=" << distribute_cs
                  << " ranks=" << size << " max_abs=" << error << " reference_max=" << magnitude
                  << " fixed_R_transpose_gap=" << transpose_gap
                  << " fixed_R_adjoint_gap=" << adjoint_gap << '\n';
    require_all(error < 2e-11 * std::max(1.0, magnitude),
                "multiatom full Sigma differs from independent dense C G W C pair sums");
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
        check_multiatom(MAJOR::ROW, false);
        check_multiatom(MAJOR::ROW, true);
        check_multiatom(MAJOR::COL, true);
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
