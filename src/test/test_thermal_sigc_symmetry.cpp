#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

namespace
{
using namespace librpa_int;
using Z = std::complex<double>;
const double PI = std::acos(-1.0);

void require_all(bool condition, const char *message)
{
    int failed = !condition;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (failed) throw std::runtime_error(message);
}

void check(MAJOR major, bool distributed_c, bool identity_only = false, int nbands = 8)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr int NK = 3, NAO = 8;
    constexpr double BETA = 8, MU = 0.1;
    const auto parity = [](int a) { return a == 0 ? 1.0 : -1.0; };
    PeriodicBoundaryData pbc;
    pbc.set_latvec({4, 0, 0, 0, 5, 0, 0, 0, 6});
    std::vector<double> kvec;
    for (int k = 0; k < NK; ++k)
    {
        kvec.push_back(2 * PI * k / (4 * NK));
        kvec.push_back(0);
        kvec.push_back(0);
    }
    pbc.set_kgrids_kvec(NK, 1, 1, kvec);
    AtomicBasis ao(std::vector<std::size_t>{4, 4}), abf(std::vector<std::size_t>{1, 1});
    ao.set_l_shells({{0, 1}, {0, 1}});
    abf.set_l_shells({{0}, {0}});
    SymmetryContext symmetry;
    symmetry.set_crystal_structure(pbc.latvec, pbc.G, {{0, 0}, {1, 0}},
                                   {{0, {0.25, 0, 0}}, {1, {0.75, 0, 0}}});
    auto inversion = SpaceGroupSymOp::INVERSE;
    inversion.translation = {1, 0, 0};
    symmetry.set_rspace_operations(
        identity_only ? std::vector<SpaceGroupSymOp>{SpaceGroupSymOp::IDENTITY}
                      : std::vector<SpaceGroupSymOp>{SpaceGroupSymOp::IDENTITY, inversion});
    symmetry.build_periodic_mappings(pbc, pbc.Rlist);
    const BasisConvention convention{-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M,
                                     LIBRPA_RSH_COEFF_1_M};
    symmetry.build_rsh_rotations(convention, 1);
    symmetry.set_available();
    require_all(symmetry.count_irreducible_blocks() == (identity_only ? 12 : 6),
                "fixture does not have the expected real-space reduction");

    MeanField mf(1, NK, nbands, NAO);
    mf.get_efermi() = MU;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / BETA, MU, 2, 1e-12));
    // Each Bloch eigenstate has definite inversion parity. On atom 1 the p
    // components reverse sign. Mixing s and p makes the block rotation essential.
    for (int k = 0; k < NK; ++k)
    {
        auto &wfc = mf.get_eigenvectors()[0][0][k];
        wfc.create(nbands, NAO);
        for (int n = 0; n < nbands; ++n)
        {
            const int orbital = n % 4, sector = n / 4;
            const double xi =
                -0.35 + 0.14 * orbital + 0.07 * sector + 0.03 * std::cos(2 * PI * k / NK);
            mf.get_eigenvals()[0](k, n) = MU + xi;
            mf.get_weight()[0](k, n) = 2 / (NK * (1 + std::exp(BETA * xi)));
            for (int a = 0; a < 4; ++a)
            {
                double u = a == orbital ? 1 : 0;
                if (a < 2 && orbital < 2)
                    u = a == orbital ? std::cos(0.31) : (a == 0 ? -1 : 1) * std::sin(0.31);
                wfc(n, a) = u / std::sqrt(2.0);
                wfc(n, 4 + a) = (sector == 0 ? 1 : -1) * parity(a) * u / std::sqrt(2.0);
            }
        }
    }
    Cs_LRI cs;
    cs.use_libri = true;
    int index = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int r = -1; r <= 1; ++r)
            {
                const int owner = distributed_c ? index % size : 0;
                ++index;
                if (rank != owner) continue;
                RI::Tensor<double> c({1, 4, 4});
                const int source_j = i == 0 ? j : 1 - j, source_r = i == 0 ? r : -r;
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b)
                    {
                        const double value =
                            0.02 * std::sin(0.4 + 0.3 * source_j + 0.5 * source_r + 0.7 * a +
                                            0.9 * b) +
                            ((source_j == 0 && source_r == 0 && a == b) ? 0.12 : 0);
                        c(0, a, b) = (i == 0 ? 1 : parity(a) * parity(b)) * value;
                    }
                cs.data_libri[i][{j, {r, 0, 0}}] = std::move(c);
            }
    const std::vector<double> times{0.4, 1.7, 4.1, 7.6}, weights{0.8, 1.8, 3.0, 2.4};
    const std::vector<int> bosons{1, 0, -1}, fermions{2, -1, 0, -3, 4, -5};
    const auto transform =
        ThermalGWTransform::from_quadrature(BETA, times, weights, bosons, fermions);
    KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, NK);
    const auto desc = context.create_array_desc(2, 2);
    std::map<double, std::map<Vector3_Order<double>, Matz>> wc;
    for (std::size_t m = 0; m < bosons.size(); ++m)
        for (int q = 0; q < NK; ++q)
        {
            Matz w(desc.m_loc(), desc.n_loc(), major);
            for (int a = 0; a < w.nr(); ++a)
                for (int b = 0; b < w.nc(); ++b)
                    w(a, b) = -(desc.indx_l2g_r(a) == desc.indx_l2g_c(b) ? 0.3 : 0.08) *
                              (1 + 0.2 * std::cos(2 * PI * q / NK)) / (1 + bosons[m] * bosons[m]);
            wc[transform.get_bosonic_frequencies_ha()[m]][pbc.klist_full[q]] = std::move(w);
        }
    TFGrids legacy_grid;
    if (!identity_only && nbands == NAO)
    {
        const auto saved_stars = symmetry.rspace_sector_stars;
        for (int diagnostic = 0; diagnostic < (size > 1 ? 3 : 2); ++diagnostic)
        {
            if (rank == size - 1 && diagnostic < 2)
            {
                auto &members = symmetry.rspace_sector_stars.begin()->second.begin()->second;
                if (diagnostic == 0)
                    members.pop_back();
                else
                    members.push_back(members.front());
            }
            G0W0 gw(mf, ao, pbc, symmetry, legacy_grid, context, context, desc, false,
                    !(diagnostic == 2 && rank == size - 1));
            gw.libri_threshold_C = gw.libri_threshold_G = gw.libri_threshold_Wc = 0;
            int caught = 0;
            try
            {
                (void)gw.build_thermal_spacetime(abf, cs, wc, desc, transform);
            }
            catch (const std::exception &)
            {
                caught = 1;
            }
            symmetry.rspace_sector_stars = saved_stars;
            MPI_Allreduce(MPI_IN_PLACE, &caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            require_all(caught == size, "rank-local symmetry error did not fail collectively");
        }
    }
    const auto run = [&](bool use_symmetry)
    {
        G0W0 gw(mf, ao, pbc, symmetry, legacy_grid, context, context, desc, false, use_symmetry);
        gw.libri_threshold_C = gw.libri_threshold_G = gw.libri_threshold_Wc = 0;
        const auto result = gw.build_thermal_spacetime(abf, cs, wc, desc, transform);
        require_all(
            result.fermionic_grid.get_frequencies_ha() == transform.get_fermionic_frequencies_ha(),
            "symmetry changed supplied frequencies");
        std::vector<Z> values(fermions.size() * NK * NAO * NAO);
        for (const auto &[spin, frequencies] : result.blocks)
            for (const auto &[label, pairs] : frequencies)
            {
                const int n = std::distance(fermions.begin(),
                                            std::find(fermions.begin(), fermions.end(), label));
                for (const auto &[ij, cells] : pairs)
                    for (const auto &[r, block] : cells)
                        for (int a = 0; a < block.nr(); ++a)
                            for (int b = 0; b < block.nc(); ++b)
                                values[((n * NK + (r.x + NK) % NK) * NAO + 4 * ij.first + a) * NAO +
                                       4 * ij.second + b] += block(a, b);
            }
        MPI_Allreduce(MPI_IN_PLACE, values.data(), values.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                      MPI_COMM_WORLD);
        return values;
    };
    const double start = MPI_Wtime();
    const auto full = run(false);
    const double middle = MPI_Wtime();
    const auto reduced = run(true);
    const double end = MPI_Wtime();
    double error = 0, magnitude = 0, odd_block = 0;
    for (std::size_t n = 0; n < full.size(); ++n)
    {
        error = std::max(error, std::abs(reduced[n] - full[n]));
        magnitude = std::max(magnitude, std::abs(full[n]));
    }
    for (std::size_t n = 0; n < fermions.size(); ++n)
        for (int r = 0; r < NK; ++r)
            odd_block = std::max(odd_block, std::abs(full[(n * NK + r) * NAO * NAO + 1]));
    require_all(magnitude > 1e-5 && odd_block > 1e-6, "vacuous symmetry fixture");
    if (rank == 0)
        std::cout << "THERMAL_SIGC_SYMMETRY ranks=" << size << " major=" << int(major)
                  << " distributed_C=" << distributed_c << " identity=" << identity_only
                  << " nbands=" << nbands << " max_abs=" << error
                  << " relative=" << error / magnitude << " odd_block=" << odd_block
                  << " full_s=" << middle - start << " symmetry_s=" << end - middle << '\n';
    require_all(error < 2e-12 * std::max(1.0, magnitude),
                "symmetry restored Sigma differs from full contraction");
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
        check(MAJOR::ROW, false);
        check(MAJOR::COL, true);
        check(MAJOR::ROW, true, true);
        check(MAJOR::ROW, true, false, 7);
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
