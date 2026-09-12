#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;

namespace
{
using Z = std::complex<double>;
const double PI = std::acos(-1.0);
constexpr double BETA = 8.0;
constexpr double MU = 0.15;
const std::vector<int> LABELS{2, -1, 0, -3};

void require(bool value, const char *message)
{
    int failed = !value;
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (failed) throw std::runtime_error(message);
}

double frequency(int n) { return (2.0 * n + 1.0) * PI / BETA; }

template <typename Action>
void rejects(Action action, const char *message)
{
    bool caught = false;
    try
    {
        action();
    }
    catch (const std::exception &)
    {
        caught = true;
    }
    require(caught, message);
}

Z element(int spin, int n, int i, int j, const Vector3_Order<int> &r, int a, int b)
{
    return Z(0.13 * (1 + spin + 2 * n + 3 * i - j + r.x + a + 2 * b),
             0.09 * (2 - spin + n - i + 3 * j - 2 * r.x + 2 * a - b));
}

// Independent finite search, including all degenerate nearest images in the skew cell.
std::vector<Vector3_Order<int>> nearest(const PeriodicBoundaryData &pbc, const Atoms &atoms, int i,
                                        int j, const Vector3_Order<int> &r)
{
    double minimum = std::numeric_limits<double>::infinity();
    std::vector<Vector3_Order<int>> images;
    for (int x = -3; x <= 3; ++x)
        for (int y = -3; y <= 3; ++y)
            for (int z = -3; z <= 3; ++z)
            {
                const Vector3_Order<int> image{r.x + x * pbc.period.x, r.y + y * pbc.period.y,
                                               r.z + z * pbc.period.z};
                const auto delta = atoms.coords_frac.at(j) - atoms.coords_frac.at(i);
                const double frac[3]{delta.x + image.x, delta.y + image.y, delta.z + image.z};
                double distance = 0;
                for (int c = 0; c < 3; ++c)
                {
                    double cart = 0;
                    for (int d = 0; d < 3; ++d) cart += frac[d] * pbc.latvec_array[d][c];
                    distance += cart * cart;
                }
                if (distance < minimum - 1e-11)
                {
                    minimum = distance;
                    images.clear();
                }
                if (std::abs(distance - minimum) < 1e-11) images.push_back(image);
            }
    return images;
}

void check_passed_grid_import()
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2, 0, 0, 0, 2, 0, 0, 0, 2});
    pbc.set_kgrids_kvec(1, 1, 1, {0, 0, 0});
    Atoms atoms;
    atoms.set({0}, {{0, 0, 0}}, pbc.latvec);
    const AtomPairBvKRemap<atom_t> remap(atoms.coords_frac, pbc.Rlist, pbc.period, pbc.latvec, 1);
    AtomicBasis ao(std::vector<std::size_t>{1});
    MeanField mf(1, 1, 1, 1);
    mf.get_efermi() = MU;
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(1, 1);
    wfc(0, 0) = 1.0;
    KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, 1);
    BlacsCtxtHandler global_blacs;
    global_blacs.reset_comm(context.comm_global_h.comm);
    global_blacs.set_grid(context.blacs_h.nprows, context.blacs_h.npcols);
    const auto desc = context.create_array_desc(1, 1);
    SymmetryContext symmetry;
    const std::vector<int> labels{-16001, -4, -1, 0, 3, 16000};
    for (const double beta : {8.0, 3157.75024849497186, 315.7750248494972, 105.25834161649906})
    {
        mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / beta, MU, 1, 1e-12));
        TFGrids response_grid(4);
        response_grid.generate_finite_beta_matsubara(8, beta);
        G0W0 gw(mf, ao, pbc, symmetry, response_grid, context, context, desc, false, false);
        const auto make_result = [&]()
        {
            const auto transform =
                ThermalGWTransform::from_quadrature(beta, {beta / 2}, {beta}, {0}, labels);
            ThermalSigcRspace result{transform.get_fermionic_grid(), {}, MU};
            if (rank == 0)
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    Matz block(1, 1, MAJOR::ROW);
                    block(0, 0) = Z(labels[i], 0.25);
                    result.blocks[0][labels[i]][{0, 0}][{0, 0, 0}] = std::move(block);
                }
            return result;
        };
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            auto result = make_result();
            std::vector<double> positive;
            const auto &frequencies = result.fermionic_grid.get_frequencies_ha();
            for (std::size_t i = 0; i < labels.size(); ++i)
                if (labels[i] >= 0) positive.push_back(frequencies[i]);
            gw.set_thermal_sigc(std::move(result));
            require(gw.get_sigc_frequency_nodes() == positive,
                    "import did not preserve the supplied frequency values exactly");
            gw.build_sigc_matrix_KS_band_blacs(mf.get_eigenvectors(), {{0, 0, 0}}, remap,
                                               global_blacs);
            bool unchanged = true;
            if (rank == 0)
                for (const double omega : gw.get_sigc_frequency_nodes())
                {
                    const int n = int(std::llround((omega * beta / PI - 1) / 2));
                    unchanged = unchanged &&
                                gw.sigc_diag_is_ik_f_KS.at(0).at(0).at(omega).at(0) == Z(n, 0.25);
                }
            require(unchanged, "integer-label import changed or dropped a Sigma block");
        }
    }
}

void check_projection(MAJOR major, int ownership, bool k_distributed = false)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    PeriodicBoundaryData pbc;
    pbc.set_latvec({2, 0, 0, 0.6, 3, 0, 0.2, 0.4, 4});
    pbc.set_kgrids_kvec(2, 1, 1, {0, 0, 0, pbc.G.e11 * PI, pbc.G.e12 * PI, pbc.G.e13 * PI});
    Atoms atoms;
    atoms.set({0, 1}, {{0, 0, 0}, {0.8, 0.1, 0.2}}, pbc.latvec);
    AtomicBasis ao(std::vector<std::size_t>{1, 2});
    MeanField mf(2, 2, 3, 3);
    mf.get_efermi() = MU;
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / BETA, MU, 1, 1e-12));
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            auto &wfc = mf.get_eigenvectors()[spin][0][k];
            wfc.create(3, 3);
            for (int band = 0; band < 3; ++band)
                for (int a = 0; a < 3; ++a)
                    wfc(band, a) = std::polar(1 / std::sqrt(3.0),
                                              2 * PI * band * a / 3 + 0.17 * a * (spin + k));
        }
    KPointBlacsParallelContext context({1, size}, MPI_COMM_WORLD, 2);
    // The projection argument is global; context.blacs_h uses a split communicator.
    BlacsCtxtHandler global_blacs;
    global_blacs.reset_comm(context.comm_global_h.comm);
    global_blacs.set_grid(context.blacs_h.nprows, context.blacs_h.npcols);
    const auto desc = context.create_array_desc(3, 3);
    TFGrids response_grid(4);
    response_grid.generate_finite_beta_matsubara(8, BETA);
    const auto bosons = response_grid.get_freq_nodes();
    const auto times = response_grid.get_time_nodes();
    SymmetryContext symmetry;
    G0W0 gw(mf, ao, pbc, symmetry, response_grid, context, context, desc, k_distributed, false);
    require(global_blacs.comm() == gw.comm_h.comm,
            "projection fixture must use the exact global communicator");
    char output_directory[] = "/tmp/librpa_thermal_sigc_projection_XXXXXX";
    const bool directory_ok = rank != 0 || mkdtemp(output_directory) != nullptr;
    require(directory_ok, "cannot create temporary Sigma output directory");
    MPI_Bcast(output_directory, sizeof(output_directory), MPI_CHAR, 0, MPI_COMM_WORLD);
    gw.output_dir = output_directory;
    require(gw.get_sigc_frequency_nodes() == bosons, "legacy grid accessor changed");
    for (std::size_t i = 0; i < bosons.size(); ++i)
        require(gw.get_sigc_frequency_index(bosons[i]) == int(i), "legacy grid indexing changed");

    const auto make_result = [&]()
    {
        ThermalSigcRspace result{ThermalFrequencyGrid(BETA, LABELS, true), {}};
        result.chemical_potential_ha = MU;
        int block_index = 0;
        for (int spin = 0; spin < 2; ++spin)
            for (int n : LABELS)
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        for (const auto &r : pbc.Rlist)
                        {
                            const int owner = block_index++ % size;
                            if ((ownership == 0 && rank != owner) || (ownership == 2 && rank != 0))
                                continue;
                            const double weight =
                                ownership == 1 ? 2.0 * (rank + 1) / (size * (size + 1.0)) : 1.0;
                            Matz block(ao.get_atom_nb(i), ao.get_atom_nb(j), major);
                            for (int a = 0; a < block.nr(); ++a)
                                for (int b = 0; b < block.nc(); ++b)
                                    block(a, b) = weight * element(spin, n, i, j, r, a, b);
                            result.blocks[spin][n][{i, j}][r] = std::move(block);
                        }
        return result;
    };
    gw.sigc_diag_is_ik_f_KS[9][7][123] = {Z(3, 4)};
    auto caller_result = make_result();
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(std::nextafter(1 / BETA, 1.0),
                                                            std::nextafter(MU, 1.0), 1, 1e-12));
    mf.get_efermi() = std::nextafter(MU, 1.0);
    gw.set_thermal_sigc(caller_result);
    mf.set_fermi_dirac_reference(make_fermi_dirac_reference(1 / BETA, MU, 1, 1e-12));
    mf.get_efermi() = MU;
    for (auto &[spin, frequencies] : caller_result.blocks)
        for (auto &[omega, pairs] : frequencies)
            for (auto &[ij, cells] : pairs)
                for (auto &[r, block] : cells) block.zero_out();
    const std::vector<double> positive{frequency(0), frequency(2)};
    require(gw.is_rspace_built() && gw.sigc_diag_is_ik_f_KS.empty() && gw.sigc_is_ik_f_KS.empty(),
            "thermal import did not replace state and invalidate KS caches");
    require(gw.get_sigc_frequency_nodes() == positive, "positive fermionic grid not retained");
    require(gw.get_sigc_frequency_index(positive[1]) == 1,
            "fermionic index is not its grid offset");
    rejects([&] { gw.get_sigc_frequency_index(bosons[0]); }, "accepted a bosonic grid node");
    rejects([&] { gw.get_sigc_frequency_index(frequency(-1)); }, "retained a negative sample");
    rejects([&] { gw.get_sigc_frequency_index(std::numeric_limits<double>::quiet_NaN()); },
            "accepted nonfinite frequency lookup");

    const auto invalid = [&](const auto &mutate)
    {
        auto result = make_result();
        // Only one rank is malformed: every rank must fail before state mutation.
        if (rank == 0) mutate(result);
        rejects([&] { gw.set_thermal_sigc(std::move(result)); }, "invalid thermal import accepted");
        require(gw.is_rspace_built() && gw.get_sigc_frequency_nodes() == positive,
                "failed import changed the previous state");
    };
    invalid([](auto &r)
            { r.fermionic_grid = ThermalFrequencyGrid(BETA * (1 + 2e-10), LABELS, true); });
    invalid([](auto &r) { r.fermionic_grid = ThermalFrequencyGrid(BETA, LABELS, false); });
    invalid([](auto &r) { r.chemical_potential_ha = MU + 1e-8; });
    invalid([](auto &r) { r.chemical_potential_ha = std::numeric_limits<double>::quiet_NaN(); });
    invalid([](auto &r) { r.fermionic_grid = ThermalFrequencyGrid(BETA, {-1, -3}, true); });
    invalid([](auto &r) { r.blocks[0][1] = {}; });
    invalid([](auto &r) { r.blocks[0][-99] = {}; });
    invalid([](auto &r) { r.blocks[2][0] = {}; });
    invalid([](auto &r) { r.blocks[0][0][{2, 0}][{0, 0, 0}] = Matz(1, 1, MAJOR::ROW); });
    invalid([](auto &r) { r.blocks[0][0][{0, 1}][{0, 0, 0}] = Matz(2, 1, MAJOR::ROW); });
    invalid(
        [](auto &r)
        {
            auto storage = std::make_shared<std::valarray<Z>>(Z(0), 0);
            r.blocks[0][0][{0, 0}][{0, 0, 0}] = Matz(1, 1, storage, MAJOR::ROW);
        });
    invalid([](auto &r) { r.blocks[0][0][{0, 0}][{99, 0, 0}] = Matz(1, 1, MAJOR::ROW); });
    invalid(
        [](auto &r)
        {
            Matz bad(1, 1, MAJOR::ROW);
            bad(0, 0) = Z(0, std::numeric_limits<double>::infinity());
            r.blocks[0][-1][{0, 0}][{0, 0, 0}] = std::move(bad);
        });
    if (size > 1)
        invalid(
            [](auto &r)
            {
                auto labels = r.fermionic_grid.get_indices();
                std::reverse(labels.begin(), labels.end());
                r.fermionic_grid = ThermalFrequencyGrid(BETA, labels, true);
            });
    rejects([&] { gw.read_sigc("thermal_legacy_restart_must_not_be_opened"); },
            "thermal legacy restart was not rejected");
    require(gw.is_rspace_built() && gw.get_sigc_frequency_nodes() == positive,
            "rejected restart cleared thermal state");

    const AtomPairBvKRemap<atom_t> remap(atoms.coords_frac, pbc.Rlist, pbc.period, pbc.latvec, 1);
    const std::vector<Vector3_Order<double>> targets{{0.19, 0.11, -0.07}, {0.37, -0.13, 0.09}};
    bool has_degeneracy = false, has_pair_shift = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (const auto &r : pbc.Rlist)
            {
                auto expected = nearest(pbc, atoms, i, j, r);
                const auto *mapped = remap.find_R_bvk({i, j}, r);
                auto actual = mapped ? *mapped : std::vector<Vector3_Order<int>>{r};
                std::sort(expected.begin(), expected.end());
                std::sort(actual.begin(), actual.end());
                require(actual == expected,
                        "nearest-image remap differs from brute-force geometry");
                has_degeneracy = has_degeneracy || expected.size() > 1;
                has_pair_shift =
                    has_pair_shift || (i != j && expected != std::vector<Vector3_Order<int>>{r});
            }
    require(has_degeneracy && has_pair_shift, "fixture does not exercise nearest-image averaging");
    gw.output_sigc_ks_mat_kf = true;
    gw.build_sigc_matrix_KS_band_blacs(mf.get_eigenvectors(), targets, remap, global_blacs);
    bool complete = true;
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            complete = complete && gw.sigc_is_ik_f_KS.count(spin) &&
                       gw.sigc_is_ik_f_KS.at(spin).count(k) &&
                       gw.sigc_is_ik_f_KS.at(spin).at(k).size() == positive.size();
            if (rank == 0)
                complete = complete && gw.sigc_diag_is_ik_f_KS.count(spin) &&
                           gw.sigc_diag_is_ik_f_KS.at(spin).count(k) &&
                           gw.sigc_diag_is_ik_f_KS.at(spin).at(k).size() == positive.size();
        }
    require(complete, "projection lost spin/k/frequency entries");
    double error = 0;
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
            for (int n : {0, 2})
                for (int band = 0; band < 3; ++band)
                {
                    Z expected = 0;
                    const auto &wfc = mf.get_eigenvectors().at(spin).at(0).at(k);
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            for (const auto &r : pbc.Rlist)
                            {
                                const auto images = nearest(pbc, atoms, i, j, r);
                                Z phase = 0;
                                for (const auto &image : images)
                                    phase += std::exp(Z(0, 2 * PI * (targets[k] * image))) /
                                             double(images.size());
                                for (int a = 0; a < int(ao.get_atom_nb(i)); ++a)
                                    for (int b = 0; b < int(ao.get_atom_nb(j)); ++b)
                                        expected +=
                                            std::conj(wfc(band, ao.get_global_index(i, a))) *
                                            phase * element(spin, n, i, j, r, a, b) *
                                            wfc(band, ao.get_global_index(j, b));
                            }
                    if (rank == 0)
                        error = std::max(
                            error,
                            std::abs(
                                expected -
                                gw.sigc_diag_is_ik_f_KS.at(spin).at(k).at(frequency(n)).at(band)));
                    const auto &ad = gw.desc_sigc_is_ik_f_KS;
                    const auto &block = gw.sigc_is_ik_f_KS.at(spin).at(k).at(frequency(n));
                    for (int a = 0; a < ad.m_loc(); ++a)
                        for (int b = 0; b < ad.n_loc(); ++b)
                            if (ad.indx_l2g_r(a) == band && ad.indx_l2g_c(b) == band)
                                error = std::max(error, std::abs(expected - block(a, b)));
                }
    require(error < 3e-12, "complex KS projection differs from explicit ordered-pair MPI sum");
    bool grid_ok = true;
    const auto grid_path = std::string(output_directory) + "/Sigc_fermionic_grid.dat";
    if (rank == 0)
    {
        std::ifstream grid(grid_path);
        std::string line;
        bool independent = false, beta_ok = false, mu_ok = false;
        int rows = 0;
        while (std::getline(grid, line))
        {
            independent =
                independent || line.find("Independent positive fermionic") != std::string::npos;
            std::istringstream values(line);
            if (!line.empty() && line[0] == '#')
            {
                std::string marker, key;
                double value = 0;
                if (values >> marker >> key >> value)
                {
                    beta_ok = beta_ok || (key == "beta_ha_inv" && value == BETA);
                    mu_ok = mu_ok || (key == "chemical_potential_ha" && value == MU);
                }
                continue;
            }
            int index = -1, label = -1;
            double omega = 0;
            grid_ok = grid_ok && bool(values >> index >> label >> omega) && index == rows &&
                      rows < 2 && label == (rows == 0 ? 0 : 2) && omega == frequency(label);
            ++rows;
        }
        grid_ok = grid_ok && independent && beta_ok && mu_ok && rows == 2;
    }
    require(grid_ok, "thermal Sigma output did not identify its independent FD frequency grid");
    require(response_grid.get_freq_nodes() == bosons && response_grid.get_time_nodes() == times &&
                response_grid.get_n_grids() == 4 && mf.get_efermi() == MU &&
                mf.get_fermi_dirac_reference().kbt_ha == 1 / BETA,
            "thermal import/projection changed the legacy grid or FD reference");
    gw.reset_rspace();
    require(!gw.is_rspace_built() && gw.get_sigc_frequency_nodes() == bosons,
            "reset did not restore legacy frequency selection");
    mf.clear_fermi_dirac_reference();
    rejects([&] { gw.set_thermal_sigc(make_result()); }, "thermal import accepted no FD reference");
    if (rank == 0)
    {
        std::remove(grid_path.c_str());
        rmdir(output_directory);
    }
    if (rank == 0)
        std::cout << "thermal Sigma projection major=" << int(major) << " ownership=" << ownership
                  << " ranks=" << size << " k_distributed=" << k_distributed << " max_abs=" << error
                  << '\n';
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    int status = 0;
    try
    {
        check_passed_grid_import();
        check_projection(MAJOR::ROW, 0);
        check_projection(MAJOR::COL, 1);
        check_projection(MAJOR::ROW, 2);
        check_projection(MAJOR::COL, 1, true);
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
