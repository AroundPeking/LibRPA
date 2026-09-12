#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "../core/thermal_gw_prepare.h"
#include "../math/utils_matrix_m_mpi.h"
#include "librpa.hpp"

using namespace librpa_int;

void check_q_restore(bool shrink, bool time_reversal)
{
    librpa::Handler h(MPI_COMM_WORLD);
    const auto ds = api::get_dataset_instance(h);
    const double pi = std::acos(-1.0);
    const double lattice[]{4, 0, 0, 0, 5, 0, 0, 0, 6};
    const double reciprocal[]{2 * pi / 4, 0, 0, 0, 2 * pi / 5, 0, 0, 0, 2 * pi / 6};
    h.set_latvec_and_G(lattice, reciprocal);
    h.set_atoms({0, 0}, {0, 0, 0, 2, 0, 0});
    h.set_ao_basis_wfc({1, 1}, {{0}, {0}});
    h.set_ao_basis_aux({1, 1}, {{0}, {0}});
    h.set_basis_convention(-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M,
                           LIBRPA_RSH_COEFF_1_M);
    const int rotations[]{1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1};
    h.set_symmetry_operations(2, 1, rotations);
    h.set_kgrids_kvec(3, 1, 1, std::vector<double>{0, 0, 0, 2 * pi / 12, 0, 0, 4 * pi / 12, 0, 0});
    initialize_symmetry_context(*ds, true);
    if (time_reversal)
        for (auto &star : ds->symmetry_context.kstars)
            for (auto &member : star.members)
            {
                const bool reversed = !same_fractional_kpoint(member.k_bz, star.k_ibz, 1e-5);
                member = build_symmetry_kspace_operation_member(ds->symmetry_context, 0, reversed,
                                                                member.k_bz, star.k_ibz, 0);
            }
    const auto &pbc = ds->pbc;
    const auto view = build_symmetry_qpoint_view(ds->symmetry_context, pbc, true);
    if (view.representatives.size() != 2)
        throw std::runtime_error("unfold test requires a nontrivial q star");
    BlacsCtxtHandler blacs(MPI_COMM_WORLD);
    blacs.init();
    blacs.set_square_grid();
    ArrayDesc input(blacs), output(blacs);
    input.init_square_blk(shrink ? 1 : 2, shrink ? 1 : 2, 0, 0);
    output.init_square_blk(2, 2, 0, 0);
    const auto transform =
        ThermalGWTransform::from_quadrature(8, {1, 3, 5, 7}, {2, 2, 2, 2}, {1, 0, -1}, {0, -1});
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    // A rank-one analytic W with intracell/intercell couplings to the atom at
    // x=1/2. Inversion carries that atom into its neighboring cell. The
    // W_01 has equal R=0 and R=-1 coefficients, hence 1+exp(-i*2*pi*q)
    // in the inverse of the library's exp(-i*q*R) q-to-R transform.
    // The compressed amplitude has an additional q-dependent complex gauge.
    auto vector = [pi, &pbc](const Vector3_Order<double> &q)
    {
        const double x = Vector3_Order<double>{pbc.latvec * q}.x;
        return std::array<cplxdb, 2>{1.0, 0.5 * (1.0 + std::exp(cplxdb(0, -2 * pi * x)))};
    };
    ThermalWcQMap wc;
    std::map<Vector3_Order<double>, ComplexMatrix> u;
    for (const auto &q : view.representatives)
    {
        auto v = vector(q);
        const cplxdb gauge = std::exp(cplxdb(0, 0.37 + q.x));
        u[q].create(1, 2);
        for (int i = 0; i < 2; ++i) u[q](0, i) = gauge * v[i];
        for (double f : {0.0, frequencies[0]})
        {
            auto w = init_local_mat<cplxdb>(input, MAJOR::ROW);
            for (int i = 0; i < input.m_loc(); ++i)
                for (int j = 0; j < input.n_loc(); ++j)
                    w(i, j) = -0.4 / (1 + f * f) *
                              (shrink ? cplxdb(1)
                                      : std::conj(v[input.indx_l2g_r(i)]) * v[input.indx_l2g_c(j)]);
            wc[f][q] = std::move(w);
        }
    }
    const auto result =
        prepare_thermal_wc(ds->comm_h, wc, pbc, transform, input, output, shrink ? &u : nullptr,
                           &view, &ds->symmetry_context, &ds->basis_aux);
    double error = 0, imag = 0;
    for (double f : frequencies)
        for (const auto &q : pbc.klist_full)
        {
            const auto v = vector(q);
            for (int i = 0; i < output.m_loc(); ++i)
                for (int j = 0; j < output.n_loc(); ++j)
                {
                    const auto expected = -0.4 / (1 + f * f) * std::conj(v[output.indx_l2g_r(i)]) *
                                          v[output.indx_l2g_c(j)];
                    error = std::max(error, std::abs(result.at(f).at(q)(i, j) - expected));
                    imag = std::max(imag, std::abs(expected.imag()));
                }
        }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &imag, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (error > 1e-13 || imag < 0.1)
        throw std::runtime_error("q-star unfolded W differs from complex analytic reference: " +
                                 std::to_string(error));
    if (ds->comm_h.is_root())
        std::cout << "THERMAL_WC_Q_RESTORE shrink=" << shrink << " time_reversal=" << time_reversal
                  << " error=" << error << '\n';
}

void check(bool shrink)
{
    const int small = shrink ? 2 : 3, large = 3;
    MpiCommHandler comm(MPI_COMM_WORLD, true);
    BlacsCtxtHandler blacs(MPI_COMM_WORLD);
    blacs.init();
    blacs.set_square_grid();
    ArrayDesc input(blacs), output(blacs);
    input.init_square_blk(small, small, 0, 0);
    output.init_square_blk(large, large, 0, 0);
    PeriodicBoundaryData pbc;
    pbc.set_latvec({3, 0, 0, 0.4, 4, 0, 0.2, 0.3, 5});
    pbc.set_kgrids_kvec(1, 1, 1, {0, 0, 0});
    const auto transform =
        ThermalGWTransform::from_quadrature(8, {1, 3, 5, 7}, {2, 2, 2, 2}, {2, 0, -2}, {0, -1});
    const auto frequencies = transform.get_bosonic_frequencies_ha();
    const auto q = pbc.klist_full.front();
    ThermalWcQMap wc;
    for (double frequency : {0.0, frequencies[0]})
    {
        auto w = init_local_mat<cplxdb>(input, MAJOR::ROW);
        for (int i = 0; i < input.m_loc(); ++i)
            for (int j = 0; j < input.n_loc(); ++j)
                w(i, j) = cplxdb(0.2 + 0.03 * input.indx_l2g_r(i) + frequency,
                                 -0.04 * (1 + input.indx_l2g_c(j)));
        wc[frequency][q] = std::move(w);
    }
    std::map<Vector3_Order<double>, ComplexMatrix> u;
    u[q].create(small, large);
    for (int i = 0; i < small; ++i)
        for (int j = 0; j < large; ++j) u[q](i, j) = cplxdb(0.1 * (1 + i + j), 0.02 * (i - j));
    const auto result =
        prepare_thermal_wc(comm, wc, pbc, transform, input, output, shrink ? &u : nullptr);
    double error = 0;
    int nonfinite = 0;
    for (double frequency : frequencies)
        for (int i = 0; i < output.m_loc(); ++i)
            for (int j = 0; j < output.n_loc(); ++j)
            {
                int a = output.indx_l2g_r(i), b = output.indx_l2g_c(j);
                if (frequency < 0) std::swap(a, b);
                cplxdb expected = 0;
                if (!shrink)
                    expected = cplxdb(0.2 + 0.03 * a + std::abs(frequency), -0.04 * (1 + b));
                else
                    for (int x = 0; x < small; ++x)
                        for (int y = 0; y < small; ++y)
                            expected +=
                                std::conj(u.at(q)(x, a)) *
                                cplxdb(0.2 + 0.03 * x + std::abs(frequency), -0.04 * (1 + y)) *
                                u.at(q)(y, b);
                if (frequency < 0) expected = std::conj(expected);
                const auto actual = result.at(frequency).at(q)(i, j);
                nonfinite = nonfinite || !std::isfinite(actual.real()) ||
                            !std::isfinite(actual.imag()) || !std::isfinite(expected.real()) ||
                            !std::isfinite(expected.imag());
                error = std::max(error, std::abs(actual - expected));
            }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nonfinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (nonfinite || error > 1e-13)
        throw std::runtime_error("distributed unfold/adjoint differs from explicit sum");
    ArrayDesc invalid_desc;
    int descriptor_caught = 0;
    try
    {
        (void)prepare_thermal_wc(comm, wc, pbc, transform,
                                 comm.myid == comm.nprocs - 1 ? invalid_desc : input, output,
                                 shrink ? &u : nullptr);
    }
    catch (const std::invalid_argument &)
    {
        descriptor_caught = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &descriptor_caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (descriptor_caught != comm.nprocs)
        throw std::runtime_error("invalid Wc descriptor was not rejected collectively");
    auto invalid_storage = wc;
    if (comm.myid == comm.nprocs - 1)
    {
        auto &bad = invalid_storage.at(frequencies[0]).at(q);
        bad = bad.copy();
        bad.sptr()->resize(0);
    }
    int storage_caught = 0;
    try
    {
        (void)prepare_thermal_wc(comm, invalid_storage, pbc, transform, input, output,
                                 shrink ? &u : nullptr);
    }
    catch (const std::invalid_argument &)
    {
        storage_caught = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &storage_caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (storage_caught != comm.nprocs)
        throw std::runtime_error("invalid Wc storage was not rejected collectively");
    if (comm.myid == comm.nprocs - 1) wc.erase(frequencies[0]);
    int caught = 0;
    try
    {
        (void)prepare_thermal_wc(comm, wc, pbc, transform, input, output, shrink ? &u : nullptr);
    }
    catch (const std::invalid_argument &)
    {
        caught = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &caught, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (caught != comm.nprocs) throw std::runtime_error("Wc validation was not collective");
    if (comm.myid == 0)
        std::cout << "THERMAL_WC_PREPARE_PASS shrink=" << shrink << " error=" << error << '\n';
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int status = 0;
    try
    {
        check(false);
        check(true);
        for (bool shrink : {false, true})
            for (bool time_reversal : {false, true}) check_q_restore(shrink, time_reversal);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
