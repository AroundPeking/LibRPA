#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <chrono>
#include <complex>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../core/qsgw/checkpoint.h"
#include "../core/qsgw/effective_hamiltonian.h"

using librpa_int::cplxdb;
using librpa_int::Matz;
using librpa_int::qsgw::assemble_effective_hamiltonian;
using librpa_int::qsgw::build_reference_hamiltonian;
using librpa_int::qsgw::SpinKMatrixMap;

namespace
{

template <typename Function>
std::string exception_message(Function&& function)
{
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    assert(false);
    return {};
}

Matz hermitian(const double diagonal0, const double diagonal1, const cplxdb off_diagonal,
               const librpa_int::MAJOR major = librpa_int::MAJOR::ROW)
{
    Matz result(2, 2, major);
    result(0, 0) = diagonal0;
    result(1, 1) = diagonal1;
    result(0, 1) = off_diagonal;
    result(1, 0) = std::conj(off_diagonal);
    return result;
}

SpinKMatrixMap one_block(const Matz& value)
{
    SpinKMatrixMap result;
    result[0][0] = value.copy();
    return result;
}

void assert_close(const cplxdb actual, const cplxdb expected, const double tolerance = 1.0e-13)
{
    assert(std::abs(actual - expected) < tolerance);
}

void test_qsgw_formula()
{
    const auto hks = one_block(hermitian(1.0, 2.0, {0.1, -0.2}, librpa_int::MAJOR::ROW));
    const auto vxc = one_block(hermitian(0.3, 0.4, {-0.2, 0.1}, librpa_int::MAJOR::COL));
    const auto exchange = one_block(hermitian(-0.5, -0.6, {0.05, 0.3}, librpa_int::MAJOR::COL));
    const auto correlation = one_block(hermitian(0.2, 0.1, {0.15, -0.1}, librpa_int::MAJOR::ROW));

    const Matz hks_before = hks.at(0).at(0).copy();
    const Matz vxc_before = vxc.at(0).at(0).copy();
    const Matz exchange_before = exchange.at(0).at(0).copy();
    const Matz correlation_before = correlation.at(0).at(0).copy();
    Matz expected(2, 2);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            expected(row, column) = hks.at(0).at(0)(row, column) - vxc.at(0).at(0)(row, column) +
                                    exchange.at(0).at(0)(row, column) +
                                    correlation.at(0).at(0)(row, column);
        }
    }

    const auto result = assemble_effective_hamiltonian(hks, vxc, exchange, correlation);
    const auto& matrix = result.at(0).at(0);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            assert_close(matrix(row, column), expected(row, column));
            assert_close(hks.at(0).at(0)(row, column), hks_before(row, column));
            assert_close(vxc.at(0).at(0)(row, column), vxc_before(row, column));
            assert_close(exchange.at(0).at(0)(row, column), exchange_before(row, column));
            assert_close(correlation.at(0).at(0)(row, column), correlation_before(row, column));
        }
    }
}

void test_nonhermitian_components_use_the_legacy_upper_triangle()
{
    const auto valid = one_block(hermitian(1.0, 2.0, {0.0, 0.0}));
    auto exchange = one_block(hermitian(-0.5, -0.6, {0.05, 0.3}));
    exchange.at(0).at(0)(0, 0) += cplxdb(0.0, 4.0e-7);
    exchange.at(0).at(0)(1, 0) += cplxdb(0.0, 2.5e-7);

    const auto result = assemble_effective_hamiltonian(valid, valid, exchange, valid);
    const auto& matrix = result.at(0).at(0);
    assert_close(matrix(0, 0), cplxdb(0.5, 0.0));
    assert_close(matrix(0, 1), cplxdb(0.05, 0.3));
    assert_close(matrix(1, 0), std::conj(matrix(0, 1)));
}

void test_reference_hamiltonian_is_the_initial_eigenvalue_diagonal()
{
    librpa_int::MeanField reference(1, 2, 3, 1, 1);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        for (int band = 0; band < 3; ++band)
        {
            reference.get_eigenvals()[0](kpoint, band) = 10.0 * kpoint + band + 0.25;
        }
    }
    const auto result = build_reference_hamiltonian(reference);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                const cplxdb expected =
                    row == column ? cplxdb(10.0 * kpoint + row + 0.25, 0.0) : cplxdb(0.0, 0.0);
                assert_close(result.at(0).at(kpoint)(row, column), expected);
            }
        }
    }
}

void test_nonzero_hartree_feedback_and_qp_subtraction()
{
    const auto h0 = one_block(hermitian(-0.8, 0.6, {0.0, 0.0}));
    const auto vxc = one_block(hermitian(-0.2, -0.1, {0.03, 0.02}));
    const auto sx = one_block(hermitian(-0.4, -0.3, {0.08, -0.06}));
    const auto sc = one_block(hermitian(0.05, 0.12, {-0.02, 0.04}));
    const auto dh = one_block(hermitian(0.17, -0.09, {0.07, 0.13}, librpa_int::MAJOR::COL));
    const auto initial_dh = dh.at(0).at(0).copy();
    const auto updated = assemble_effective_hamiltonian(h0, vxc, sx, sc, &dh);
    const auto subtract = librpa_int::qsgw::build_qp_static_subtraction(updated, h0, vxc, &dh);
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j)
        {
            assert_close(updated.at(0).at(0)(i, j),
                         h0.at(0).at(0)(i, j) - vxc.at(0).at(0)(i, j) + sx.at(0).at(0)(i, j) +
                             sc.at(0).at(0)(i, j) + dh.at(0).at(0)(i, j));
            // A QP correction must retain the explicit Hartree already in H_live.
            assert_close(subtract.at(0).at(0)(i, j), sx.at(0).at(0)(i, j) + sc.at(0).at(0)(i, j));
            assert_close(dh.at(0).at(0)(i, j), initial_dh(i, j));
        }
    // Mixing/cuts can change the entering H independently. Subtract the actual
    // entering static part, rather than reconstructing it from the latest GW build.
    auto entering = one_block(hermitian(-0.51, 0.72, {-0.02, 0.11}));
    const auto actual = librpa_int::qsgw::build_qp_static_subtraction(entering, h0, vxc, &dh);
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j)
            assert_close(actual.at(0).at(0)(i, j),
                         entering.at(0).at(0)(i, j) - h0.at(0).at(0)(i, j) + vxc.at(0).at(0)(i, j) -
                             dh.at(0).at(0)(i, j));
}

void test_hartree_restart_preserves_dft_reference_and_live_operator()
{
#ifdef LIBRPA_USE_LIBRI
    using namespace librpa_int;
    using namespace librpa_int::qsgw;
    MeanField reference(1, 1, 2, 2, 1);
    reference.get_efermi() = -0.15;
    reference.get_eigenvals()[0](0, 0) = -0.8;
    reference.get_eigenvals()[0](0, 1) = 0.6;
    reference.get_weight()[0](0, 0) = 2.0;
    reference.get_eigenvectors()[0][0][0] = ComplexMatrix(2, 2);
    reference.get_eigenvectors()[0][0][0](0, 0) = 1.0;
    reference.get_eigenvectors()[0][0][0](1, 1) = 1.0;
    MeanField live = reference;
    live.get_eigenvectors()[0][0][0](0, 0) = 0.8;
    live.get_eigenvectors()[0][0][0](0, 1) = cplxdb(0.0, 0.6);
    live.get_eigenvectors()[0][0][0](1, 0) = cplxdb(0.0, 0.6);
    live.get_eigenvectors()[0][0][0](1, 1) = 0.8;
    live.get_weight()[0](0, 0) = 1.2;
    live.get_weight()[0](0, 1) = 0.8;
    live.get_eigenvals()[0](0, 0) = -0.51;
    live.get_eigenvals()[0](0, 1) = 0.72;
    const auto vxc = one_block(hermitian(-0.2, -0.1, {0.03, 0.02}));
    const auto entering = one_block(hermitian(-0.51, 0.72, {-0.02, 0.11}));
    SpinKMatrixMap empty;
    const std::vector<Vector3_Order<double>> points{{0.0, 0.0, 0.0}}, no_points;
    const std::vector<double> weights{1.0};
    Matz metric(1, 1);
    metric(0, 0) = 1.7;
    Cs_LRI raw;
    raw.use_libri = true;
    RI::Tensor<double> c({1, 2, 2});
    c(0, 0, 0) = 0.3;
    c(0, 0, 1) = -0.11;
    c(0, 1, 1) = 0.21;
    raw.data_libri[0][{0, {3, 0, 0}}] = c;
    CheckpointContext context{reference, nullptr, points, no_points, weights, vxc, empty,
                              0,         2,       0.0,    1.0,       8,       0};
    context.hartree_coulomb_q0 = &metric;
    context.hartree_cs_raw = &raw;
    IterationSummary summary;
    summary.iteration = 2;
    summary.electron_count = 2.0;
    const auto path =
        std::filesystem::temp_directory_path() /
        ("librpa-qsgw-hartree-control-" +
         std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
         ".bin");
    write_qsgw_checkpoint(path.string(), context, summary, live, nullptr, entering, empty);
    MeanField restored = reference;
    SpinKMatrixMap restored_h, unused;
    const auto restored_summary =
        read_qsgw_checkpoint(path.string(), context, restored, nullptr, restored_h, unused);
    std::filesystem::remove(path);
    assert(restored_summary.iteration == 2);
    assert_close(reference.get_weight()[0](0, 0), 2.0);
    assert_close(reference.get_eigenvectors().at(0).at(0).at(0)(0, 1), 0.0);
    for (int i = 0; i != 2; ++i)
    {
        assert_close(restored.get_weight()[0](0, i), live.get_weight()[0](0, i));
        for (int j = 0; j != 2; ++j)
        {
            assert_close(restored.get_eigenvectors().at(0).at(0).at(0)(i, j),
                         live.get_eigenvectors().at(0).at(0).at(0)(i, j));
            assert_close(restored_h.at(0).at(0)(i, j), entering.at(0).at(0)(i, j));
        }
    }
#endif
}

}  // namespace

int main()
{
    test_qsgw_formula();
    test_nonhermitian_components_use_the_legacy_upper_triangle();
    test_reference_hamiltonian_is_the_initial_eigenvalue_diagonal();
    test_nonzero_hartree_feedback_and_qp_subtraction();
    test_hartree_restart_preserves_dft_reference_and_live_operator();
    std::cout << "test_qsgw_effective_hamiltonian: all tests passed\n";
    return 0;
}
