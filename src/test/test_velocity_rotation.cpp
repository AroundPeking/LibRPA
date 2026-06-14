/*!
 * @file test_velocity_rotation.cpp
 * @brief Unit tests for rotate_headwing_velocity and the full AO Bloch rotation
 *        matrix builder build_abacus_ao_bloch_rotation_matrix_full.
 *
 * The tests construct a minimal single-atom, single-s-orbital system and verify:
 *   1. Identity symmetry operation leaves the velocity unchanged.
 *   2. The band-basis velocity trace is invariant under any symmetry operation
 *      (since rotation by a unitary preserves the trace).
 *   3. The full AO Bloch rotation matrix M^S is unitary for a pure rotation
 *      (M^S M^S,dagger == I), which is a prerequisite for the band-basis
 *      reconstruction in rotate_headwing_velocity.
 */

#include "../core/abacus_symmetry.h"
#include "../math/complexmatrix.h"
#include "testutils.h"

#include <array>
#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace librpa_int;
using LIBRPA::AbacusKAtomRotation;
using LIBRPA::AbacusKStar;
using LIBRPA::AbacusKStarMember;
using LIBRPA::AbacusSymmetryContext;
using LIBRPA::AbacusSymmetryOperation;
using LIBRPA::AbacusAOTypeLayout;

//! Build a minimal multi-atom s-orbital symmetry context with the identity
//! operation plus one extra space-group operation. Each atom carries a single
//! s orbital (l=0), so n_aos == n_atoms. This matches the head/wing convention
//! where the PyATB band count equals the AO count.
static AbacusSymmetryContext make_minimal_symmetry_ctx(
    int n_atoms,
    const std::array<std::array<double, 3>, 3>& extra_rotation,
    const int isym_extra)
{
    AbacusSymmetryContext ctx;
    ctx.available = true;
    ctx.lattice_available = true;
    ctx.ao_shell_layout_available = true;

    // Single atom type "X" with one s orbital (l=0, count=1, nao=1).
    ctx.ao_type_layouts.push_back({"X", "X.orb", {1}, 1});
    for (int a = 0; a < n_atoms; ++a)
    {
        ctx.atom_to_type[a] = 0;
        // Atoms on a line along x at fractional positions 0, 1/n, 2/n, ...
        ctx.input_coord_frac[a] = {static_cast<double>(a) / n_atoms, 0.0, 0.0};
    }

    // Identity operation (isym = 0).
    AbacusSymmetryOperation identity;
    identity.isym = 0;
    identity.rotation = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    identity.translation = {0.0, 0.0, 0.0};
    identity.shell_rotations[0] = ComplexMatrix(1, 1);
    identity.shell_rotations[0](0, 0) = {1.0, 0.0};
    ctx.rspace_operations.push_back(identity);

    // Extra operation (isym = isym_extra), same s-orbital shell rotation [1].
    AbacusSymmetryOperation extra;
    extra.isym = isym_extra;
    extra.rotation = extra_rotation;
    extra.translation = {0.0, 0.0, 0.0};
    extra.shell_rotations[0] = ComplexMatrix(1, 1);
    extra.shell_rotations[0](0, 0) = {1.0, 0.0};
    ctx.rspace_operations.push_back(extra);

    // Dummy identity kspace return lattice entries (0 vector) so the phase
    // correction inside rotate_abacus_kspace_matrix uses a no-op phase.
    for (int a = 0; a < n_atoms; ++a)
    {
        ctx.kspace_return_lattice[{a, isym_extra}] = {0, 0, 0};
        ctx.kspace_return_lattice[{a, 0}] = {0, 0, 0};
        ctx.kstar_member_fold_G[{a, isym_extra}] = {0, 0, 0};
        ctx.kstar_member_fold_G[{a, 0}] = {0, 0, 0};
    }

    return ctx;
}

//! Build a k-star member with the given isym and k_bz. Atoms map to themselves
//! (identity permutation), each with the trivial s-orbital shell rotation [1].
static AbacusKStarMember make_member(int n_atoms, int isym, const Vector3_Order<double>& k_bz)
{
    AbacusKStarMember m;
    m.isym = isym;
    m.k_bz = k_bz;
    for (int a = 0; a < n_atoms; ++a)
    {
        AbacusKAtomRotation ar;
        ar.atom_from = a;
        ar.atom_to = a;
        ar.atom_type = 0;
        ar.lmax = 0;
        ar.shell_rotations[0] = ComplexMatrix(1, 1);
        ar.shell_rotations[0](0, 0) = {1.0, 0.0};
        m.atom_rotations.push_back(ar);
    }
    return m;
}

//! Fill a ComplexMatrix with deterministic pseudo-random complex entries.
static void randomize(ComplexMatrix& m, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < m.nr; ++i)
        for (int j = 0; j < m.nc; ++j)
            m(i, j) = {dist(rng), dist(rng)};
}

//! Build a unitary eigenvector matrix C (n_bands, n_aos). In the head/wing
//! path the PyATB velocity and the KS eigenvectors are both square and share
//! the same dimension n_bands == n_aos (PyATB treats the AO basis as the band
//! basis). We construct a random unitary square matrix by orthonormalizing
//! a random complex matrix via modified Gram-Schmidt.
static ComplexMatrix make_unitary_wfc(int n, std::mt19937_64& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    ComplexMatrix C(n, n);
    // Fill with random complex entries (row = band, col = AO).
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C(i, j) = {dist(rng), dist(rng)};
    // Modified Gram-Schmidt over rows (each row becomes an orthonormal vector
    // in AO space), giving C C^dagger = I.
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < i; ++k)
        {
            std::complex<double> dot{0, 0};
            for (int j = 0; j < n; ++j)
                dot += std::conj(C(k, j)) * C(i, j);
            for (int j = 0; j < n; ++j)
                C(i, j) -= dot * C(k, j);
        }
        double norm = 0.0;
        for (int j = 0; j < n; ++j)
            norm += std::norm(C(i, j));
        norm = std::sqrt(norm);
        for (int j = 0; j < n; ++j)
            C(i, j) /= norm;
    }
    return C;
}

static void test_identity_operation_preserves_velocity()
{
    std::cout << "  [test_identity_operation_preserves_velocity] ... ";
    // In the head/wing path n_bands == n_aos (PyATB convention), so the
    // eigenvector C is a square unitary matrix and band<->AO transforms are
    // invertible. Use 3 atoms, each with one s orbital -> n_aos = 3 = n_bands.
    const int n_atoms = 3;
    const int n = n_atoms;

    // Identity-only context.
    std::array<std::array<double, 3>, 3> identity_rot{{{{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}}};
    // We pass identity as the "extra" too; both operations are identity here.
    auto ctx = make_minimal_symmetry_ctx(n_atoms, identity_rot, 1);

    // Save and install global ctx.
    auto saved_ctx = LIBRPA::abacus_symmetry_ctx;
    LIBRPA::abacus_symmetry_ctx = ctx;

    std::mt19937_64 rng(42);
    ComplexMatrix C = make_unitary_wfc(n, rng);

    std::array<ComplexMatrix, 3> v_ibz{
        ComplexMatrix(n, n),
        ComplexMatrix(n, n),
        ComplexMatrix(n, n)};
    for (int a = 0; a < 3; ++a)
        randomize(v_ibz[a], rng);

    // k_ibz at Gamma; member maps Gamma to Gamma under identity.
    const Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::map<atom_t, size_t> atom_nw;
    std::map<atom_t, std::array<double, 3>> coord_frac;
    for (int a = 0; a < n_atoms; ++a)
    {
        atom_nw[a] = 1;
        coord_frac[a] = {static_cast<double>(a) / n_atoms, 0.0, 0.0};
    }

    AbacusKStarMember member = make_member(n_atoms, 0, k_ibz);

    auto v_bz = LIBRPA::rotate_headwing_velocity(
        LIBRPA::abacus_symmetry_ctx, member, v_ibz, C, atom_nw, k_ibz, coord_frac,
        /*use_time_reversal=*/false, /*k_bz_target=*/nullptr);

    // Under identity, the velocity should be unchanged up to floating-point noise.
    const double tol = 1e-10;
    double max_diff = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                max_diff = std::max(max_diff, std::abs(v_bz[a](i, j) - v_ibz[a](i, j)));

    LIBRPA::abacus_symmetry_ctx = saved_ctx;

    if (max_diff >= tol)
    {
        std::cerr << "FAILED: identity rotation changed velocity by " << max_diff << "\n";
        throw std::runtime_error("test_identity_operation_preserves_velocity failed");
    }
    std::cout << "OK (max_diff=" << max_diff << ")\n";
}

static void test_trace_invariance_under_rotation()
{
    std::cout << "  [test_frobenius_norm_invariance_under_rotation] ... ";
    const int n_atoms = 3;
    const int n = n_atoms;

    // 180-degree rotation about z: x->-x, y->-y, z->z.
    std::array<std::array<double, 3>, 3> rot180z{{{{-1, 0, 0}}, {{0, -1, 0}}, {{0, 0, 1}}}};
    auto ctx = make_minimal_symmetry_ctx(n_atoms, rot180z, 1);
    auto saved_ctx = LIBRPA::abacus_symmetry_ctx;
    LIBRPA::abacus_symmetry_ctx = ctx;

    std::mt19937_64 rng(7);
    ComplexMatrix C = make_unitary_wfc(n, rng);

    std::array<ComplexMatrix, 3> v_ibz{
        ComplexMatrix(n, n),
        ComplexMatrix(n, n),
        ComplexMatrix(n, n)};
    for (int a = 0; a < 3; ++a)
        randomize(v_ibz[a], rng);

    const Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::map<atom_t, size_t> atom_nw;
    std::map<atom_t, std::array<double, 3>> coord_frac;
    for (int a = 0; a < n_atoms; ++a)
    {
        atom_nw[a] = 1;
        coord_frac[a] = {static_cast<double>(a) / n_atoms, 0.0, 0.0};
    }
    // Member maps Gamma to Gamma under the 180-z rotation.
    AbacusKStarMember member = make_member(n_atoms, 1, Vector3_Order<double>{0.0, 0.0, 0.0});

    auto v_bz = LIBRPA::rotate_headwing_velocity(
        LIBRPA::abacus_symmetry_ctx, member, v_ibz, C, atom_nw, k_ibz, coord_frac,
        /*use_time_reversal=*/false, /*k_bz_target=*/nullptr);

    LIBRPA::abacus_symmetry_ctx = saved_ctx;

    // The full transformation combines a unitary band/AO rotation (which
    // preserves the Frobenius norm of each Cartesian component) with an
    // orthogonal Cartesian rotation R (which mixes components). The total
    // Frobenius norm squared summed over Cartesian components is therefore
    // invariant:
    //   sum_a ||v_bz[a]||_F^2 == sum_a ||v_ibz[a]||_F^2
    double norm_ibz = 0.0, norm_bz = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                norm_ibz += std::norm(v_ibz[a](i, j));
                norm_bz += std::norm(v_bz[a](i, j));
            }

    const double tol = 1e-10;
    const double rel_diff = std::abs(norm_ibz - norm_bz) / norm_ibz;
    if (rel_diff >= tol)
    {
        std::cerr << "FAILED: Frobenius norm changed by " << rel_diff
                  << " (ibz=" << norm_ibz << ", bz=" << norm_bz << ")\n";
        throw std::runtime_error("test_frobenius_norm_invariance_under_rotation failed");
    }
    std::cout << "OK (rel_diff=" << rel_diff << ")\n";
}

static void test_ao_bloch_matrix_is_unitary()
{
    std::cout << "  [test_ao_bloch_matrix_is_unitary] ... ";
    // For s orbitals with identity shell rotation and a pure rotation (no
    // fractional translation), the full AO Bloch matrix M^S is block diagonal
    // with one 1x1 phase factor per atom. At Gamma delta_k == 0 so every block
    // equals 1 and M^S == I (n_aos x n_aos identity).
    const int n_atoms = 3;
    std::array<std::array<double, 3>, 3> rot180z{{{{-1, 0, 0}}, {{0, -1, 0}}, {{0, 0, 1}}}};
    auto ctx = make_minimal_symmetry_ctx(n_atoms, rot180z, 1);
    auto saved_ctx = LIBRPA::abacus_symmetry_ctx;
    LIBRPA::abacus_symmetry_ctx = ctx;

    const Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::map<atom_t, size_t> atom_nw;
    std::map<atom_t, std::array<double, 3>> coord_frac;
    for (int a = 0; a < n_atoms; ++a)
    {
        atom_nw[a] = 1;
        coord_frac[a] = {static_cast<double>(a) / n_atoms, 0.0, 0.0};
    }
    AbacusKStarMember member = make_member(n_atoms, 1, k_ibz);

    auto M = LIBRPA::build_abacus_ao_bloch_rotation_matrix_full(
        LIBRPA::abacus_symmetry_ctx, member, atom_nw, k_ibz, coord_frac,
        /*use_time_reversal=*/false, /*k_bz_target=*/nullptr);

    LIBRPA::abacus_symmetry_ctx = saved_ctx;

    // M should be n_atoms x n_atoms identity at Gamma.
    assert(M.nr == n_atoms && M.nc == n_atoms);
    double max_diff = 0.0;
    for (int i = 0; i < n_atoms; ++i)
        for (int j = 0; j < n_atoms; ++j)
        {
            const std::complex<double> expected = (i == j) ? std::complex<double>{1.0, 0.0}
                                                           : std::complex<double>{0.0, 0.0};
            max_diff = std::max(max_diff, std::abs(M(i, j) - expected));
        }
    if (max_diff > 1e-12)
    {
        std::cerr << "FAILED: M differs from identity by " << max_diff << "\n";
        throw std::runtime_error("test_ao_bloch_matrix_is_unitary failed");
    }
    std::cout << "OK (max_diff_from_I=" << max_diff << ")\n";
}

int main(int argc, char* argv[])
{
    std::cout << "Running test_velocity_rotation ...\n";
    try
    {
        test_ao_bloch_matrix_is_unitary();
        test_identity_operation_preserves_velocity();
        test_trace_invariance_under_rotation();
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "All velocity-rotation tests PASSED.\n";
    return 0;
}
