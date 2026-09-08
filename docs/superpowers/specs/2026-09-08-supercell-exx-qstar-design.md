# Supercell EXX Q-Star Reconstruction Design

## Problem

For the symmetry-enabled diamond-C equivalence test, primitive `k444` and
`2x2x2` supercell `k222` calculations have matching KS eigenvalues and XC
matrix elements, but their EXX matrix elements differ by several eV.

The primitive calculation reconstructs every full-BZ Coulomb member from its
IBZ representative with the complete atom, auxiliary-shell, and phase
transformation.  A supercell whose supplied space-group representatives do not
form an affine group has no real-space irreducible sector.  In that case
`FT_Vq` falls back to a legacy path that treats every non-representative member
as the complex conjugate of its representative.  That fallback is valid only
for inversion-related pairs and is incorrect for rotational q-star members.

## Required Behavior

When symmetry metadata and q-stars are available, `FT_Vq` must reconstruct
every full-BZ Coulomb operator before the inverse Fourier transform even when
the real-space irreducible sector is unavailable.  For a q-star member `gq`,
the reconstruction must use the existing k-space operator rotation:

```text
V_IJ(gq) = phase_IJ(g,q) U_I(g) V_I'J'(q) U_J(g)^dagger.
```

The resulting full-sector transform is

```text
V_IJ(R) = (1/Nq) sum_q exp[-i q.R] V_IJ(q).
```

The optimized real-space irreducible-sector route remains unchanged for
ordinary primitive cells.  The inversion-only fallback remains available for
legacy inputs that do not provide usable symmetry metadata.

## Implementation

Add a full-real-space q-star accumulation route beside the existing
irreducible-sector accumulator in `src/core/coulmat.cpp`.  It will:

1. gather each IBZ representative operator across MPI ranks;
2. rotate it to every full-BZ q-star member using
   `rotate_symmetry_kspace_operator_blocks`;
3. accumulate all atom-pair blocks for every BvK `R` with the standard
   `1/Nq` Fourier normalization;
4. return the real part after the complete BZ sum.

Select this route only when q-star reconstruction is valid and the optimized
irreducible-sector route is not.  Do not synthesize missing affine operations
or silently use the inversion-only fallback for rotational q-stars.

## Validation

1. Add a unit test that clears the real-space irreducible sector while keeping
   valid rotational q-stars.  The reduced-IBZ `FT_Vq` result must match an
   explicit full-BZ transform.
2. Build and run tests on `df_dcu`; do not compile LibRPA locally.
3. Run `task = exx` with symmetry enabled for the frozen primitive `k444` and
   supercell `k222` C producers.  Compare physically folded band-edge states;
   require EXX agreement within `1 meV` before running GW.
4. Run GW with the automatically generated 16-point minimax grid.
5. Run a second GW comparison with common explicit `minimax_emin` and
   `minimax_emax`.  This is a diagnostic comparison, not a requirement that
   production supercells reference a primitive-cell calculation.

## Non-Goals

- No change to the RPA head/wing formulas or Gamma averaging.
- No change to minimax grid generation unless the fixed-grid diagnostic later
  demonstrates a material numerical error.
- No full band unfolding in this patch; the initial gate uses folded SCF-grid
  band-edge states.
