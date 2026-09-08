# Supercell EXX Q-Star Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make symmetry-enabled EXX reconstruct rotational q-star members correctly for supercells that do not have a real-space irreducible sector, then quantify primitive/supercell GW agreement with automatic and common minimax windows.

**Architecture:** Keep the existing optimized irreducible-sector transform for primitive cells. Add a full-real-space fallback that gathers an IBZ Coulomb operator, little-group symmetrizes it, rotates it to every full-BZ q-star member with the existing atom/shell/phase machinery, and then performs the inverse Fourier sum over all requested atom pairs and BvK translations.

**Tech Stack:** C++17, LibRPA symmetry helpers, MPI, CTest, Slurm on `df_dcu`, ABACUS producer data, LibRPA EXX/GW.

---

### Task 1: Add the failing matrix-level regression

**Files:**
- Modify: `src/test/test_rpa_headwing.cpp`

- [x] **Step 1: Include the Coulomb transform interface**

Add `#include "../core/coulmat.h"` beside the existing core includes.

- [x] **Step 2: Add an `atpair_R_mat_t` comparison helper**

Implement a helper that checks identical atom-pair and `R` keys, matrix dimensions, and every real element within `1e-12`.

- [x] **Step 3: Add a rotational q-star test without a real-space sector**

Build the existing BN rotational q-star fixture on a `3x3x1` mesh with p shells. Construct an explicit full-BZ Coulomb map by little-group symmetrizing each representative and rotating it to every star member. Clear `ctx.irreducible_sector` and `ctx.rspace_sector_stars`, call `FT_Vq(..., use_symmetry_context=true)` on the reduced map, and compare it with `FT_Vq(..., use_symmetry_context=false)` on the explicit full-BZ map.

- [x] **Step 4: Run the test remotely and verify the old code fails for the intended reason**

Build only the affected test target on `df_dcu`, then run it with one and four MPI ranks. Expected before the implementation: the new `V(R)` comparison fails because the old fallback conjugates rotational q-star members instead of rotating them.

### Task 2: Implement full-sector q-star reconstruction for `FT_Vq`

**Files:**
- Modify: `src/core/coulmat.cpp`

- [x] **Step 1: Separate q-star capability from irreducible-sector capability**

Add a capability check that requires valid symmetry metadata, k-stars, k-star/grid mappings, atom and basis layouts, reduced IBZ storage, and complete or distributed Coulomb coverage, but does not require `irreducible_sector` or `rspace_sector_stars`.

- [x] **Step 2: Add full `V(R)` storage and accumulation**

Allocate every ordered atom pair and every `pbc.Rlist` entry. For each IBZ representative, gather its distributed blocks, build the upper atom-pair closure, enforce its little-group symmetry, rotate to each mapped q-star member, and accumulate

```text
V_IJ(R) += exp[-2 pi i q_frac.R] V_IJ(q) / Nq.
```

Convert the completed complex sum to the real `atpair_R_mat_t` representation used by EXX.

- [x] **Step 3: Select the new path only when needed**

Preserve the existing irreducible-sector path. When q-star data are valid but the real-space sector is unavailable, print one concise diagnostic and use the full-sector path. Retain the inversion-only legacy fallback only when usable q-star metadata are absent.

- [x] **Step 4: Run the new regression remotely**

Rebuild the test target on `df_dcu`; require the new test to pass with one and four MPI ranks.

### Task 3: Run remote software verification

**Files:**
- No source changes expected.

- [x] **Step 1: Run focused symmetry and head/wing tests**

Run `test_rpa_headwing` and symmetry-related CTest entries on `df_dcu`.

- [x] **Step 2: Run the LibRPA regression suite**

Run the available regression suite from the same remote executable and record failures separately from numerical result validation.

- [x] **Step 3: Commit the code fix**

Commit only the test, implementation, design, and plan with Codex as author and AroundPeking as committer.

### Task 4: Gate the fix with C primitive/supercell EXX

**Files:**
- Create: a new sibling calculation directory under `/work1/ghj/gw/c_diamond_k444_equiv_optTZDP_a3p6_v1_shrink_sym_hw_nohw_20260623`

- [x] **Step 1: Freeze all inputs except cell/k mesh**

Reuse the completed ABACUS producers for primitive `k444` and `2x2x2` supercell `k222`. Keep symmetry on, shrink on, the same PP/NAO/ABFS, occupations, band window, and EXX settings.

- [x] **Step 2: Run `task = exx` with the fixed LibRPA**

Use one MPI rank per node, full-node OpenMP/MKL, and a square MPI rank count when more than one node is required.

- [x] **Step 3: Compare folded equivalent states**

Require matching KS and Vxc values and EXX matrix elements within `1 meV`. Stop before GW if this gate fails.

### Task 5: Compare primitive/supercell GW and minimax windows

**Files:**
- Create: automatic-window and fixed-window GW result directories beside the EXX gate.

- [x] **Step 1: Run the automatic 16-point minimax comparison**

Run no-headwing and headwing GW for primitive `k444` and supercell `k222` using each calculation's automatically generated minimax range.

- [x] **Step 2: Run a common explicit minimax-window comparison**

Set identical `minimax_emin` and `minimax_emax` in both primitive and supercell inputs, retaining 16 points and every other setting from Step 1.

- [x] **Step 3: Report separate error sources**

Tabulate KS, Vxc, EXX, correlation self-energy, and QP differences for folded equivalent states. Attribute any change between automatic and fixed windows only to frequency integration; do not use it to mask residual EXX disagreement.

### Task 6: Restore supercell q-stars in the GW screened interaction

**Files:**
- Modify: `src/core/epsilon.cpp`
- Modify: `src/test/test_rpa_headwing.cpp`

- [x] **Step 1: Isolate the post-EXX residual**

After the EXX repair, compare primitive `k444` with supercell `k222`. The EXX elements
agree within `0.01 meV`, while the no-headwing and headwing gaps still differ by
`2.31294 eV` and `1.97231 eV`, respectively. The automatic and common minimax windows
produce identical printed self-energies and QP energies, so frequency-window selection is
not the source.

- [x] **Step 2: Add a failing screened-interaction regression**

Reuse the rotational BN q-star fixture, clear `irreducible_sector` and
`rspace_sector_stars`, and require reduced-IBZ `FT_Wc_q2R` to match an explicit full-BZ
transform. Before the fix, one representative matrix element is
`-0.00109729-0.000777778i` instead of `0.277772-0.204880i`.

- [x] **Step 3: Remove the unrelated real-space-sector gate**

Allow full-q-star `W_c(q) -> W_c(R)` reconstruction whenever the k-star/grid mapping is
complete and covers the BvK mesh. Do not require an affine real-space irreducible sector;
the full-sector transform does not use it.

- [x] **Step 4: Verify software and material results remotely**

On `df_dcu`, require the focused test to pass with one and four MPI ranks, all 49 C++
tests to pass, all 24 LibRPA regressions to pass, and the symmetry-enabled C supercell GW
calculation to enter both complete-q-star `V(R)` and `W_c(R)` paths.

## Validation Results

All energies below are in eV. The material calculation reuses the frozen integer-occupation
ABACUS producers, with symmetry and shrink enabled, four MPI ranks on four nodes, and 30
OpenMP threads per rank.

| Mode | Primitive gap | Supercell gap | Gap difference | Max `V_exx` difference | Max `ReSigma_c` difference | Max QP difference |
|---|---:|---:|---:|---:|---:|---:|
| no headwing | 8.50603 | 8.50607 | +0.00004 | 0.00001 | 0.00028 | 0.00028 |
| headwing | 5.94600 | 5.94595 | -0.00005 | 0.00001 | 0.00013 | 0.00014 |

The maximum differences are evaluated over the 16 calculated supercell edge states after
matching each state to its primitive-cell folded counterpart. The fixed and automatic
16-point minimax controls agree to the `1e-5 eV` print precision for every reported field;
the automatic grid is therefore sufficient for this comparison.

Remote receipts:

- Red test: job `21912340`, expected failure in `FT_Wc_q2R`.
- Focused green test: job `21912364`, one- and four-rank passes.
- Full build and CTest: job `21912392`, 49/49 passed.
- LibRPA regressions: job `21912409`, 24/24 passed; the independent Si symmetry control
  also passed its EXX and self-energy tolerances.
- C supercell GW: jobs `21912420` (no headwing) and `21912421` (headwing), both completed
  successfully with integer occupations and no non-finite values.
