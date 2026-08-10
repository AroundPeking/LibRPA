# MnF2 shrink ScaLAPACK optimization implementation plan

> **Required sub-skill:** Use `superpowers:executing-plans` to execute this plan task by task, with `superpowers:test-driven-development` for every code change and `superpowers:verification-before-completion` before any success claim.

**Goal:** Make the production-size MnF2 auxiliary-basis shrink transformation complete reliably on df_dcu, prove numerical equivalence, then continue the unchanged full GW calculation to six band files and a checked band figure.

**Architecture:** Add a focused ScaLAPACK layout helper and an MPI benchmark that reproduce the two production GEMMs `U * chi0 * U^H`. Use the benchmark to choose one common square block size for all three shrink descriptors. Add bounded root-rank timings around the existing collection, GEMM, and redistribution stages without changing matrix mathematics or physical inputs.

**Tech stack:** C++11/14, MPI, BLACS/ScaLAPACK, MKL, CMake/CTest, Slurm on df_dcu, ABACUS reader-v1 inputs, LibRPA `g0w0_band`.

---

## Task 1: Add a failing shrink-layout unit test

**Files:**

- Create: `src/test/test_shrink_scalapack.cpp`
- Modify: `src/test/CMakeLists.txt`
- Create later: `src/mpi/shrink_scalapack_layout.h`

- [ ] Add `test_shrink_scalapack` to the four-process test list.
- [ ] Write the test against the not-yet-existing `shrink_scalapack_layout.h` API:

```cpp
const auto layout = make_shrink_scalapack_layout(blacs_h, 1884, 1078, 128);
assert(layout.large_large.mb() == 128);
assert(layout.large_large.nb() == 128);
assert(layout.small_large.mb() == 128);
assert(layout.small_large.nb() == 128);
assert(layout.small_small.mb() == 128);
assert(layout.small_small.nb() == 128);
```

- [ ] Configure/build only this target and record the expected compile failure caused by the missing helper.
- [ ] Commit the red test.

## Task 2: Implement the minimal common-block descriptor helper

**Files:**

- Create: `src/mpi/shrink_scalapack_layout.h`
- Modify: `src/test/test_shrink_scalapack.cpp`

- [ ] Add a small value type holding the three `ArrayDesc` objects.
- [ ] Implement `make_shrink_scalapack_layout(...)` using explicit `ArrayDesc::init(m, n, block, block, 0, 0)` for all shapes.
- [ ] Reject non-positive dimensions or block sizes with an exception/assert consistent with nearby code.
- [ ] Build and run the four-rank descriptor test; confirm the requested block size is preserved rather than replaced by the old one-block-per-process layout.
- [ ] Commit the green helper implementation.

## Task 3: Add the two-GEMM numerical and timing benchmark

**Files:**

- Modify: `src/test/test_shrink_scalapack.cpp`
- Modify: `src/test/CMakeLists.txt` only if a separate benchmark target is clearer after implementation.

- [ ] Extend the test to read optional environment variables:

```text
LIBRPA_TEST_SHRINK_LARGE
LIBRPA_TEST_SHRINK_SMALL
LIBRPA_TEST_SHRINK_BLOCK
LIBRPA_TEST_SHRINK_REPEATS
```

- [ ] First write assertions for a deterministic complex Hermitian `chi0`, deterministic dense rectangular `U`, and the distributed result `U * chi0 * U^H`; run them before the benchmark implementation and record the expected failure.
- [ ] Implement distributed initialization/gather using existing `ArrayDesc`, `init_local_mat`, `pgemr2d`, and `pgemm` helpers.
- [ ] Compare the gathered result with a serial BLAS/LAPACK reference using relative Frobenius error and Hermiticity residual, both required to be at most `1e-10`.
- [ ] Print one parseable root-rank line:

```text
SHRINK_SCALAPACK_BENCH large=... small=... block=... ranks=... grid=... gemm1_s=... gemm2_s=... relerr=... herm=... status=PASS
```

- [ ] Run the small default case on one and four ranks, then run existing `test_matrix_m_mpi`.
- [ ] Commit the benchmark and tests.

## Task 4: Add production shrink progress and timing hooks

**Files:**

- Modify: `src/core/chi0.cpp`
- Modify: `src/mpi/shrink_scalapack_layout.h`
- Modify: `src/test/test_shrink_scalapack.cpp`

- [ ] Add a test for the compact progress-selection rule: normal output selects q indices 0, every tenth q, and the last q; verbose output selects every q.
- [ ] Run the test and record the expected failure before adding the helper.
- [ ] Replace the three `init_square_blk_capped(..., 2048, ...)` calls in `shrink_abfs_chi0` with the common-block layout helper. Keep the initial production constant isolated so the measured winner can be changed in one place.
- [ ] Add `MPI_Wtime`-based root-rank timing for input collection, GEMM 1, GEMM 2, output redistribution, and total q time.
- [ ] Emit compact parseable `SHRINK_CHI0_PROGRESS` records under the tested selection rule. Do not print matrices or per-rank noise.
- [ ] Build and run targeted tests plus `test_matrix_m_mpi` and `test_blacs`.
- [ ] Commit the production integration.

## Task 5: Build and benchmark on df_dcu

**Files:**

- Create remote versioned source/build directory under `/work1/ghj/app/src/`
- Create remote benchmark directory under the preserved MnF2 root
- Update: `docs/develop/mnf2_libcomm_bounded_exchange_validation.md`

- [ ] Sync the committed worktree to a new versioned remote source directory without overwriting the previous executable.
- [ ] Build with the same Intel MPI/MKL, LibRI, LibComm, ELPA, and compiler profile as the preserved MnF2 binary.
- [ ] Record source commit, dependency commits, executable path, and SHA256.
- [ ] Run `--test-only`/smoke checks, then production-shape benchmarks for blocks 64, 128, 256, and the previous coarse block on 1, 4, and 16 ranks. Use one rank per node with OpenMP filling the node where applicable.
- [ ] Keep TCP as the matched formal runtime because it is the provider that passed the earlier MPI initialization gate; run a high-speed-provider control only if MPI initialization succeeds.
- [ ] Select the fastest stable block that passes `relerr <= 1e-10`, `herm <= 1e-10`, and all rank-count completion gates.
- [ ] If the selected block differs from the provisional constant, change only that constant, rerun local targeted tests, rebuild remotely, and record the new hash.
- [ ] Commit the measured selection and validation record.

## Task 6: Run the 65-q shrink and downstream health gates

**Files:**

- Create a new isolated remote validation directory beside the canceled formal job
- Update remote `run-report.md`
- Update: `docs/develop/mnf2_libcomm_bounded_exchange_validation.md`

- [ ] Stage byte-identical physical inputs from the preserved MnF2 calculation and change only the validated executable/runtime configuration.
- [ ] Run a 65-q shrink gate and require all q progress records plus total shrink time below 30 minutes for one time/spin call.
- [ ] Run a reduced-frequency job through chi0 shrink, at least one Wc block, screened-Coulomb solution, and one self-energy construction.
- [ ] Require no MPI/BLACS error, finite matrices, and explicit progress beyond the former shrink boundary.
- [ ] If the block-size variants fail the runtime gate, stop and implement the separately designed q-owned local-BLAS fallback rather than continuing blind block tuning.

## Task 7: Continue the full MnF2 GW calculation

**Files:**

- Create a fresh formal remote run directory under the preserved MnF2 root
- Update remote `run-report.md`

- [ ] Copy/hard-link only immutable producer, band, PyATB, Coulomb, and reader-v1 artifacts; preserve all old failed directories.
- [ ] Verify exact physical inputs: k mesh 6x6x9, 16 frequencies, head/wing route, symmetry, shrink, PP/basis/ABFS chain, and band path.
- [ ] Submit one formal job after scheduler dry-run and record its job ID.
- [ ] Monitor `SHRINK_CHI0_PROGRESS`, Wc, self-energy, memory, and MPI stderr; never duplicate-submit an existing job.
- [ ] Completion gate: scheduler `COMPLETED 0:0`, `libRPA finished successfully`, `LIBRPA_OK`, and six nonempty exactly-310-line KS/EXX/GW spin band files.

## Task 8: Download, validate, and plot the MnF2 bands

**Files:**

- Populate: `/Users/ghj/同步空间/overleaf/GW_pseudopotential_NAO/.codex_tmp/mnf2_libcomm_fix_20260810/final`
- Update: `docs/develop/mnf2_libcomm_bounded_exchange_validation.md`

- [ ] Download the six band files, `band_kpath_info`, `KPT` as `KPT_nscf`, and `band_out`.
- [ ] Determine the occupied-band boundary from continuous nonzero occupations in `band_out`.
- [ ] Plot the ten-segment path `Γ-X-M-Γ-Z-R-A-Z-X-R-M-A` using 31 explicit points per segment.
- [ ] Never reorder across the occupied/conduction boundary; sort occupied states and only the lowest requested conduction subset separately at each k point.
- [ ] Align to the GW VBM, restrict CBM search to the lowest conduction subset, and report gap, VBM/CBM location, spin-mirror residual, NaN count, and discontinuity checks.
- [ ] Generate PNG, PDF, and a machine-readable summary; inspect the PNG visually before reporting completion.
- [ ] Commit the final developer report and provide absolute artifact links.

