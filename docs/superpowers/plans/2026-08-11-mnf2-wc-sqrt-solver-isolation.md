# MnF2 Wc square-root solver isolation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate the MnF2 Gamma-point Wc matrix-square-root stop, select a numerically valid distributed solver/network lane, and resume the unchanged GW workflow through final band files.

**Architecture:** Add one focused MPI test executable that calls the same `LaConnector::power_hemat_la_real` path as Wc on a deterministic dense positive-definite matrix. Build the existing feature branch with bundled CPU ELPA, compare ScaLAPACK/TCP, ScaLAPACK/native OFI, and ELPA/native OFI under identical dimensions and rank topology, then promote only a passing lane to reduced- and full-frequency GW gates.

**Tech Stack:** C++17, MPI, BLACS/ScaLAPACK, Intel MKL, bundled ELPA 2026.02.001, CMake/CTest, Slurm on df_dcu, LibRPA/LibRI/LibComm, Markdown validation reports.

---

## File map

- Create `src/test/test_wc_sqrt_solver.cpp`: deterministic Wc-like square-root test, timing, residuals, and parseable result marker.
- Modify `src/test/CMakeLists.txt`: register `test_wc_sqrt_solver` as a four-rank MPI target.
- Create `docs/develop/mnf2_wc_sqrt_solver_validation.md`: append-only local and remote evidence log.
- Modify `docs/superpowers/plans/2026-08-11-mnf2-wc-sqrt-solver-isolation.md`: mark completed checkboxes during execution.
- Create remote versioned source/build and run directories under `/work1/ghj/app/src/` and the preserved MnF2 calculation root; never overwrite old runs.

## Task 1: Establish the append-only validation record

**Files:**

- Create: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [x] **Step 1: Record the preserved failure boundary**

Create the report with the exact job `21568982`, source commit
`e097c60b0d5ebf9ec3d064b7c3808d2573dc52c4`, executable SHA256
`d39baf4c1ff9264f7494fec715ec3f151cf607b92b53763b9b6e2eba49b97627`,
16 completed chi0/shrink passes, and the stop at
`epsilon_prepare_coulwc_sqrt_4 -> power_hemat_blacs_real -> pdsyev` for
`n=1078`, block 128, grid `4x4`, forced TCP.

- [x] **Step 2: Add the per-attempt template**

Use this table for every attempt:

```markdown
| Field | Value |
| --- | --- |
| Purpose / changed variable | |
| Local and remote directory | |
| Source commit / dirty state | |
| Executable SHA256 | |
| CMake ELPA/LibRI/LibComm settings | |
| Nodes / MPI ranks / OMP / MKL threads | |
| MPI provider variables | |
| Slurm job and scheduler result | |
| Last completed phase | |
| Wall time / residual / Hermiticity | |
| Evidence-bounded conclusion | |
| Next action | |
```

- [x] **Step 3: Verify and commit the record skeleton**

Run:

```bash
grep -nE '21568982|e097c60b|pdsyev|1078|append-only' \
  docs/develop/mnf2_wc_sqrt_solver_validation.md
git diff --check
```

Expected: all five provenance terms are present and `git diff --check` exits 0.

Commit only the new report:

```bash
git add docs/develop/mnf2_wc_sqrt_solver_validation.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'docs: start MnF2 Wc solver validation log'
```

## Task 2: Add the failing MPI square-root test

**Files:**

- Create: `src/test/test_wc_sqrt_solver.cpp`
- Modify: `src/test/CMakeLists.txt`

- [x] **Step 1: Register the test target**

Add `test_wc_sqrt_solver` to the existing four-process `foreach(target ...)`
list in `src/test/CMakeLists.txt`.

- [x] **Step 2: Write the red test against missing benchmark functions**

Create `src/test/test_wc_sqrt_solver.cpp` with MPI initialization identical to
`test_shrink_scalapack.cpp`, a square BLACS grid, environment parsing for
`LIBRPA_TEST_WC_DIM`, `LIBRPA_TEST_WC_BLOCK`, and
`LIBRPA_TEST_WC_USE_ELPA`, and calls to the not-yet-defined functions:

```cpp
const auto result = run_wc_sqrt_benchmark(
    blacs_h, env_positive_int("LIBRPA_TEST_WC_DIM", 32),
    env_positive_int("LIBRPA_TEST_WC_BLOCK", 8),
    env_flag("LIBRPA_TEST_WC_USE_ELPA", false));
assert(result.finite);
assert(result.filtered == 0);
assert(result.relative_residual <= 1.0e-9);
assert(result.hermiticity_residual <= 1.0e-12);
```

- [x] **Step 3: Run the red build on df_dcu**

Transfer the uncommitted red-test source snapshot to a temporary versioned
directory under `/work1/ghj/app/src/` on df_dcu. In the same remote shell,
source `/public/home/ghj/app/src/env_60_245_intel2021.sh`, then run:

```bash
cmake -S . -B build_mnf2_wc_red \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpiicpc \
  -DCMAKE_Fortran_COMPILER=mpiifort \
  -DLIBRPA_USE_LIBRI=OFF \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_DRIVER=OFF
cmake --build build_mnf2_wc_red --target test_wc_sqrt_solver -j16
```

Expected: compilation fails because `run_wc_sqrt_benchmark`,
`env_positive_int`, `env_flag`, and `WcSqrtBenchmarkResult` are not defined.
Record the first compiler error in the validation report.

- [x] **Step 4: Commit the red test**

```bash
git add src/test/CMakeLists.txt src/test/test_wc_sqrt_solver.cpp \
  docs/develop/mnf2_wc_sqrt_solver_validation.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test: define MnF2 Wc square-root gate'
```

## Task 3: Implement the minimal deterministic benchmark

**Files:**

- Modify: `src/test/test_wc_sqrt_solver.cpp`
- Modify: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [x] **Step 1: Implement strict environment parsing**

Add `env_positive_int` and `env_flag`. Accept flag values `0`, `1`, `false`,
and `true`; throw `std::invalid_argument` for any other value. Reject a block
larger than the matrix dimension.

- [x] **Step 2: Construct a deterministic dense SPD matrix**

On BLACS source rank 0 fill the matrix with:

```cpp
double test_element(const int i, const int j, const int n)
{
    if (i == j)
        return 2.0 + static_cast<double>(i + 1) / static_cast<double>(n);
    return 1.0e-2 / (1.0 + std::abs(i - j));
}
```

Store it as `matrix_m<std::complex<double>>`, distribute it with
`ScalapackConnector::pgemr2d_f`, and keep the source copy for residual checks.
The diagonal dominance keeps the matrix positive definite while the nonzero
off-diagonal terms exercise eigenvector communication.

- [x] **Step 3: Call the production Wc square-root dispatcher**

Initialize distributed A/Z descriptors with the requested block. When ELPA is
requested and `LIBRPA_USE_ELPA` is defined, call
`desc.set_elpa_handle(false)` before:

```cpp
LaConnector::power_hemat_la_real<double>(
    a_local, desc, z_local, desc, filtered, eigenvalues.data(), 0.5,
    -1.0e5, false, use_elpa);
```

When ELPA is requested in a build without `LIBRPA_USE_ELPA`, throw a clear
runtime error rather than silently falling back to ScaLAPACK.

- [x] **Step 4: Compute independent numerical gates**

Keep a distributed copy of the original matrix. Compute `S*S` with
`ScalapackConnector::pgemm_f` into a separate distributed matrix; do not use a
root-rank cubic loop. Each rank accumulates the squared norm of its local
residual block, and `MPI_Allreduce` produces the global Frobenius residual.
Gather `S` only for the quadratic Hermiticity check, then calculate:

```text
relative_residual = ||S*S-A||F / ||A||F
hermiticity_residual = ||S-S^H||F / ||S||F
```

Broadcast the result structure fields. Require finite values, zero filtered
eigenvalues, relative residual at most `1e-9`, and Hermiticity residual at most
`1e-12`. Report the square-root and residual-multiplication times separately so
the validation work is not mistaken for eigensolver time.

- [x] **Step 5: Emit one parseable root-rank line**

Print exactly one summary line beginning:

```text
WC_SQRT_BENCH solver=scalapack|elpa n=... block=... ranks=... grid=... sqrt_s=... residual_gemm_s=... filtered=... relres=... herm=... finite=1 status=PASS
```

- [x] **Step 6: Run the green df_dcu test**

```bash
cmake --build build_mnf2_wc_green --target test_wc_sqrt_solver -j16
ctest --test-dir build_mnf2_wc_green -R '^test_wc_sqrt_solver$' \
  --output-on-failure
```

Run this command on df_dcu, not on the local workstation. Expected: four-rank
ScaLAPACK default case prints `WC_SQRT_BENCH ... status=PASS` and CTest reports
`100% tests passed`.

- [x] **Step 7: Run nearby regression tests**

```bash
ctest --test-dir build_mnf2_wc_green \
  -R '^(test_matrix_m_mpi|test_shrink_scalapack|test_wc_sqrt_solver)$' \
  --output-on-failure
```

Expected: all three tests pass. Record commands and results in the report.

- [x] **Step 8: Commit the green implementation**

```bash
git add src/test/test_wc_sqrt_solver.cpp \
  docs/develop/mnf2_wc_sqrt_solver_validation.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m 'test: benchmark distributed Wc square root'
```

## Task 4: Build the feature branch with bundled CPU ELPA on df_dcu

**Files:**

- Create: versioned remote source/build directory under `/work1/ghj/app/src/`
- Modify: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [ ] **Step 1: Verify the local source state without building locally**

```bash
git status --short
git rev-parse HEAD
git log -1 --format='%H %an <%ae> / %cn <%ce> / %s'
```

Expected: only the already-known untracked build directories, `.DS_Store`, and
preserved developer summary remain; tracked files are clean.

No LibRPA configure, compile, MPI test, or calculation is run on the local
workstation. Local actions are limited to source/document editing, Git checks,
packaging, and inspection of downloaded artifacts.

- [ ] **Step 2: Stage without Git metadata or build outputs**

Set `commit=$(git rev-parse --short=12 HEAD)` and stage to
`/work1/ghj/app/src/librpa_mnf2_wc_sqrt_${commit}_20260811/LibRPA` using a tar
stream that excludes `.git`, `build-*`, and `.DS_Store`. Write the full commit
to `CODEX_SOURCE_COMMIT` in the remote source root.

- [ ] **Step 3: Configure the ELPA-enabled build**

In the same remote shell source
`/public/home/ghj/app/src/env_60_245_intel2021.sh`, then run:

```bash
cmake -S . -B build_df_dcu_intel2021_wc_elpa \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpiicpc \
  -DCMAKE_Fortran_COMPILER=mpiifort \
  -DLIBRPA_USE_LIBRI=ON \
  -DLIBRPA_USE_BUNDLED_ELPA=ON \
  -DLIBRPA_BUNDLED_ELPA_OPENMP=ON \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_DRIVER=ON
cmake --build build_df_dcu_intel2021_wc_elpa -j16
```

- [ ] **Step 4: Verify the build instead of trusting configuration intent**

Require cache entries `LIBRPA_USE_BUNDLED_ELPA:BOOL=ON`,
`LIBRPA_BUNDLED_ELPA_OPENMP:BOOL=ON`, `LIBRPA_USE_LIBRI:BOOL=ON`; require
`test_wc_sqrt_solver` and `chi0_main.exe`; run `ldd` and confirm an ELPA library
is linked; record both SHA256 values and the CMake configure log.

- [ ] **Step 5: Run a small four-rank remote smoke**

Run `n=32`, block 8 for ScaLAPACK and ELPA on one allocated node. Both must
print `status=PASS`; otherwise stop before production-dimension jobs.

## Task 5: Run the three production-dimension solver lanes

**Files:**

- Create: remote run root under the preserved MnF2 root named
  `wc_sqrt_solver_isolation_20260811`
- Modify: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [ ] **Step 1: Create one common Slurm template**

Use partition `long`, 16 nodes, one MPI rank per node, 30 OpenMP threads and 30
MKL threads, 20-minute time limit, matrix dimension 1078, block 128. The script
must print selected environment variables, source commit, binary SHA, node
list, and `MPI_Query_thread` result before the benchmark.

- [ ] **Step 2: Validate native networking on the same allocation**

Before lanes B/C, run a two-rank Allreduce smoke with TCP overrides unset and
the intended native OFI/verbs provider. Require size 2, correct sum, and exit
0. If MPI initialization fails with the prior GID assertion, record the exact
nodes and provider and do not label the matrix solver as failed.

- [ ] **Step 3: Submit lane A, the TCP control**

Set `FI_PROVIDER=tcp`, `I_MPI_OFI_PROVIDER=tcp`, and `UCX_TLS=tcp,self`; set
`LIBRPA_TEST_WC_USE_ELPA=0`. Record job ID immediately. A 20-minute timeout is
an expected diagnostic outcome, not a passing result.

- [ ] **Step 4: Submit lane B, ScaLAPACK on native networking**

Unset all TCP/UCX overrides and set `LIBRPA_TEST_WC_USE_ELPA=0`. Require
`COMPLETED 0:0`, `status=PASS`, and both numerical thresholds.

- [ ] **Step 5: Submit lane C, ELPA on native networking**

Use the same native environment and set `LIBRPA_TEST_WC_USE_ELPA=1`. Require
the same completion and numerical gates.

- [ ] **Step 6: Record a decision table**

Append job ID, state, exit code, elapsed time, solver time, residuals, node
list, and final log marker for A/B/C. Apply the design decision rules without
changing production code.

- [ ] **Step 7: Commit the completed solver evidence**

Commit only the updated validation report and checked plan boxes with the user
attribution convention.

## Task 6: Run the reduced `nfreq=6` downstream GW gate

**Files:**

- Create: isolated remote directory beside
  `/work1/ghj/gw/mnf2_dojo_tzdp10_abfs_shrink_sym_headwing_k6x6x9_gw_20260807/librpa_qowner_full_nfreq16_e097c60b_20260810`
- Modify: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [ ] **Step 1: Clone inputs without overwriting old evidence**

Hard-link immutable producer, band, PyATB, Coulomb and reader-v1 artifacts.
Copy scripts and `librpa.in`. Preserve all old run directories.

- [ ] **Step 2: Make only the validated changes**

Set `nfreq=6`; select the passing square-root backend; remove forced TCP if a
native lane passed. Keep k mesh 6x6x9, PP/basis/ABFS, symmetry, shrink, head and
wing, quasiparticle path, 16 nodes/ranks and 30 OpenMP threads unchanged.

- [ ] **Step 3: Verify provenance and submit one job**

Record an input diff, source commit, binary SHA, cache ELPA state and Slurm dry
run. Submit only after all checks pass and write the job ID to `FORMAL_JOB_ID`.

- [ ] **Step 4: Apply downstream completion gates**

Require all six chi0/shrink passes, all 65 Wc q points, and entry into the
first `Sigma_c` work. Also require no NaN, MPI/BLACS/ELPA error, or missing
reader-v1 input. Record stage timings and maximum memory.

- [ ] **Step 5: Stop on an attributable failure**

If Wc fails, record the precise q point and solver phase. Do not submit another
full job until the failure is explained. If both native distributed solvers
failed earlier, return to a separately specified q-owned/local-LAPACK design.

## Task 7: Restore `nfreq=16`, validate and plot final GW bands

**Files:**

- Create: fresh formal remote run directory
- Populate: `/Users/ghj/同步空间/overleaf/GW_pseudopotential_NAO/.codex_tmp/mnf2_wc_solver_fix_20260811/final`
- Modify: `docs/develop/mnf2_wc_sqrt_solver_validation.md`

- [ ] **Step 1: Restore only `nfreq=16`**

Copy the passing reduced directory, clear only generated LibRPA outputs, set
`nfreq=16`, verify the input diff, and submit one formal job.

- [ ] **Step 2: Require final artifacts**

Require scheduler `COMPLETED 0:0`, `libRPA finished successfully`,
`LIBRPA_OK`, and six nonempty exactly-310-line files:
`GW_band_spin_1.dat`, `GW_band_spin_2.dat`, `EXX_band_spin_1.dat`,
`EXX_band_spin_2.dat`, `KS_band_spin_1.dat`, and `KS_band_spin_2.dat`.

- [ ] **Step 3: Download and plot with the established policy**

Download the six bands, `band_kpath_info`, `KPT` as `KPT_nscf`, and
`band_out`. Use the ten-segment path
`Gamma-X-M-Gamma-Z-R-A-Z-X-R-M-A`, 31 explicit points per segment. Never
reorder across the occupied/conduction boundary; align to the GW VBM and report
the gap, VBM/CBM locations, spin residual, NaN count and discontinuity checks.

- [ ] **Step 4: Inspect the PNG and close the evidence record**

Generate PNG/PDF/machine-readable summary, inspect the PNG visually, append
absolute artifact paths and remaining limitations, run all targeted local
tests once more, then commit the final developer report.
