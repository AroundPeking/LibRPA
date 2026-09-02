# MoS2 Strict-2D RPA K-Mesh Head/Wing Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and validate a matched MoS2 `N=8,10,12,16` strict-2D SOS-RPA comparison between qavg head/wing and a no-headwing diagnostic control, then generate convergence and Gamma-contribution figures.

**Architecture:** Reuse immutable reader-v1 ABACUS/PyATB producers and the accepted LibRPA binary. Stage fresh symlink-only LibRPA run directories, vary only `replace_w_head`, submit six duplicate-guarded `normal` jobs, validate every job independently, then combine the six new results with the two accepted qavg anchors.

**Tech Stack:** LibRPA `c87103df`, Intel MPI 2021.3, Slurm on `df_dcu`, Bash staging and receipts, Python 3 validation/analysis, Matplotlib scientific figures, OML admission profile `abacus-librpa-2026-09-02-strict2d-sos-rpa-v1`.

---

### Task 1: Lock profile, producer, binary, and duplicate gates

**Files:**
- Read: `/work1/ghj/strict2d_rpa_validation_20260901/validation_package_c87103df_20260902/producer_manifest.csv`
- Read: `/work1/ghj/strict2d_rpa_validation_20260901/build_c87103df00b7/chi0_main.exe`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/PREFLIGHT_RECEIPT`

- [ ] **Step 1: Inspect the OML profile and admission manifest**

Call `inspect_profile` for `abacus-librpa-2026-09-02-strict2d-sos-rpa-v1` and `inspect_admission_manifest` for `df-dcu-strict2d-sos-rpa-2026-09-02-v1`.

Expected: route `strict_2d_sos_rpa`, status `TESTABLE`, LibRPA revision `c87103df00b772ddbfc21597884c2787cf685037`, binary SHA-256 `0ca485dde5833dd190709c59d4689c263c2344041f2c71be4bc33c7e0b5da3c9`, `normal`, `/work1`, four ranks, and 30 threads.

- [ ] **Step 2: Revalidate the four selected producers**

Run the existing producer validator on a manifest filtered to N=8,10,12,16. Expected counts are `(full_k,ibz,basis)=(64,34,77),(100,52,77),(144,74,77),(256,130,77)` and full/cut Coulomb file counts equal `3*ibz`.

- [ ] **Step 3: Verify executable identity**

Run:

```bash
sha256sum /work1/ghj/strict2d_rpa_validation_20260901/build_c87103df00b7/chi0_main.exe
```

Expected: `0ca485dde5833dd190709c59d4689c263c2344041f2c71be4bc33c7e0b5da3c9`.

- [ ] **Step 4: Check active and completed duplicates**

Search `squeue`, `sacct`, existing `SUBMITTED_JOB_ID`, `RUN_RECEIPT`, and fingerprint files for the six exact `(N,branch,commit,binary,input,resource)` identities. Expected: zero active or accepted duplicates.

### Task 2: Stage six fresh LibRPA-only cases

**Files:**
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N8/nohw/`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N10/qavg/`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N10/nohw/`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N12/nohw/`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N16/qavg/`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/N16/nohw/`

- [ ] **Step 1: Stage symlink-only datasets from producer manifests**

Use `stage_librpa_runs.sh` with a filtered N=8,10,12,16 manifest, then place each staged dataset under the branch-specific paths above. Do not copy producer payload files and do not modify producer directories.

- [ ] **Step 2: Create matched inputs**

Copy the accepted qavg `librpa.in` into all six directories. Keep qavg inputs byte-identical. In each no-headwing input, mechanically replace only:

```text
replace_w_head = t
```

with:

```text
replace_w_head = f
```

Verify `cmp` after normalizing this one line; expected: no other difference.

- [ ] **Step 3: Create branch-specific Slurm scripts**

Derive each script from the accepted N=12 script. Set job name, N label, branch label, and `--chdir`; retain `normal`, four nodes, four tasks, one task per node, 30 CPUs per task, `110000M` per node, four-hour wall time, Intel environment, `mpirun -ppn 1`, OMP/MKL equality, actual MPI-world fail-closed check, binary hash check, and finite-q diagnostics.

- [ ] **Step 4: Write immutable submission fingerprints**

For each case, hash `librpa.in`, Slurm script, binary, producer path, branch, and resource tuple. Write `SUBMISSION_FINGERPRINT` and `SUBMISSION_INPUTS.sha256` before submission.

### Task 3: Run static and scheduler admission gates

**Files:**
- Create: each case `PRESUBMISSION_RECEIPT`

- [ ] **Step 1: Validate dataset links and producer counts**

Require zero broken links, basis 77, matching IBZ/full-grid counts, complete full/cut reader-v1 Coulomb families, the producer-local `librpa_2d_coulomb_head.dat`, and full-grid PyATB velocity/eigenvector data.

- [ ] **Step 2: Validate input isolation**

Require qavg cases to echo `replace_w_head=t`; require controls to echo `replace_w_head=f`; require both to have `task=rpa`, `nfreq=16`, reader-v1, full 2D Ewald files, no `head_only`, and identical remaining keys.

- [ ] **Step 3: Syntax-check scripts**

Run `bash -n` on all six scripts. Expected: exit 0.

- [ ] **Step 4: Run `sbatch --test-only`**

Run once per script. Expected: accepted by Slurm on `normal`, with WorkDir/StdOut/StdErr under `/work1`.

- [ ] **Step 5: Repeat duplicate check immediately before submission**

Expected: zero active or accepted exact fingerprints. A match blocks only that case and reuses its existing receipt.

### Task 4: Submit exactly six jobs and monitor them

**Files:**
- Create: each case `SUBMITTED_JOB_ID`
- Create: each case `RUNTIME_RECEIPT.<jobid>`
- Create: each case `APPLICATION_EXIT.<jobid>`
- Create: each case `PROCESS_LAYOUT_CHECK.<jobid>`

- [ ] **Step 1: Submit each eligible script once**

Capture the numeric job id atomically in `SUBMITTED_JOB_ID`. Never retry an uncertain submission until `squeue`, `sacct`, and run receipts prove no job exists.

- [ ] **Step 2: Verify started layouts**

For each running job verify `normal`, four nodes, four tasks, 120 CPUs, one rank per node, WorkDir under `/work1`, `OMP_NUM_THREADS=MKL_NUM_THREADS=30`, and application echo `Total number of tasks : 4`.

- [ ] **Step 3: Monitor without mutation**

Observe scheduler state, log growth, stderr, and application markers. Do not cancel, migrate, or resubmit a queued/running job. Unchanged queue state is not a failure.

### Task 5: Validate each completed job

**Files:**
- Create: each case `VALIDATION.<jobid>.json`
- Update: each case `RUN_RECEIPT`

- [ ] **Step 1: Validate scheduler, application, and process gates**

Require Slurm `COMPLETED/0:0`, application/wrapper exit zero, and process-layout `PASS`.

- [ ] **Step 2: Validate numerical outputs**

Require finite energies, maximum single-q and summed imaginary energies below `1e-8 Ha`, real q sum equal to total within `max(5e-9,abs(E)*1e-8)`, finite-q LU info zero, and maximum physical anti-Hermitian relative residual no larger than `5e-12`.

- [ ] **Step 3: Validate branch semantics**

Require exactly 16 qavg records with weight error below `1e-6` for qavg. Require no active qavg replacement records for no-headwing and confirm the input echo has `replace_w_head=f`.

- [ ] **Step 4: Seal provenance**

Record job, N, branch, commit, binary hash, producer, input hashes, resources, energies, diagnostics, and validation status. A failed job remains preserved and is excluded from convergence plots.

### Task 6: Check matched-branch invariants and convergence

**Files:**
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/analysis/mos2_rpa_headwing_kmesh.csv`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/analysis/convergence_analysis.json`

- [ ] **Step 1: Combine six new results with two accepted qavg anchors**

Use jobs `21833983` and `21834156` only after rechecking their validation JSON hashes.

- [ ] **Step 2: Verify Gamma-only differences**

For every N require `Delta_E_nonGamma` and `Delta_E_HW-Delta_E_Gamma` within the registered mixed tolerance. Any failure blocks fitting and plotting as accepted physics.

- [ ] **Step 3: Fit fixed convergence models**

Fit `E_inf+a/N^2` and `E_inf+b/N^3` on N=8,10,12,16 and on N=10,12,16. Record coefficients, residual sum of squares, maximum residual, and extrapolated energy. Do not claim an exponent solely from the lower residual of one four-point fit.

### Task 7: Generate and verify the requested figure

**Files:**
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/analysis/mos2_rpa_kmesh_headwing_comparison.png`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/analysis/mos2_rpa_kmesh_headwing_comparison.pdf`
- Create: `/work1/ghj/strict2d_rpa_validation_20260901/runs/mos2_kmesh_headwing_control_c87103df_20260902/analysis/mos2_rpa_kmesh_headwing_comparison.svg`

- [ ] **Step 1: Plot three aligned panels**

Panel 1 shows total EcRPA versus N for qavg and no-headwing. Panel 2 shows Gamma contribution and a separately scaled `N^2 E_Gamma`. Panel 3 shows `Delta_E_HW`, `Delta_E_Gamma`, and `Delta_E_nonGamma`.

- [ ] **Step 2: Distinguish observations and fits**

Use markers for calculated points and lines only for labeled fits/guides. Label no-headwing as a diagnostic control and show Hartree/eV units consistently.

- [ ] **Step 3: Verify rendered outputs**

Open the PNG, confirm all eight accepted points, readable labels, no clipped axes, correct Gamma signs, and agreement with CSV values. Exclude failed points rather than silently interpolating them.

### Task 8: Record the validated campaign

**Files:**
- Modify: `/Users/ghj/.config/superpowers/worktrees/oh-my-librpa/oml-mcp-phase1/docs/live-benchmarks/2026-09-02-df-dcu-strict2d-sos-rpa.md`
- Modify: OML admission evidence only if all registered gates pass

- [ ] **Step 1: Append results and figure provenance**

Record all job ids, validation hashes, matched-branch invariants, convergence-fit scope, CSV/figure hashes, and remaining physical limitations.

- [ ] **Step 2: Run OML regression tests**

Run `.venv/bin/python -m unittest discover -s tests -q`. Expected: all tests pass.

- [ ] **Step 3: Commit documentation with required attribution**

Commit with author `Codex <codex@openai.com>` and committer `AroundPeking <gonghuanjing@iphy.ac.cn>`. Keep the route `TESTABLE` unless the convergence evidence independently meets the registered promotion policy.
