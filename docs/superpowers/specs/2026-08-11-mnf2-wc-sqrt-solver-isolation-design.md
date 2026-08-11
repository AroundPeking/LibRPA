# MnF2 Wc square-root solver isolation design

## Purpose

Identify why the full MnF2 GW calculation stops during the first Gamma-point
Wc preparation, without mixing this new failure boundary with the previously
resolved `set_Cs` and chi0-shrink problems. The selected solver and runtime
environment must be validated at the production matrix dimension before the
expensive GW workflow is repeated.

Every attempted lane, including failed or cancelled jobs, must be recorded in
`docs/develop/mnf2_wc_sqrt_solver_validation.md`. The report is part of the
eventual code-change evidence and must not be replaced by a success-only
summary.

## Established evidence

- The production-code baseline is LibRPA commit
  `e097c60b0d5ebf9ec3d064b7c3808d2573dc52c4` on branch
  `codex/mnf2-libcomm-bounded-exchange`.
- df_dcu job `21568982` used a binary with SHA256
  `d39baf4c1ff9264f7494fec715ec3f151cf607b92b53763b9b6e2eba49b97627`.
- All 16 chi0/shrink passes completed. The run reached Wc at the first Gamma
  point and stopped in the square-root step of the truncated Coulomb matrix.
- The active operation was the ScaLAPACK real-symmetric eigensolver `pdsyev`
  for a `1078 x 1078` matrix, block size 128, on a `4 x 4` BLACS grid.
- The run forced the TCP OFI provider. It was cancelled after 7:37:23 and
  produced no GW band files.
- Although `librpa.in` requested `use_elpa_sqrt_coulomb=t`, that executable was
  built with both bundled and external ELPA disabled. The runtime request
  therefore fell back to the ScaLAPACK implementation.

The evidence localizes the stop to the distributed matrix square root. It does
not yet distinguish a TCP transport problem from a ScaLAPACK problem.

## Scope

This work will:

1. add a standalone Wc-like real-symmetric matrix-square-root validation test;
2. build the same feature branch with bundled CPU ELPA enabled;
3. compare solver and network lanes at the production dimension;
4. use the validated lane for a reduced `nfreq=6` GW downstream gate; and
5. restore the full `nfreq=16` GW run only after Wc and the first self-energy
   work have been observed.

This work will not initially change the Wc physical equations, Coulomb data,
shrink basis, k mesh, frequency integration, or quasiparticle solver. A new
q-owner/local-LAPACK Wc implementation is a fallback only if both distributed
solvers fail under the native high-speed network.

## Alternatives considered

### Selected: production-dimension solver isolation

Run a small standalone executable that exercises the same real symmetric
matrix-square-root path at dimension 1078. Compare ScaLAPACK and ELPA while
holding the matrix, MPI layout, node count, rank count, OpenMP count, compiler,
and source revision fixed. This gives the most attributable result before the
multi-hour GW calculation is repeated.

### Rejected for now: immediately repeat the full ScaLAPACK GW run

Removing the TCP override and repeating the full job changes only one runtime
factor, but several hours are needed to reach Wc. A second stop would still not
show whether ELPA is a viable solution.

### Deferred: implement a q-owned local Wc square root

This could avoid distributed eigensolver communication, but it changes
production data movement and introduces a memory bottleneck at one rank. It is
premature until the existing ScaLAPACK and ELPA paths have been tested under a
working native network.

## Standalone validation test

The test will construct a deterministic, dense, well-conditioned,
real-symmetric positive-definite matrix distributed with the production
descriptor:

- dimension: 1078;
- BLACS grid: `4 x 4` on 16 MPI ranks;
- ScaLAPACK block size: 128;
- topology: one MPI rank per node and 30 OpenMP/MKL threads per rank.

The matrix must be non-diagonal so that the eigensolver and redistribution
paths are exercised. After computing the square root `S`, the test will report:

- wall time of the square-root call;
- relative Frobenius residual `||S S - A||_F / ||A||_F`;
- Hermiticity residual `||S - S^T||_F / ||S||_F`;
- finite-value checks; and
- the solver and provider selected at runtime.

The initial numerical gates are:

- square-root relative residual no larger than `1e-9`;
- Hermiticity relative residual no larger than `1e-12`;
- no NaN or infinity; and
- completion within a 20-minute lane limit.

The time limit is a diagnostic stop rule, not a performance target. A timed-out
lane is recorded with its last completed phase and resource state.

## Controlled lanes

All lanes use the same ELPA-enabled feature-branch binary and input matrix.
Runtime selection chooses the backend.

| Lane | Square-root backend | Network provider | Role |
| --- | --- | --- | --- |
| A | ScaLAPACK | forced TCP | reproduce the suspect runtime combination |
| B | ScaLAPACK | native high-speed OFI/verbs | isolate the TCP effect |
| C | bundled CPU ELPA | native high-speed OFI/verbs | test the alternative solver |

Native-network lanes must clear `FI_PROVIDER=tcp`, `I_MPI_OFI_PROVIDER=tcp`,
and `UCX_TLS=tcp,self`. The exact working provider and MPI initialization are
verified with a same-allocation MPI smoke test before the matrix test.

## Build and provenance

The build remains on `codex/mnf2-libcomm-bounded-exchange`; upstream `master`
is not substituted. Bundled CPU ELPA is enabled explicitly. For every binary,
the validation report records:

- Git branch and full commit;
- dirty-tree status;
- full CMake configure command and relevant cache entries;
- compiler, MPI, MKL, LibRI, LibComm, and ELPA versions;
- executable path, size, modification time, and SHA256; and
- linked MPI/MKL/ELPA libraries.

The existing untracked build directories and developer report are preserved.

## Decision rules

1. If lane B passes and is substantially faster than lane A, retain the
   ScaLAPACK implementation and remove forced TCP from the production retry.
2. If lane B fails but lane C passes, use the genuinely ELPA-enabled binary and
   `use_elpa_sqrt_coulomb=t` for the production retry.
3. If both lanes B and C pass, select the more stable/faster backend, while
   retaining the other result as an independent numerical cross-check.
4. If both lanes B and C fail, stop before another full GW run. Then design a
   q-owned/local-LAPACK Wc square-root path with a separate specification and
   numerical equivalence tests.

No conclusion is drawn from scheduler state alone. A passing lane requires the
completion marker, numerical residuals, scheduler `COMPLETED 0:0`, and retained
logs.

## GW downstream gates

After selecting a backend:

1. Create a fresh isolated retry directory from the original MnF2 inputs.
2. Change only the proven solver/network settings and set `nfreq=6`.
3. Require all chi0/shrink passes, all Wc q points, and entry into at least the
   first `Sigma_c` work. Record per-stage timing and memory behavior.
4. If the reduced run passes, restore `nfreq=16` with otherwise identical
   physical inputs.
5. Final completion requires `libRPA finished successfully`, `LIBRPA_OK`, and
   six nonempty 310-line GW/EXX/KS band files before plotting.

## Evidence record

`docs/develop/mnf2_wc_sqrt_solver_validation.md` will contain an append-only
chronology with one section per attempt:

- date, purpose, and changed variable;
- local/remote directories and Slurm job ID;
- source/binary provenance;
- complete resource and MPI/OpenMP environment;
- input diff against the immediately preceding lane;
- scheduler result and exact first/last relevant log lines;
- timing, residual, and memory table;
- conclusion limited to what that lane establishes; and
- next action.

Remote run directories, Slurm scripts, CMake cache, standard output/error,
rank-0 LibRPA log, and numerical result files are retained. The branch report
links to them rather than copying only selected success messages.
