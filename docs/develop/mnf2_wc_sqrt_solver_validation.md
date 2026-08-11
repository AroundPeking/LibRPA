# MnF2 Wc square-root solver validation

This is an append-only evidence log for the MnF2 Gamma-point Wc square-root
diagnosis. Keep successful, failed, timed-out, and cancelled attempts. Later
summaries may interpret these entries, but must not replace or delete them.

## Preserved production failure boundary

### Attempt P0: full `nfreq=16` q-owner run

| Field | Value |
| --- | --- |
| Date | 2026-08-10 |
| Purpose / changed variable | Continue the q-owned local-BLAS chi0 shrink implementation through the full MnF2 GW workflow |
| Local and remote directory | Local report: `GW_pseudopotential_NAO/.codex_tmp/mnf2_qowner_full_nfreq16_e097c60b_20260810/run-report.md`; remote: `/work1/ghj/gw/mnf2_dojo_tzdp10_abfs_shrink_sym_headwing_k6x6x9_gw_20260807/librpa_qowner_full_nfreq16_e097c60b_20260810` |
| Source commit / dirty state | Production-code baseline `e097c60b0d5ebf9ec3d064b7c3808d2573dc52c4`; remote source recorded this commit |
| Executable SHA256 | `d39baf4c1ff9264f7494fec715ec3f151cf607b92b53763b9b6e2eba49b97627` |
| CMake ELPA/LibRI/LibComm settings | `LIBRPA_USE_LIBRI=ON`; bundled ELPA `OFF`; external ELPA `OFF`; bounded LibComm/q-owner branch |
| Nodes / MPI ranks / OMP / MKL threads | 16 / 16 / 30 / 30; one MPI rank per node; BLACS grid `4x4` |
| MPI provider variables | Forced `FI_PROVIDER=tcp`, `I_MPI_OFI_PROVIDER=tcp`, `UCX_TLS=tcp,self`; `LIBCOMM_TRANS_MODE=sendrecv_ring` |
| Slurm job and scheduler result | `21568982`; cancelled after 7:37:23 |
| Last completed phase | All 16 chi0/shrink passes completed; Wc truncated-Coulomb preparation reached `epsilon_prepare_coulwc_sqrt_4` |
| Wc failure boundary | `power_hemat_blacs_real -> pdsyev`, real symmetric `1078 x 1078`, block 128, grid `4x4` |
| Final artifacts | No GW/EXX/KS band files |
| Evidence-bounded conclusion | The Wc Coulomb collection and preparation before the square root completed. The stop is at the distributed real-symmetric eigensolver. This attempt does not distinguish forced TCP from a ScaLAPACK fault. |
| Next action | Compare ScaLAPACK/TCP, ScaLAPACK/native OFI, and genuinely enabled ELPA/native OFI at dimension 1078 before repeating GW. |

The input requested `use_elpa_sqrt_coulomb=t`, but the executable had no ELPA
backend compiled. The request therefore selected the ScaLAPACK fallback. This
distinction must be checked from `CMakeCache.txt` and linked libraries in every
new attempt, rather than inferred from `librpa.in`.

## Attempt template

Copy this section for every new attempt and fill all fields. Do not edit an
older attempt to make a later interpretation appear retroactive.

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

For remote attempts retain the submission script, exact input, environment
dump, CMake cache, executable hash, `sacct` result, standard output/error, and
rank log. The report may link to large files in the preserved run directory.

## Attempt T1: red build of the Wc square-root test

| Field | Value |
| --- | --- |
| Date | 2026-08-11 |
| Purpose / changed variable | Prove that the newly registered MPI test fails specifically because the benchmark API has not yet been implemented |
| Local and remote directory | Local worktree `mnf2-libcomm-bounded-exchange`; remote `/work1/ghj/app/src/librpa_mnf2_wc_sqrt_red_20260811_01/LibRPA` |
| Source commit / dirty state | Local HEAD `2d595abe` plus uncommitted red-test additions to `src/test/CMakeLists.txt` and `src/test/test_wc_sqrt_solver.cpp`; recorded remotely in `CODEX_SOURCE_STATE` |
| Build settings | Intel C++/Fortran 2021.3; Release; `LIBRPA_USE_LIBRI=OFF`; tests ON; driver OFF; bundled/external ELPA OFF |
| Build location | df_dcu login shell after sourcing `/public/home/ghj/app/src/env_60_245_intel2021.sh`; no local compilation |
| Result | CMake configuration and `rpa_lib` completed; `test_wc_sqrt_solver.cpp` compilation exited with code 2 |
| First relevant errors | `env_positive_int` undefined; `env_flag` undefined; `run_wc_sqrt_benchmark` undefined |
| Evidence-bounded conclusion | The test target reaches its source compile under the intended df_dcu compiler environment and fails for the intended missing implementation. No matrix solver ran and this is not a solver failure. |
| Retained logs | Remote `configure-red.log` and `build-red.log` |
| Next action | Commit the red test, implement only the deterministic benchmark and numerical gates, then rebuild on df_dcu. |

## Workflow correction: all builds and tests run on df_dcu

On 2026-08-11 an initial local configure attempt stopped before compiling the
new test because AppleClang did not automatically locate Homebrew OpenMP. A
second local configure supplied the known `libomp` paths, but it was terminated
before completion when the user clarified that LibRPA builds and tests must run
on the server. Neither local attempt reached the Wc test and neither provides a
solver result. All subsequent configure, compile, MPI-test, and GW work is
performed on df_dcu; the local worktree is used only for source, Git, reports,
and downloaded-result inspection.
