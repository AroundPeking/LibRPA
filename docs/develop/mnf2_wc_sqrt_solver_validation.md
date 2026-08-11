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

## Attempt T2: green ScaLAPACK test and nearby regressions

| Field | Value |
| --- | --- |
| Date | 2026-08-11 |
| Purpose / changed variable | Implement the deterministic dense SPD matrix, call the production `power_hemat_la_real` dispatcher, and verify distributed residuals |
| Local and remote directory | Local worktree `mnf2-libcomm-bounded-exchange`; remote `/work1/ghj/app/src/librpa_mnf2_wc_sqrt_red_20260811_01/LibRPA` |
| Source commit / dirty state | Local HEAD `dba19be4` plus the uncommitted green implementation; remote `CODEX_SOURCE_STATE` identifies the snapshot |
| Test executable SHA256 | `c2ba0d769f2f56f9e3836926df5af3924716b03c573d3b88ccf55720588c9a7f` |
| Build settings | Intel C++/Fortran 2021.3; Release; LibRI OFF; bundled/external ELPA OFF; tests ON; driver OFF |
| First green compile correction | The first green compile exposed missing declarations for `init_local_mat` and `power_hemat_blacs_real`; including `utils_matrix_m_mpi.h` before `la_connector.h` resolved the test-only include dependency |
| MPI execution | df_dcu login node, four MPI ranks, `I_MPI_FABRICS=shm`, one OpenMP/MKL thread per rank; matrix `n=32`, block 8, grid `2x2` |
| Result | `WC_SQRT_BENCH solver=scalapack ... sqrt_s=0.0853893570602 residual_gemm_s=0.000558719038963 filtered=0 relres=3.17641173216e-15 herm=3.11443271155e-17 finite=1 status=PASS` in the three-test regression run |
| Nearby regressions | `test_matrix_m_mpi`, `test_shrink_scalapack`, and `test_wc_sqrt_solver`: 3/3 passed, total 3.08 s |
| Retained logs | Remote `build-green.log`, `build-green-2.log`, `ctest-green-login.log`, `build-nearby-tests.log`, and `ctest-nearby-login.log` |
| Scheduler attempts | `21577991` (`debug`) and `21578004` (`normal`) were cancelled before starting. The debug reason was maintained/reserved nodes; the tiny 32x32 smoke was then run on the df_dcu login node. |
| Evidence-bounded conclusion | The new test and existing ScaLAPACK/shrink tests pass on df_dcu for the small four-rank case. This does not validate dimension 1078, inter-node communication, TCP, or ELPA. |
| Next action | Commit the green test, build the same feature branch with bundled CPU ELPA and LibRI enabled, then run the controlled production-dimension lanes through Slurm. |

## Attempt B0: first bundled-ELPA full build

| Field | Value |
| --- | --- |
| Date | 2026-08-11 |
| Purpose / changed variable | Build the committed feature branch with bundled CPU ELPA, OpenMP, LibRI, tests, and driver enabled |
| Remote source/build | `/work1/ghj/app/src/librpa_mnf2_wc_sqrt_5e0a9556_20260811/LibRPA`; `build_df_dcu_intel2021_wc_elpa` |
| Source commit | Clean archive of `5e0a9556df3de82738c81d0e9d038a54bd4fa2a3` before the CMake correction |
| Toolchain | `mpiicc`, `mpiicpc`, `mpiifort` from Intel MPI/compiler 2021.3; Release |
| Dependency settings | LibRI ON using bundled LibRI `e978e96`; bundled LibComm `12457e7+bounded-exchange`; bundled ELPA 2026.02.001 ON with OpenMP; external ELPA OFF; `MPI_THREAD_MULTIPLE` |
| Configure result | Completed with bundled ELPA and OpenMP explicitly reported ON |
| Build result | Bundled ELPA and `rpa_lib` completed; full build exited 2 when driver and `test_rpa_headwing` included public LibRPA headers |
| First relevant error | `mpi/base_blacs.h: cannot open source file "elpa/elpa.h"` in targets consuming `rpa_lib` |
| Root cause | `rpa_lib` linked the ELPA CMake target as `PRIVATE`, although public header `base_blacs.h` includes `elpa/elpa.h`; ELPA include/link requirements were not propagated to consumers |
| Retained logs | Remote `configure-elpa.log` and `build-elpa.log` |
| Evidence-bounded conclusion | ELPA itself compiled. The failure was a CMake target-interface defect before any ELPA numerical test or GW execution. |
| Minimal correction | Change only `target_link_libraries(rpa_lib PRIVATE ${LIBRPA_ELPA_TARGET})` to `PUBLIC`, with an explanatory comment. |
| Verification after correction | Reconfigure completed; `test_wc_sqrt_solver`, `test_rpa_headwing`, and `rpa_exe` built; a subsequent all-target build completed with exit 0. |

## Workflow correction: all builds and tests run on df_dcu

On 2026-08-11 an initial local configure attempt stopped before compiling the
new test because AppleClang did not automatically locate Homebrew OpenMP. A
second local configure supplied the known `libomp` paths, but it was terminated
before completion when the user clarified that LibRPA builds and tests must run
on the server. Neither local attempt reached the Wc test and neither provides a
solver result. All subsequent configure, compile, MPI-test, and GW work is
performed on df_dcu; the local worktree is used only for source, Git, reports,
and downloaded-result inspection.
