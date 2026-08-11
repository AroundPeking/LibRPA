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
