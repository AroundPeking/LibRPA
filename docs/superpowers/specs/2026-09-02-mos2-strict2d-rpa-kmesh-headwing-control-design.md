# MoS2 strict-2D SOS-RPA k-mesh and head/wing control design

## Status and scope

This design extends the accepted MoS2 strict-2D SOS-RPA smoke from two qavg
meshes to a matched k-mesh comparison with and without analytic Gamma
head/wing treatment. It is a LibRPA-only campaign. It must not rerun ABACUS or
PyATB, overwrite an existing producer, reuse an invalid historical result, or
promote the OML route beyond `TESTABLE` on its own.

The physical result is the qavg branch. The no-headwing branch is a numerical
control that measures the omitted Gamma-cell correction; it is not a second
strict-2D production route.

## Immutable calculation identity

- Host: `df_dcu`
- Partition: `normal` only
- Work root: `/work1`
- LibRPA revision: `c87103df00b772ddbfc21597884c2787cf685037`
- LibRPA executable SHA-256:
  `0ca485dde5833dd190709c59d4689c263c2344041f2c71be4bc33c7e0b5da3c9`
- Reader contract: Coulomb reader v1 and LRI reader v1
- Response: SOS RPA, `task = rpa`
- Frequency grid: the existing minimax grid with `nfreq = 16`
- Coulomb: full 2D Ewald; cut Coulomb remains diagnostic-only
- Basis, structure, PP, NAO, ABFS, shrink data, occupations, and symmetry:
  inherited without modification from each validated producer
- Resources: four nodes, four MPI ranks, one rank per node with
  `mpirun -ppn 1 -np 4`, and 30 OpenMP plus 30 MKL threads per rank
- WorkDir, StdOut, and StdErr: under `/work1`

## Producer set

Use the existing validated producer family under:

```text
/work1/ghj/2d_gw_gth_multimaterial_validation_20260806/
validation_20260817/mos2_dualpp_current_campaign_20260817/
gth_siab_dzp77/kmesh/n<N>_lz25/producer
```

The selected in-plane meshes are `N = 8, 10, 12, 16`. Producer availability
has been observed for all four meshes. Completeness, reader-v1 identity, basis
size, q-point counts, and the matching `librpa_2d_coulomb_head.dat` must still
be revalidated immediately before a run is staged.

## Matched calculation matrix

| Mesh | qavg head/wing | No head/wing |
|---:|:---:|:---:|
| 8 | reuse accepted job `21833983` | new |
| 10 | new | new |
| 12 | reuse accepted job `21834156` | new |
| 16 | new | new |

This requires six new LibRPA-only jobs. Accepted N=8 and N=12 qavg jobs must
be referenced by their validation hashes instead of resubmitted.

Each new run uses a fresh sibling directory below:

```text
/work1/ghj/strict2d_rpa_validation_20260901/runs/
mos2_kmesh_headwing_control_c87103df_20260902/N<N>/<branch>
```

where `<branch>` is `qavg` or `nohw`. Before each submission, calculate a
fingerprint from the producer path, `librpa.in`, executable SHA-256, resource
layout, and branch label. Search active and completed Slurm jobs plus existing
receipts for that exact fingerprint. An existing active or passed duplicate
blocks submission.

## Input isolation

The qavg input retains the accepted route:

```text
task = rpa
nfreq = 16
replace_w_head = t
option_dielect_func = 3
use_2d_dielectric = t
use_pyatb = t
rpa_headwing_mode = qavg
rpa_headwing_body_start = 1
version_coul_reader = 1
version_lri_reader = 1
```

The no-headwing control changes exactly one active switch:

```text
replace_w_head = f
```

All other input keys, paths, thresholds, frequency nodes, and resource values
must remain byte-for-byte identical to the matched qavg case. In particular,
`rpa_headwing_mode = qavg` may remain in the input but is inactive when
`replace_w_head = f`. Keeping `use_pyatb = t` preserves the same reader bundle;
the production code must report that the RPA head/wing route itself is disabled.

Do not add a `head_only` branch to this campaign.

## Per-job acceptance

Every new job must pass all of the following independently:

1. Slurm `COMPLETED` with `ExitCode=0:0` on `normal`.
2. Application and wrapper exit codes are both zero.
3. Actual MPI world size is four, with one rank per node and 30 OpenMP/MKL
   threads per rank.
4. Input echo confirms `task=rpa`, `nfreq=16`, reader-v1, and the intended
   `replace_w_head` value.
5. All q contributions and the total RPA energy are finite.
6. Maximum single-q and summed imaginary energies are below `1e-8 Ha`.
7. The weighted real q sum agrees with `Total EcRPA` within
   `max(5e-9 Ha, |E| * 1e-8)`.
8. Finite-q LU diagnostics have `info=0`; the maximum physical
   anti-Hermitian relative residual is at most `5e-12`.
9. A qavg run has exactly 16 qavg records and maximum weight-sum error below
   `1e-6`.
10. A no-headwing run has no active qavg replacement markers and does not
    consume an analytic Gamma correction.

Scheduler or application completion alone is not numerical acceptance.

## Cross-branch invariants

For each mesh define:

```text
Delta_E_HW(N) = E_qavg(N) - E_nohw(N)
Delta_E_Gamma(N) = E_Gamma_qavg(N) - E_Gamma_nohw(N)
Delta_E_nonGamma(N) = E_nonGamma_qavg(N) - E_nonGamma_nohw(N)
```

Because only the Gamma treatment changes, require:

```text
abs(Delta_E_nonGamma) < max(5e-9 Ha, abs(E_nonGamma) * 1e-8)
abs(Delta_E_HW - Delta_E_Gamma) < max(5e-9 Ha, abs(Delta_E_HW) * 1e-8)
```

A failure of either identity blocks convergence interpretation and triggers an
input/source audit. It must not be hidden by taking a real part, loosening the
tolerance, or relabeling the no-headwing branch.

## Convergence analysis

Analyze qavg and no-headwing totals separately. Compare the fixed models

```text
E(N) = E_inf + a / N^2
E(N) = E_inf + b / N^3
```

using all four meshes and report residuals. Also repeat the comparison on the
largest three meshes as an asymptotic-window check. Do not infer an exponent
from two points, and do not treat N=16 as an exact reference.

For the Gamma contribution, report `E_Gamma(N)`, its fraction of the total and
non-Gamma energy, and `N^2 * E_Gamma(N)`. The qavg branch is expected to show a
smooth two-dimensional q-cell area scaling, but this is an acceptance
observation rather than an assumed fitted law.

## Figure and data products

Write one machine-readable CSV containing mesh, branch, job, source and binary
identity, total, Gamma, non-Gamma, imaginary diagnostics, and validation status.

After all eight matrix entries are accepted, produce one three-panel figure:

1. `Total EcRPA versus N` for qavg and no-headwing.
2. `Gamma contribution versus N`, with `N^2 E_Gamma` shown on a separate axis
   or inset without mixing units.
3. `Delta_E_HW`, `Delta_E_Gamma`, and `Delta_E_nonGamma` versus N.

The figure must distinguish calculated observations from fitted curves, show
Hartree and eV conversions consistently, and label no-headwing as a diagnostic
control. Save PNG and vector PDF/SVG variants alongside the CSV and validation
summary. Do not plot failed or incomplete jobs as accepted points.

## OML treatment

The qavg cases remain under profile
`abacus-librpa-2026-09-02-strict2d-sos-rpa-v1`. The no-headwing series is a
one-axis diagnostic evolution candidate and must not be recorded as a passed
strict-2D production route. Only after all cross-branch invariants pass may the
control and figure be appended to the OML live benchmark record.

Completion of this campaign may support a k-mesh convergence assessment. It
does not automatically promote the route beyond `TESTABLE`.
