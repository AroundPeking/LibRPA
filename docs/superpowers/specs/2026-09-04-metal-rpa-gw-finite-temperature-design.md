# Finite-Temperature Metallic RPA and GW Design

## Goal

Extend the ABACUS-to-LibRPA space-time workflow to a genuine thermal
Fermi-Dirac reference, then validate metallic RPA and one-shot GW on bcc Na.
The extension must leave the current zero-temperature semiconductor behavior
unchanged unless the finite-temperature mode is selected explicitly.

The formal derivation and acceptance equations are maintained in
`/Users/ghj/同步空间/AITP_project/metal-gw/notebook/overleaf/metal_gw/main.tex`.

## Scope

The first supported statistical reference has one chemical potential and one
positive electronic temperature shared by all spins, k points, and bands:

```text
f(epsilon) = 1 / (exp((epsilon - mu) / kBT) + 1)
```

Gaussian, Methfessel-Paxton, cold-smearing, fixed, and arbitrary ensemble
occupations are rejected by the thermal path. They remain valid inputs to the
unchanged legacy path where currently supported.

Implementation is staged:

1. Producer metadata and occupation validation.
2. Stable thermal Green-function factors and a dense finite-beta reference
   transform.
3. Complete finite-q response from the existing CGGC contraction.
4. Analytic metallic Gamma head/wing treatment, with a distinct static mode.
5. Finite-temperature RPA free energy.
6. Finite-temperature screened interaction and GW self-energy.
7. Frequency-grid and k-integration acceleration after dense references pass.
8. bcc Na convergence and literature comparison.

## Producer Contract

ABACUS continues to write `band_out`. A versioned sidecar named
`thermal_occupation_v1.dat` is added when the RPA producer uses Fermi-Dirac
occupations. It contains:

```text
format thermal_occupation_v1
occupation_model fermi_dirac
chemical_potential_ha <value>
kbt_ha <value>
smearing_sigma_ry <value>
occupation_storage band_out_times_nk
spin_channels <value>
kpoints_per_spin <value>
bands <value>
```

The sidecar does not duplicate all occupations because `band_out` already
contains them. LibRPA reads both files and verifies every occupation against
the declared Fermi-Dirac distribution after undoing the producer's k-point
normalization.

The ABACUS temperature input is corrected so that for Fermi-Dirac smearing

```text
smearing_sigma [Ry] = k_B [Ry/K] * smearing_sigma_temp [K].
```

This corrects a factor-of-two unit error in the convenience temperature input.
Direct `smearing_sigma` input keeps its existing Ry meaning.

## Runtime Model

The LibRPA API gains an explicit occupation-reference enum. The default is
legacy. The Fermi-Dirac value requires positive `kbt_ha` and a chemical
potential consistent with the mean-field Fermi energy. The standalone driver
activates it only after reading valid producer metadata.

No thermal behavior is inferred merely from fractional occupations. This
prevents accidental changes to semiconductor and ensemble-occupation runs.

## Stable Green-Function Factors

For `xi = epsilon - mu`, the thermal branch computes logarithmic amplitudes:

```text
tau > 0: log[(1-f) exp(-xi tau)] = -xi tau - log1pexp(-xi/kBT)
tau < 0: log[f exp(-xi tau)]     = -xi tau - log1pexp( xi/kBT)
```

The result is multiplied by the existing k-point and spin normalization. The
legacy branch retains its existing clipping behavior until its own regression
tests justify a separate cleanup. All serial, k-distributed, and symmetry-
restored Green-function builders call the same thermal helper.

## Dense Finite-Beta Reference Grid

The first implementation uses `N_tau` midpoint samples

```text
tau_j = (j + 1/2) beta / N_tau,  j = 0, ..., N_tau - 1
```

and nonnegative bosonic frequencies

```text
nu_l = 2 pi l / beta,  l = 0, ..., N_nu - 1.
```

The forward transform matrix is complex:

```text
T_lj = (beta / N_tau) exp(i nu_l tau_j).
```

It is kept separate from the current zero-temperature minimax cosine
matrices. The initial correctness tests use sufficiently large `N_tau`; DLR,
cppdlr, sparse-ir, and any spline acceleration are evaluated only after this
reference path is validated.

## Response and Long-Wavelength Treatment

At finite q, no separate Drude matrix is added. Fractional occupations enter
the positive- and negative-time Green functions, and the existing CGGC product
must reproduce the complete finite-temperature Adler-Wiser response.

At Gamma, the analytic replacement has two branches:

- `l > 0`: intraband head and wings use the velocity-weighted
  `df/de = -f(1-f)/kBT` limits.
- `l = 0`: the full same-band auxiliary matrix is constructed from
  `df/de` and same-band density vertices. The static singular Coulomb channel
  is integrated over the Gamma cell through screened `W`; it is not stored as
  an infinite dielectric-matrix entry.

Interband head and wing terms remain active and are added to the intraband
contribution. Existing velocity matrices provide diagonal band velocities for
the dynamic intraband head. The static wings/body require same-band auxiliary
density vertices and therefore are a separate implementation task, not an
assumed zero.

## RPA and GW

Finite-temperature RPA uses the bosonic Matsubara sum with the zero mode once
and positive modes twice. The existing zero-temperature frequency integral is
unchanged in legacy mode.

Finite-temperature GW uses the same thermal Green function and screened
interaction on compatible grids. Exchange uses the same Fermi-Dirac
occupations. Analytic continuation is validated independently of response,
k-grid, and Gamma-cell errors.

## Numerical Selection

Dense direct sums are the correctness references. Candidate accelerators are
compared at fixed temperature, spectral range, and physical tolerances:

- DLR/libdlr/cppdlr;
- sparse-ir/libsparseir;
- direct Matsubara truncation with analytic high-frequency tails;
- tetrahedron integration for static Fermi-surface sums;
- symmetry-preserving double-grid or interpolation for finite-q response.

The chosen default must improve wall time or node-hours while meeting the same
response, RPA, screened-interaction, and self-energy tolerances. Fewer grid
points alone are not an acceptance result.

## Regression and Failure Policy

- Thermal mode is opt-in and requires valid metadata.
- Missing, malformed, non-FD, or inconsistent metadata fails before response
  construction.
- Existing runtime defaults and file readers remain compatible.
- Serial, k-parallel, and symmetry-restored thermal paths share numerical
  helpers and matching tests.
- Semiconductor RPA/GW regression references are run with thermal mode off
  before any production result is accepted.

## First Material Benchmark

bcc Na is computed with a fixed lattice, PBE pseudopotential, TZDP 10 au NAO
basis, matching auxiliary basis, empty-state window, and Coulomb settings.
Convergence axes are k mesh, temperature, time/frequency grid, Gamma treatment,
and continuation. Report PBE and G0W0 occupied bandwidths, their difference,
Fermi-level alignment, numerical uncertainty, wall time, and peak memory.

