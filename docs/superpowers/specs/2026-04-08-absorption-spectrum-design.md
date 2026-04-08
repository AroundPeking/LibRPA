# LibRPA Absorption Spectrum Design

## Context

This design starts from branch `codex/librpa-gw-symmetry-final` and targets a new
development line for an RPA-level optical absorption workflow in LibRPA.

The user goal is not a generic dielectric-matrix refactor. The immediate goal is:

1. add a dedicated driver task for absorption-spectrum calculations;
2. reuse the existing `gw_band` front half as much as possible;
3. compute only `chi0(R)` and `chi0(q=0, i\omega)` for the optical workflow;
4. use the existing head/wing formalism at `q=0` to obtain
   `\epsilon^{-1}_{00}(i\omega)`;
5. invert that scalar to form the macroscopic dielectric function on the
   imaginary axis;
6. analytically continue to the real axis and inspect whether the resulting
   absorption spectrum is physically reasonable, starting with Si.

The user also wants all major design and implementation notes mirrored into a
synced Overleaf note outside the code repository.

## Problem Statement

LibRPA currently has mature workflows for `RPA`, `G0W0`, `G0W0_band`, `QSGW`,
`QSGW_band`, and `Wc_Rf`, but it does not expose a dedicated task that takes the
head/wing-corrected `q=0` dielectric response all the way to an optical
observable.

The codebase already contains most of the required building blocks:

- `driver/task_gw.cpp` and `driver/task_gw_band.cpp` construct
  `chi0 -> epsilon^{-1} -> Wc`;
- `src/dielecmodel.cpp` already computes the head, wing, `Lind`, and the
  averaged inverse dielectric quantity used by the head/wing machinery;
- `src/analycont.cpp` already provides scalar Pade analytic continuation.

The missing piece is a focused workflow that packages these pieces into a
physically transparent optical task with stable outputs and validation hooks.

## Scope

### In Scope

- Add a new driver task dedicated to absorption-spectrum calculations.
- Restrict the optical workflow to `q=0`.
- Reuse the existing head/wing path for the optical `Gamma`-point treatment.
- Export the imaginary-axis scalar/macroscopic response needed for optical
  continuation.
- Export the full `3x3` tensor components that naturally arise from the
  head/wing path for diagnostics and future extension.
- Perform scalar analytic continuation per tensor component using the existing
  Pade implementation.
- Validate the first implementation on Si.

### Out of Scope for the First Development Round

- Full BSE / excitonic corrections.
- A large refactor of the generic `epsilon.cpp` GW screening kernels.
- Aggressive matrix-size minimization beyond the explicit `q=0` reduction.
- In-code plotting.
- New command-line tooling outside the normal LibRPA driver.

## Design Goals

1. **Physics-first correctness**
   The first version should produce a believable Si optical response before any
   optimization aimed only at reducing cost.

2. **Minimal intrusion into GW workflows**
   The new optical path should not destabilize existing `G0W0` or `QSGW`
   calculations.

3. **Explicit data products**
   The task must write clearly documented text outputs for both the imaginary and
   real frequency axes so that the user can inspect and post-process the results
   directly.

4. **Structured validation**
   The first implementation must preserve enough intermediate information to
   diagnose whether a mismatch comes from head/wing construction, scalar
   inversion, or analytic continuation.

## Recommended Architecture

### 1. New Dedicated Driver Task

Add a new task, tentatively named `absorption`, with its own driver entry point.

The new task should be wired through:

- `src/task.h`
- `driver/main.cpp`
- `driver/CMakeLists.txt`
- `driver/task_absorption.h`
- `driver/task_absorption.cpp`

This task keeps the optical workflow independent from `Wc_Rf` and `gw_band`.

### 2. Reuse `gw_band` Front-Half Data Preparation

The new task should reuse the same early-stage setup pattern already proven in
`task_gw_band.cpp`:

- construct `Rlist`;
- build time/frequency grids;
- create `Chi0`;
- set `gf_R_threshold`;
- call `chi0.build(...)`.

The crucial optical simplification is that `qlist` should contain only the
Gamma point.

### 3. Keep the First Optical Kernel Conservative

Even though the target observable is scalar
`\epsilon^{-1}_{00}(i\omega)`, the first implementation should not attempt to
re-derive a brand-new ultra-minimal linear-algebra path.

Instead, it should:

- reuse the existing head/wing machinery in `src/dielecmodel.cpp`;
- expose the scalar quantity already computed through the head/wing averaging
  logic;
- keep the naturally available tensor information (`head`, `Lind`) for
  diagnostics.

This keeps the first version physically aligned with current LibRPA conventions
and reduces implementation risk.

### 4. Separate Optical Post-Processing from the Driver

Introduce a small optics-oriented helper module, tentatively:

- `src/optics.h`
- `src/optics.cpp`

Responsibilities:

- define a compact result structure for imaginary-axis optical data;
- perform scalar inversion from `\epsilon^{-1}_{00}(i\omega)` to
  `\epsilon_M(i\omega)`;
- perform per-component Pade continuation to the real axis;
- write well-labeled output tables.

The driver task remains an orchestrator rather than becoming another
large monolithic file.

## Data Flow

### Imaginary Axis

1. Read standard LibRPA inputs.
2. Build `chi0(R)` and `chi0(q=0, i\omega)`.
3. Initialize and run `df_headwing` for the same frequency grid.
4. Extract the head tensor `head(i\omega)`.
5. Extract or reconstruct the head/wing-derived optical scalar
   `\epsilon^{-1}_{00}(i\omega)`.
6. Form
   `\epsilon_M(i\omega) = 1 / \epsilon^{-1}_{00}(i\omega)`.
7. Also keep tensor diagnostics such as `Lind(i\omega)` and the head tensor
   components, with full `3x3` output by default.

### Real Axis

1. Build the complex imaginary-axis sampling points
   `i\omega_n`.
2. For each scalar series selected for continuation, construct
   `AnalyContPade`.
3. Evaluate on a user-defined real-axis mesh.
4. Write `Re`/`Im` parts explicitly.

The first version should continue:

- the scalar macroscopic response used for the final optical spectrum;
- the tensor components, when available, as diagnostics.

## Output Contract

The task should write at least these files into `Params::output_dir`:

### `absorption_imag_axis.dat`

Columns should include:

- `omega_imag_ha`
- `epsinv00_re`
- `epsinv00_im`
- `epsm_re`
- `epsm_im`
- explicit tensor columns for every component needed in inspection, for example
  `head_xx`, `head_xy`, ..., `head_zz`, `Lind_xx`, `Lind_xy`, ..., `Lind_zz`

The first implementation should default to exporting the full `3x3` tensor
components rather than only diagonal entries.

### `absorption_real_axis.dat`

Columns should include:

- `omega_real_ha`
- `epsm_avg_re`
- `epsm_avg_im`
- `epsm_xx_re`, `epsm_xx_im`
- `epsm_yy_re`, `epsm_yy_im`
- `epsm_zz_re`, `epsm_zz_im`
- and, when continuation is performed component-wise, the remaining tensor
  components needed for full `3x3` inspection

### `absorption_summary.txt`

Human-readable summary of:

- the task name;
- frequency-grid settings;
- analytic-continuation settings;
- whether `use_2d_dielectric` was active;
- which quantity was used to define the final macroscopic dielectric function.

## Physical Conventions

The implementation must keep these names distinct in both code and
documentation:

- matrix-element notation such as dielectric-matrix `00` components;
- macroscopic dielectric function `\epsilon_M`.

For this feature, the working definition of the final optical scalar is:

` \epsilon_M(i\omega) = 1 / \epsilon^{-1}_{00}(i\omega) `

This should be stated verbatim in both the code-facing spec and the Overleaf
note, because it is easy to confuse with the `00` element of the dielectric
matrix itself.

For first-pass spectrum inspection, the task should make it straightforward to
plot at least:

- `Im \epsilon_M(\omega)` as the primary optical-response curve;
- the averaged complex dielectric function;
- directional components for anisotropy checks.

## Validation Strategy

### Primary Validation System

Use Si first.

Reasons:

- cubic symmetry makes anisotropy checks straightforward;
- the expected optical response is comparatively well-behaved;
- it is a practical first filter for implementation mistakes.

### Required Checks for the First Round

1. `xx`, `yy`, and `zz` components should be close on both imaginary and real
   axes.
2. `\epsilon_M(i\omega)` should decay smoothly with increasing imaginary
   frequency.
3. The continued real-axis response should not show obviously non-physical
   numerical spikes from unstable continuation.
4. The first absorption spectrum should be visually reasonable at the
   RPA/macroscopic level, even if it is not expected to match experiment exactly.

### Failure Interpretation

- If tensor components disagree strongly for Si, first suspect implementation or
  convention mismatch.
- If the imaginary-axis data look smooth but real-axis data are noisy, suspect
  the continuation stage.
- If `\epsilon^{-1}_{00}` itself behaves oddly, inspect the head/wing interface
  before changing continuation settings.

## Risks and Mitigations

### Risk 1: Confusing `\epsilon^{-1}_{00}` with `\epsilon_M`

Mitigation:

- keep separate variable names and output columns;
- state the exact scalar inversion convention in the summary file and Overleaf
  note.

### Risk 2: Over-optimizing too early

Mitigation:

- first make `q=0` the only explicit reduction;
- postpone deeper matrix-element pruning until the Si result is trustworthy.

### Risk 3: Unstable Pade continuation

Mitigation:

- always write the raw imaginary-axis data;
- keep the number of continuation parameters explicit in outputs;
- compare tensor components and averaged output on Si before trusting the
  spectrum.

## Implementation Order

1. Add the new driver task and task dispatch plumbing.
2. Implement the imaginary-axis optical data path for `q=0`.
3. Export scalar and tensor diagnostics.
4. Add Pade continuation and real-axis outputs.
5. Validate on Si.
6. Only after that, revisit whether the head/wing path can be reduced further to
   a stricter `00`-only implementation.

## External Research Record

In parallel with the code-side spec, maintain a synced Overleaf note under
`/Users/ghj/同步空间/overleaf/` for:

- design rationale;
- physics definitions;
- implementation notes;
- Si validation results;
- open questions and next steps.

The Overleaf note is the long-form research log; the repository spec is the
code-facing design reference.
