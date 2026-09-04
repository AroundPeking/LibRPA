# Finite-Temperature Metallic RPA and GW Implementation Plan

> Execute tasks in order. Each behavior change starts with a failing focused
> test and ends with the named verification command. Do not start the Na GW
> benchmark until the response and Gamma-cell gates pass.

**Goal:** Add an opt-in Fermi-Dirac finite-temperature path to LibRPA, validate
metallic RPA, and then extend the validated path to G0W0 for bcc Na.

**Architecture:** ABACUS exports a versioned thermal sidecar next to the
existing band data. LibRPA validates that contract, uses shared stable thermal
Green-function amplitudes, and adds a finite-beta complex time-to-Matsubara
transform without modifying the legacy minimax path. The finite-q CGGC bubble
remains complete; only the metallic Gamma limit is supplied analytically.

**Stack:** C++17, CMake/CTest, MPI/OpenMP, existing LibRI/LRI contractions,
ABACUS reader-v1 producer, PyATB velocity data, optional DLR/sparse-IR libraries
after dense-reference validation.

---

## Task 1: Freeze Baselines and Contracts

**Files:**
- Add: `docs/superpowers/specs/2026-09-04-metal-rpa-gw-finite-temperature-design.md`
- Add: `docs/superpowers/plans/2026-09-04-metal-rpa-gw-finite-temperature.md`
- Reference: `/Users/ghj/同步空间/AITP_project/metal-gw/notebook/overleaf/metal_gw/main.tex`

1. Record branch heads and clean worktree status.
2. Configure a local LibRPA build with C++ tests and run `test_timefreq`,
   `test_meanfield`, and `test_rpa_headwing` unchanged.
3. Configure the ABACUS unit-test build needed by the producer plan and run its
   existing input and LibRPA sidecar unit tests unchanged.
4. Commit the design and plan before implementation.

## Task 2: ABACUS Thermal Producer Metadata

**Repository:** `/Users/ghj/code-worktrees/abacus-metal-finite-t`

Follow
`docs/superpowers/plans/2026-09-04-abacus-librpa-thermal-producer.md` in that
repository. Completion requires a sidecar fixture readable independently of
ABACUS and tests for temperature conversion, metadata fields, non-FD omission,
and `band_out` normalization.

## Task 3: Thermal Reference State and Validation

**Files:**
- Add: `src/core/thermal_occupation.h`
- Add: `src/core/thermal_occupation.cpp`
- Add: `src/test/test_thermal_occupation.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `src/test/CMakeLists.txt`
- Modify: `src/api/dataset.h`
- Modify: `include/librpa_enums.h`
- Modify: `include/librpa_options.h`
- Modify: `src/api/librpa.cpp`
- Modify: `driver/driver.h`
- Modify: `driver/inputfile.cpp`
- Modify: `driver/read_data.cpp`
- Modify: `docs/user_guide/runtime_parameters.yml`

1. Write failing tests for stable FD occupation and derivative at
   `|epsilon-mu|/kBT` up to 1000.
2. Implement `log1pexp`, FD occupation, FD derivative, and thermal Green
   amplitude helpers.
3. Write failing tests for metadata parsing and rejection of nonpositive
   temperature, non-FD model, inconsistent chemical potential, and occupation
   mismatch.
4. Add the opt-in occupation-reference option with legacy default.
5. Read `thermal_occupation_v1.dat` in the driver and validate `band_out`.
6. Verify:
   `ctest --test-dir build-metal -R 'test_thermal_occupation|test_io' --output-on-failure`.

## Task 4: Dense Finite-Beta Time/Frequency Grid

**Files:**
- Modify: `src/core/librpa_enums.h`
- Modify: `src/core/timefreq.h`
- Modify: `src/core/timefreq.cpp`
- Modify: `src/test/test_timefreq.cpp`
- Modify: `include/librpa_options.h`
- Modify: `src/api/librpa.cpp`
- Modify: `driver/inputfile.cpp`
- Modify: `docs/user_guide/runtime_parameters.yml`

1. Write failing tests for midpoint time nodes, Matsubara frequencies, complex
   transform entries, and exact transforms of a constant for all represented
   nonzero modes.
2. Add an explicit finite-beta grid type and separate complex transform
   storage; do not reinterpret existing cosine matrices.
3. Require positive beta and independent positive `N_tau` and `N_nu`.
4. Verify existing grid tests and new finite-beta cases.

## Task 5: Stable Thermal Green Functions

**Files:**
- Modify: `src/core/meanfield.cpp`
- Modify: `src/core/meanfield_mpi.cpp`
- Modify: `src/core/chi0.cpp`
- Modify: `src/core/dielecmodel.cpp`
- Modify: `src/test/test_meanfield.cpp`
- Modify: `src/test/test_meanfield_mpi.cpp`

1. Add failing one-state and multi-state tests above and below the chemical
   potential, including values where the legacy exponential would overflow.
2. Route every serial, k-distributed, symmetry-restored, and head/wing Green
   builder through the shared thermal amplitude only in thermal mode.
3. Keep signs and k/spin normalization identical to each caller's current
   contract.
4. Verify serial/MPI equality and antiperiodic amplitude identities.

## Task 6: Complex Thermal Bubble Transform

**Files:**
- Modify: `src/core/chi0.cpp`
- Modify: `src/core/chi0.h`
- Add: `src/test/test_thermal_chi0.cpp`
- Modify: `src/test/CMakeLists.txt`

1. Add a failing scalar two-level test comparing the `GG` time transform with
   the direct finite-temperature Adler-Wiser expression at zero and nonzero
   bosonic frequencies.
2. Add a failing same-level test: the zero mode equals `-beta f(1-f)` and all
   represented nonzero modes vanish.
3. Use the complex finite-beta transform in thermal mode and keep cosine
   transforms in legacy mode.
4. Add a finite-q tight-binding test separating same-band and interband terms
   without adding a second Drude response.

## Task 7: Metallic Dynamic Head and Wings

**Files:**
- Modify: `src/core/dielecmodel.h`
- Modify: `src/core/dielecmodel.cpp`
- Modify: `src/test/test_rpa_headwing.cpp`

1. Add failing tests for the velocity-weighted `-df/de` tensor on anisotropic
   model bands.
2. Add the intraband tensor to, rather than replace, the existing interband
   head for every nonzero Matsubara frequency.
3. Add dynamic wing tests using explicit same-band auxiliary vertices.
4. Verify full-BZ and symmetry-restored sums agree.

## Task 8: Metallic Static Gamma Cell

**Files:**
- Modify: `src/core/dielecmodel.h`
- Modify: `src/core/dielecmodel.cpp`
- Modify: `src/core/epsilon.cpp`
- Modify: `src/core/epsilon.h`
- Modify: `src/test/test_rpa_headwing.cpp`

1. Add failing tests for the complete static same-band auxiliary matrix.
2. Represent the static singular channel by its screening coefficient and
   analytic Gamma-cell integral, not an infinite matrix value.
3. Test 3D Thomas-Fermi limits for several Gamma-cell shapes and directions.
4. Add separate strict-2D tests before enabling the existing 2D q-averaging
   infrastructure for metals.

## Task 9: Finite-Temperature RPA

**Files:**
- Modify: `src/api/compute_rpa.cpp`
- Modify: `src/core/epsilon.cpp`
- Add: `src/test/test_thermal_rpa.cpp`
- Modify: `src/test/CMakeLists.txt`

1. Add failing Matsubara-sum tests with the zero mode counted once and positive
   modes twice.
2. Implement the thermal RPA thermodynamic-potential path behind the explicit
   thermal option.
3. Verify its low-temperature limit against the converged zero-temperature
   integral on a gapped model, then verify a metallic model independently.

## Task 10: Finite-Temperature GW

**Files:**
- Modify: `src/core/epsilon.cpp`
- Modify: `src/core/gw.cpp`
- Modify: `src/api/compute_g0w0.cpp`
- Add: `src/test/test_thermal_gw.cpp`
- Modify: `src/test/CMakeLists.txt`

1. Add failing tests for inverse bosonic transform using the negative-frequency
   Hermitian relation.
2. Add the thermal `G W^c` convolution and thermal exchange occupations.
3. Validate a two-level self-energy against direct Matsubara summation.
4. Keep analytic-continuation tests separate from imaginary-axis tests.

## Task 11: Numerical Acceleration Bake-Off

**Files:**
- Add: `utilities/benchmark_thermal_frequency_grids.py`
- Add: `docs/develop/thermal_frequency_grid_benchmark.md`

1. Establish dense-reference results for model and Na response matrices.
2. Evaluate direct Matsubara truncation, cppdlr/libdlr, and sparse-ir at fixed
   physical tolerances.
3. Record matrix error, RPA error, selected self-energy error, points, runtime,
   memory, compiler/MPI integration, dependency cost, and license.
4. Select no production default until one candidate passes all gates.

## Task 12: k-Space Convergence and Na RPA/GW

**Files:**
- Add: `regression_tests/metal_na/` inputs and reference manifest after the
  remote calculation setup is fixed.
- Add: `docs/tutorial/g0w0/metal_na.md` after numerical acceptance.

1. Produce bcc Na PBE data in ABACUS with Fermi-Dirac temperature and fixed
   Dojo NC-SR TZDP 10 au basis inputs.
2. Converge direct symmetry-reduced meshes in temperature and k density.
3. Compare tetrahedron static sums and symmetry-preserving interpolation with
   the dense direct baseline.
4. Converge thermal RPA and Gamma-cell screening before GW.
5. Converge the occupied G0W0 bandwidth and compare with literature using the
   same stated structural and starting-point definitions.
6. Run semiconductor RPA/GW regression suites with thermal mode off.

