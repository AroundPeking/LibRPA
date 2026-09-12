# Sparse Finite-Temperature GW Operators

The initial public finite-temperature GW route uses independent external
operators from `fn_thermal_gw_grid`. It supports full-k, scalar-spin
3D CPU calculations with LibRI and ScaLAPACK, with optional EXX, response/q-star
and real-space Sigma symmetry. It is under synthetic and
regression validation; this is not material or arbitrary-boundary acceptance.

## Internal Spacetime Data Flow

`G0W0::build_thermal_spacetime` now joins the independent operators to the
actual distributed W buffers, AO Green builder, and complex LibRI GW contraction.
It accepts explicit complete signed-frequency/full-q Wc matrices. The internal
`thermal_Wc_freq_q_to_tau_R` routine preserves all ordered complex matrix entries,
uses `exp(-i*2*pi*q*(R*latvec))/Nq`, and applies rectangular B in the supplied
signed-label order. The coefficients already include inverse-temperature
normalization; neither a zero-mode half weight nor a conjugate completion is used.

IndexScheduler collects full auxiliary-basis atom-pair W blocks from the global
consecutive BLACS descriptor. LibRI receives complex W and only positive-time
G on the independent GW times. The current `G_lib = -G_standard` convention
gives `Sigma_c = G_lib*Wc`; F is then accumulated directly without an extra
minus, one-half, or legacy sine/cosine transform. An uninitialized response
TFGrids object is valid for this internal calculation.

`ThermalGWTransform` constructs each `ThermalFrequencyGrid` once. The grid owns
beta, ordered integer labels and their stored frequency values; read access is
const and copying the grid does not evaluate the frequencies again. The returned
`ThermalSigcRspace` carries that fermionic grid, chemical potential and rank-local
AO/R Sigma matrices indexed by **integer Matsubara label**, not floating frequency.
`set_thermal_sigc` imports the supplied values directly: it neither reconstructs
`(2*n+1)*pi/beta` nor searches for a frequency within a tolerance. The existing
projection maps receive their floating keys from this same stored array. MPI
metadata checks include the full supplied grid before projection collectives.
Missing local blocks represent zero contribution;
the ordinary distributed sum is needed for global matrices. It deliberately does
not populate legacy GW caches by itself. The public consumer calls
`G0W0::set_thermal_sigc` to import its positive fermionic samples into the
existing real-space projection path. Self-energy output and continuation use
these samples, not the bosonic response grid. `Sigc_fermionic_grid.dat` records
beta, mu and positive integer labels with their frequencies. Legacy Sigma
restart and continuation-grid resampling remain explicitly unsupported.

This initial CPU reference requires scalar spin, replicated full-k SCF data,
uncompressed matching LRI data and a global W descriptor.
Physical metadata must be identical across ranks. It validates local inputs
collectively, including empty matrix owners; B batches skip zero local columns
without skipping collectives. Its bounded temporary buffers do not bound the
complete input/output W storage. Spatial summation is explicit, not an FFT or
a production-performance claim. `prepare_thermal_wc` now unfolds compressed
buffers as U-adjoint W U, restores response q stars in the original ABF basis,
and completes negative frequencies using the full matrix adjoint. General
distributed SCF, metallic Gamma accuracy, restart and spectral-bound convergence
still need acceptance.
Do not interpret an operator test as a Na quasiparticle or bandwidth result.

## Real-Space Sigma Symmetry

`use_symmetry_gw = t` enables only the existing output-only LibRI sector filter
and AO rotation recovery in the thermal builder. It does not reduce the supplied
k/q grids, rebuild G from k stars, regenerate frequencies, or interpolate C.
The independent EXX and response switches can also be enabled as described below.

The input Hamiltonian, FD reference, C and W must obey the supplied spatial
operations within each spin channel. At each positive time, contract only
irreducible `(I,J,R)` output blocks, then restore each member using
`rotate_symmetry_rspace_block`. This routine uses the established NAO convention
`T_I^T Sigma_IJ T_J*`, including orbital parity, atom permutation and the
periodic return translations encoded in the sector map. Accumulate the full
complex restored blocks using the unchanged fermionic transform. Rotation is
linear, so rank-local partial contributions are restored before the ordinary
distributed sum; no star weight is multiplied into a recovered matrix block.

AO shell metadata and a unique, complete restore partition are checked before
LibRI communication. Rank-local failures and differing on/off decisions fail
collectively. The existing conservative complete-band guard is retained: a
truncated response-band space disables the real-space filter. Identity-only
contexts also use the unchanged full contraction. The existing
`LIBRPA_DISABLE_SIGC_RSPACE_SYMMETRY_DIAG` diagnostic remains available.

Per-time logs separate W setup, G setup, LibRI contraction, symmetry recovery
and frequency accumulation. A model test reports full complex matrix errors;
tiny timings are not a material speedup benchmark. Tests are
`test_thermal_sigc_symmetry` (two equivalent s+p atoms, full three-point mesh,
nonzero parity-sensitive blocks, row/column-major W, distributed C, malformed
metadata and conservative fallbacks) and `test_thermal_sigc_symmetry_api`
(two s-orbital atoms, full response/screening/Sigma/KS/QP route). Run both at
one and four MPI ranks with one OpenMP/MKL thread per rank.

## Response q Stars And Exchange

`use_symmetry_rpa = t` reuses the existing chi0 real-space symmetry and
irreducible q-point view. Screening is evaluated only at those representatives.
`prepare_thermal_wc` first unfolds each representative with its own `sinvS`,
then calls the same distributed `restore_symmetry_dense_wq_map` used by the
zero-temperature Fourier path. This restores orbital rotations, atom/cell phases
and time reversal in the **original spherical-shell basis**, not in an arbitrary
compressed basis. After full-q recovery, negative bosonic samples are constructed
by the complete matrix adjoint. No q-star multiplicity is applied to restored
matrix elements; the unchanged q-to-R transform supplies `1/Nq` exactly once.
Input SCF eigenvectors still cover the full uniform grid in this first route.

`use_symmetry_exx = t` reuses the existing exchange implementation. Its density
matrix uses the supplied FD occupations; neither a new exchange formula nor a
Matsubara transform is needed. The shared symmetry context retains shell
rotations whenever any of EXX, response or Sigma requests them, so initializing
a later stage with its own switch off does not invalidate an earlier q view.

The public-route regression covers Gamma and three-point meshes, EXX-only,
response-only, Sigma-only, EXX+response and all switches on. It checks that
three q points reduce to two and compares every sampled KS self-energy, EXX
value and QP energy against the fixed full-grid calculation. The W preparation
test also uses an analytic complex q-dependent matrix with nontrivial cell
phases and a rank-reducing auxiliary projection. Run at one and four MPI ranks.

## Task And Band Ranges

The current driver treats `g0w0_band` as an alias of `g0w0`. A `g0w0` task
evaluates the uniform-grid QP states and, when `fn_band_kpath_info` exists,
also evaluates the supplied path. This is shared driver behavior, not a
finite-temperature change. `i_state_low`/`i_state_high` select the zero-based,
half-open QP range, independently of the number of KS states used by G and chi0.
For a 44-state reference, `[0,8)` outputs eight GW bands; `[0,44)` requests all
44. Do not equate `nbands=nbasis` with an automatically full QP output range.

## Independent Grids

The response/RPA grid and the GW grid are separate objects. The GW exporter
`utilities/generate_thermal_gw_grid.py` uses the full signed complex sampling
API of [sparse-ir](https://sparse-ir.readthedocs.io/en/latest/sampling.html).
It constructs the normalized rectangular operators

\[
\begin{aligned}
B &= U_B(\tau)\,\operatorname{fit}[\widehat U_B(i\nu)],
& W_c(\tau_j)&=\sum_m B_{jm}W_c(i\nu_m),\\
F &= \widehat U_F(i\omega)\,\operatorname{fit}[U_F(\tau)],
& \Sigma_c(i\omega_n)&=\sum_j F_{nj}\Sigma_c(\tau_j).
\end{aligned}
\]

Here `fit` means the external library's linear coefficient-reconstruction
operator (obtained by fitting an identity batch), not a finite Matsubara sum.
The common positive-time nodes are the fermionic self-energy sampling nodes.
The three dimensions `ntau`, `nboson`, and `nfermion` need not agree.
No same-frequency Hermitian completion or elementwise real projection is used.
Using `positive_only=True` would impose real time-domain elements and is not
appropriate for unrestricted complex matrix batches.

The spectral bounds are independent inputs, in Hartree:

- `g-wmax`: a bound on every included single-particle `abs(epsilon-mu)`.
- `w-wmax`: a bound on the screened correlation-interaction spectrum, which
  need not equal the bare electron-hole transition range.
- `sigma-wmax`: at least `g-wmax + w-wmax`, covering the support of the product.

These bounds and the IR basis truncation tolerance are convergence parameters,
not guarantees on a material self-energy error. In particular, the response
bound of 15 Ha must not silently become a screened-W or self-energy bound.
The common time grid is selected from the wider self-energy support to avoid
aliasing the product of G and W onto a response-only time grid.

## Signs And The Static Mode

LibRPA stores `G_lib = -G_standard`. Its positive-time contraction is

\[
\Sigma_c(\tau)=G_{\rm lib}(\tau)W_c(\tau),\qquad 0<\tau<\beta.
\]

There is no extra minus sign or factor of one half. `Wc` excludes bare V;
exchange remains separate. An isolated static contribution obeys

\[
W_c(i\nu_m)=A\delta_{m0}
\quad\Longrightarrow\quad W_c(\tau)=A/\beta.
\]

The exporter checks the uncorrected IR representation of this mode, records
its residual, then sets the zero-mode column of B exactly to `1/beta`.
This is not the RPA zero-mode half weight. All other columns are unchanged.

## File Contract

The ASCII `LIBRPA_THERMAL_GW_V1` layout is documented in the exporter's module
docstring. It stores beta, energy/time units, Fourier signs, spectral bounds,
tolerance, the actual ordered nodes and both complex coefficient arrays.
Bosonic labels m denote `2*pi*m/beta`; fermionic labels n denote
`(2*n+1)*pi/beta`. The sparse-ir odd label -1 therefore maps to n=-1,
not n=0. Arrays are row-major real/imaginary pairs at 17-digit precision.
Existing output files are never replaced.

Consumers must match beta and ordered frequency labels, not just matrix
shapes. Sampling locations can vary between library versions/platforms;
archive this complete file together with the sampled data. The current C++
consumer uses `librpa_set_external_thermal_gw_grid`, wrapped by C++ and Fortran,
to copy independent `ExternalThermalGWGrid` data into Dataset. The response/RPA
`TFGrids` and its dimensions remain unchanged. The versioned reader is under
`driver/reader_thermal_gw_grid.cpp`; the full contract is in
[dataset_format.md](dataset_format.md#external-thermal-gw-operators-v1).
Loading input alone does not run GW; `build_g0w0_sigma` consumes it on the supported route.

Setup checks the FD beta and all included `abs(epsilon-mu)` against `g_wmax`;
grid initialization repeats these reference checks, including chemical potential.
Replacing reference energies invalidates computed objects when GW input exists.
Failed GW setters leave the old input and calculation state intact. Clearing
the GW input does not clear the response grid or the FD reference, and clearing
the response input does not clear GW input. If changing to a zero-temperature
calculation, explicitly clear both thermal inputs and the FD reference.

Fortran arrays have shapes `B(nboson,ntau)` and `F(ntau,nfermion)`, matching
C row-major storage; their signed integer labels are not shifted by one.
Reader errors on any MPI rank are reduced before subsequent work. Valid input
must still be identical on all ranks; this failure reduction does not compare
different valid files. Restart remains separate work; the initial consumer
now connects Wc transforms, Sigma projection and the existing continuation API.

Reference revalidation certifies only the imported grid's compatibility.
For analytic head/wing, `initialize_ds_headwing` rebuilds its cache whenever
the current or cached reference is thermal. This avoids reusing a previous
temperature or same-count frequency grid, including thermal-to-zero-temperature
transitions. The ordinary zero-temperature cache policy is unchanged. Numerical
Gamma accuracy still requires material and quadrature validation.

## Verification

Run the Python generator tests in an environment containing sparse-ir,
NumPy, and SciPy:

```sh
python utilities/test_generate_thermal_gw_grid.py -v
```

They compare asymmetric complex poles and the composed W/G/self-energy
calculation to analytic expressions, including near-zero and spectral-edge
energies, beta=8 and beta=315.7750248494972 Ha^-1. A separate signed uniform
Matsubara convolution checks the GW sign and normalization. It is a cheap
model sum, not another material GG calculation.

The frozen fixture `src/test/data/sparse_ir_gw_beta8_g1_w2.dat` was generated
with sparse-ir 2.1.4:

```sh
python utilities/generate_thermal_gw_grid.py grid.dat --beta 8 \
  --g-wmax 1 --w-wmax 2 --sigma-wmax 3 --tolerance 1e-10
```

It has 19 time nodes, 17 signed bosonic labels and 20 signed fermionic labels.
`test_external_thermal_gw` exercises the C++ transform with this actual
exported fixture; it does not regenerate operators at CMake configure time
or require sparse-ir on the C++ test host.

The C++ test also composes the internal scalar screening solve with B/F.
When LibRI is enabled, it evaluates `RI::GW::cal_Sigmas` at the positive
sampling times for one atom, two AO and two auxiliary functions, comparing
with an explicit four-center-assignment index sum and analytic self-energy.
This bounded test requires one MPI rank; it is not a spatial/MPI production
validation. An optional file argument allows testing a wider frozen grid.
The direct uniform convolution is explicitly limited to the default small
window, with a separate cutoff-refinement check; wide-window outputs use
analytic references at every signed fermionic frequency.

The general-route tests additionally cover complex W q/R transforms, the
positive-time LibRI path, KS projection, auxiliary unfolding, and independent
EXX/response/self-energy symmetry switches in serial and MPI. The RPA head/wing
MPI regression includes ranks with zero local wing rows and the dense Fourier
regression includes ranks with no local input. Release builds keep the test
assertions enabled. These are implementation checks, not material convergence
acceptance. Thermal restart remains explicitly unsupported; metallic Gamma
quadrature and analytic continuation still require material-level validation.
