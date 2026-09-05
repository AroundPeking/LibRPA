# Sparse Finite-Temperature GW Operators

This is an internal numerical-validation stage. The public finite-temperature
GW guard remains enabled. The exporter is not yet a production driver input.
Do not interpret an operator test as a Na quasiparticle or bandwidth result.

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
consumer is a test-only fixture reader feeding `ThermalGWTransform`.
Production API, restart, and driver integration are separate work.

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

Before material GW admission, still validate ordered full-complex W q/R
transforms, the production positive-time LibRI path, fermionic metadata
through restart and KS rotation, metallic Gamma treatment, and continuation.
Do not remove the public guard based on these model tests.
