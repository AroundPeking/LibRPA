# MnF2 shrink ScaLAPACK optimization design

## Status and objective

The MnF2 `g0w0_band` calculation with 65 q points, 16 minimax time/frequency
points, two spin channels, 1884 raw auxiliary functions, and 1078 active
auxiliary functions entered `shrink_chi0_abfs` but did not finish its first
call after more than eight hours on a 4 x 4 BLACS grid. The job was canceled
after 8:52:24. Its physical inputs and output logs are preserved.

The objective is to make the shrink transformation complete at a practical
rate without changing

\[
  \chi_0^{\mathrm{small}}(q)=U(q)\chi_0^{\mathrm{large}}(q)U(q)^\dagger,
\]

and to establish that the same MPI/ScaLAPACK runtime can proceed through the
later screened-Coulomb and self-energy stages before another full MnF2 job is
submitted.

## Evidence and root-cause hypothesis

The earlier bounded LibComm ring exchange completed the real-space chi0
collection and symmetry restoration. Live stack samples in the subsequent
shrink stage were instead inside

```text
shrink_abfs_chi0 -> pzgemm -> PB_CpgemmAB -> BLACS -> PMPI_Send/Recv
```

The current shrink descriptors are initialized with
`init_square_blk_capped(..., 2048, ...)`. For a 4 x 4 process grid this produces
coarse blocks of about 471 for the 1884-dimensional matrix and 270 for the
1078-dimensional matrix. In contrast, the screened-Coulomb implementation
redistributes matrices to an explicitly capped block size of 128 and documents
that maximal blocks are not suitable for distributed matrix operations.

The primary hypothesis is therefore the combination of:

1. coarse one-block-per-process-direction matrix layouts in shrink;
2. two distributed complex GEMMs for each of 65 q points;
3. the shrink call being repeated for 16 time points and two spins;
4. a TCP MPI provider forced by the earlier default-provider initialization
   failure.

This does not yet prove that ScaLAPACK is generally broken. It predicts that a
production-sized `pzgemm` benchmark will distinguish an unsuitable shrink
layout from a provider-wide BLACS problem.

## Selected staged design

### Stage 1: exact-dimension ScaLAPACK gate

Add an MPI test that constructs the production shapes 1884 x 1884,
1078 x 1884, and 1078 x 1078 and performs the same two complex GEMMs as
`shrink_abfs_chi0`. The test accepts a block size for benchmarking but does not
expose a new production runtime option.

Benchmark block sizes 64, 128, 256, and the existing automatic coarse layout
under the same Intel MPI/MKL build. Run representative 4- and 16-rank tests
with the TCP provider and one available high-speed-provider control. Record
wall time, process grid, descriptor block sizes, and maximum numerical error.

The selected production block size must:

- complete without MPI errors on 1, 4, and 16 ranks;
- agree with a serial reference within `1e-10` relative Frobenius norm;
- be the fastest stable candidate on the production-sized benchmark;
- avoid a sustained state in which every rank spends minutes only in BLACS
  send/receive progress.

The expected candidate is 128 because the Wc implementation already uses that
cap, but the benchmark result, not the expectation, chooses the value.

### Stage 2: production shrink descriptor and progress evidence

Replace the coarse shrink descriptors with one common tested block size for
the large-large, small-large, and small-small matrices. Keep the mathematical
operations and q-point order unchanged.

Add root-rank timing at the following boundaries:

- q-point start and completion;
- input `comm_map2_first` collection;
- first `pzgemm`, `U * chi0`;
- second `pzgemm`, `(U * chi0) * U^H`;
- output `comm_map2_first` redistribution.

Routine output should remain compact: print every q point when the existing
`LIBRPA_VERBOSE_DEBUG` level is enabled, and otherwise print the first q point,
every tenth q point, and the last q point. No new runtime switch is introduced
and no matrix contents are printed.

The production-like 65-q shrink gate must complete in less than 30 minutes on
the chosen 16-rank df_dcu configuration. This leaves a conservative upper
bound of about 16 hours for all 32 time/spin shrink calls, rather than exceeding
the seven-day queue limit in the first call.

### Stage 3: downstream ScaLAPACK health gate

Before another full GW calculation, run a reduced-frequency MnF2 job through:

1. chi0 shrink;
2. at least one Wc q/frequency block using its 128-block layout;
3. the screened-Coulomb solver and one self-energy construction step.

This separates a shrink-specific descriptor problem from a provider-wide
ScaLAPACK failure. The full 16-frequency calculation is submitted only after
these stages have explicit timing output and no MPI/BLACS errors.

### Stage 4 fallback: q-point ownership with local threaded GEMM

If no tested ScaLAPACK block size provides an acceptable 65-q runtime, stop
tuning block sizes. Assign q points to MPI ranks, gather one q-point matrix to
its owner, perform `U * chi0 * U^H` with node-local threaded BLAS, and
redistribute the result to the existing atom-pair ownership.

This fallback changes parallel ownership and is therefore a separate change
with its own serial/4-rank/16-rank numerical tests. It is not mixed into the
minimal descriptor fix.

## Numerical and physical invariants

- The raw and active auxiliary bases remain 1884 and 1078 functions.
- `sinvS`, q-point ordering, spin handling, symmetry restoration, and Fourier
  transforms are unchanged.
- No global sorting, threshold, or physical input is changed.
- Hermiticity residuals and serial-versus-distributed shrink residuals are
  recorded for every validation case.
- Different block sizes may change floating-point summation order only; any
  discrepancy larger than the stated numerical tolerance fails the gate.

## Test-driven implementation sequence

1. Add a descriptor test requiring the selected shrink block size; confirm it
   fails with the current coarse descriptor.
2. Add the production-shape two-GEMM test and confirm the new path is not yet
   available.
3. Implement the smallest descriptor helper and timing hooks needed to pass.
4. Run the targeted serial and MPI tests, then the existing matrix/BLACS tests.
5. Build the exact df_dcu executable and record source revision and SHA256.
6. Run the block-size/provider benchmark and select the measured winner.
7. Run the 65-q shrink and reduced-frequency downstream gates.
8. Submit the full MnF2 GW job only after all gates pass.

## Completion criteria

The work is complete only when:

- the new and existing targeted tests pass;
- the df_dcu 65-q shrink gate finishes within the stated limit;
- a reduced-frequency run crosses Wc and self-energy ScaLAPACK stages;
- the full MnF2 calculation reports `libRPA finished successfully` and writes
  all six expected 310-line KS/EXX/GW band files;
- the final GW band figure and quantitative summary are generated.
