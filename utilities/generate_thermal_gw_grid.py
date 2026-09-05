#!/usr/bin/env python3
"""Export full-complex sparse-ir operators for finite-temperature correlation GW.

B maps Wc(i*nu_m) to Wc(tau_j); F maps Sigma_c(tau_j) to Sigma_c(i*omega_n).
Both use IR reconstruction, not quadrature over the sparse frequency labels.
The common positive-time grid is chosen for the wider fermionic Sigma support.
No conjugate completion, real projection, RPA half weight, or bare exchange is
included. LibRPA stores G_lib = -G_standard, so Sigma_c(tau) = G_lib(tau)*Wc(tau).

File layout (row-major complex arrays are real/imaginary pairs):
  LIBRPA_THERMAL_GW_V1
  sparse-ir VERSION
  Ha Ha^-1
  B exp_minus inverse F exp_plus integral
  beta g_wmax w_wmax sigma_wmax tolerance
  ntau nboson nfermion
  tau[ntau], m[nboson], n[nfermion], B[ntau,nboson], F[nfermion,ntau]
The signed integer labels mean nu=2*pi*m/beta and omega=(2*n+1)*pi/beta.
These are independent GW metadata, not an RPA grid or a public GW admission.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def build_grid(beta, g_wmax, w_wmax, sigma_wmax, tolerance):
    parameters = [beta, g_wmax, w_wmax, sigma_wmax, tolerance]
    if (not np.all(np.isfinite(parameters)) or min(parameters) <= 0 or tolerance >= 1):
        raise ValueError("require finite positive beta and spectral bounds, tolerance in (0,1)")
    summed_wmax = g_wmax + w_wmax
    # Match the C++ metadata check at decimal equality (e.g. 0.1+0.2 vs 0.3).
    support_lower_limit = summed_wmax * (1 - 64 * np.finfo(float).eps)
    if (not np.isfinite(summed_wmax) or sigma_wmax < support_lower_limit
            or not np.isfinite(beta * sigma_wmax)):
        raise ValueError("require sigma_wmax >= g_wmax + w_wmax and finite beta*sigma_wmax")
    import sparse_ir

    boson = sparse_ir.FiniteTempBasis("B", beta=beta, wmax=w_wmax, eps=tolerance)
    fermion = sparse_ir.FiniteTempBasis("F", beta=beta, wmax=sigma_wmax, eps=tolerance)
    boson_sampling = sparse_ir.MatsubaraSampling(boson, positive_only=False)
    fermion_sampling = sparse_ir.MatsubaraSampling(fermion, positive_only=False)
    time_sampling = sparse_ir.TauSampling(fermion, use_positive_taus=True)
    tau = np.asarray(time_sampling.sampling_points)
    wb = np.asarray(boson_sampling.sampling_points, dtype=np.int64)
    wf = np.asarray(fermion_sampling.sampling_points, dtype=np.int64)
    if (len(wb) == 0 or len(wf) == 0 or np.any(wb % 2) or np.any(wf % 2 != 1)
            or not np.all(np.diff(wb) > 0) or not np.all(np.diff(wf) > 0)
            or not np.array_equal(wb, -wb[::-1]) or not np.array_equal(wf, -wf[::-1])
            or 0 not in wb):
        raise ValueError("sparse-ir returned invalid full signed Matsubara labels")
    # Floor division handles the negative odd labels; e.g. wn=-1 means n=-1.
    bm, fn = wb // 2, (wf - 1) // 2
    limit = np.iinfo(np.int32)
    if any(np.min(labels) < limit.min or np.max(labels) > limit.max for labels in (bm, fn)):
        raise ValueError("Matsubara labels exceed the LibRPA integer range")
    B = boson.u(tau).T @ boson_sampling.fit(np.eye(len(bm), dtype=complex))
    F = fermion.uhat(wf).T @ time_sampling.fit(np.eye(len(tau)))
    B, F = np.asarray(B, dtype=complex), np.asarray(F, dtype=complex)
    if (len(tau) == 0 or not np.all(np.isfinite(tau)) or tau[0] <= 0 or tau[-1] >= beta
            or not np.all(np.diff(tau) > 0) or B.shape != (len(tau), len(bm))
            or F.shape != (len(fn), len(tau))
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(F))):
        raise ValueError("sparse-ir returned invalid GW nodes or rectangular operators")
    zero = int(np.flatnonzero(bm == 0)[0])
    static_error = float(np.max(np.abs(beta * B[:, zero] - 1)))
    if static_error > max(1e-8, 100 * tolerance):
        raise ValueError("bosonic IR basis does not resolve the isolated static mode")
    # Arbitrary static anomaly: A delta_m0 maps exactly to A/beta, with no 1/2.
    B[:, zero] = 1 / beta
    return dict(tau=tau, bosonic_indices=bm, fermionic_indices=fn, B=B, F=F,
                uncorrected_static_mode_relative_error=static_error)


def generate_grid(output, beta, g_wmax, w_wmax, sigma_wmax, tolerance):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    grid = build_grid(beta, g_wmax, w_wmax, sigma_wmax, tolerance)
    version = importlib.metadata.version("sparse-ir")
    tau, bm, fn = (grid[key] for key in ("tau", "bosonic_indices", "fermionic_indices"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        handle.write(f"LIBRPA_THERMAL_GW_V1\nsparse-ir {version}\nHa Ha^-1\n")
        handle.write("B exp_minus inverse F exp_plus integral\n")
        handle.write(" ".join(f"{value:.17e}" for value in
                              (beta, g_wmax, w_wmax, sigma_wmax, tolerance)) + "\n")
        handle.write(f"{len(tau)} {len(bm)} {len(fn)}\n")
        np.savetxt(handle, tau, fmt="%.17e")
        np.savetxt(handle, bm, fmt="%d")
        np.savetxt(handle, fn, fmt="%d")
        for key in ("B", "F"):
            flat = grid[key].ravel(order="C")
            np.savetxt(handle, np.c_[flat.real, flat.imag], fmt="%.17e")
    return dict(output=str(output), method="sparse-ir", version=version,
                beta_ha_inverse=beta, g_wmax_ha=g_wmax, w_wmax_ha=w_wmax,
                sigma_wmax_ha=sigma_wmax, tolerance=tolerance,
                ntau=len(tau), nboson=len(bm), nfermion=len(fn),
                uncorrected_static_mode_relative_error=grid["uncorrected_static_mode_relative_error"],
                scope="complex correlation GW operators; spectral support and material observables unvalidated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beta", type=float, required=True, help="inverse thermal energy in Ha^-1")
    parser.add_argument("--g-wmax", type=float, required=True, help="bound on abs(epsilon-mu) in Ha")
    parser.add_argument("--w-wmax", type=float, required=True, help="screened Wc spectral bound in Ha")
    parser.add_argument("--sigma-wmax", type=float, required=True, help="Sigma bound >= g-wmax+w-wmax in Ha")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    print(json.dumps(generate_grid(args.output, args.beta, args.g_wmax, args.w_wmax,
                                   args.sigma_wmax, args.tolerance), indent=2))
