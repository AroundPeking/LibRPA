#!/usr/bin/env python3
"""Export sparse bosonic response nodes and infinite RPA trace-log sum weights.

The weights apply to the real, even RPA scalar integrand, not to individual
complex response matrix elements. The latter use an independent complex
time-to-frequency transform. Screened poles can exceed the bare excitation
range, so rpa_wmax is a separate convergence parameter.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def build_grid(beta, wmax, rpa_wmax, tolerance, rpa_tolerance=None):
    if (not np.all(np.isfinite([beta, wmax, rpa_wmax, tolerance]))
            or beta <= 0 or wmax <= 0 or rpa_wmax < wmax or not 0 < tolerance < 1):
        raise ValueError("require beta>0, rpa_wmax>=wmax>0, tolerance in (0,1), all finite")
    if rpa_tolerance is None:
        rpa_tolerance = min(tolerance, 1e-10)
    if not np.isfinite(rpa_tolerance) or not 0 < rpa_tolerance <= tolerance:
        raise ValueError("require finite 0 < rpa_tolerance <= response tolerance")
    import sparse_ir

    basis = sparse_ir.FiniteTempBasis("B", beta=beta, wmax=rpa_wmax, eps=rpa_tolerance)
    sampling = sparse_ir.MatsubaraSampling(basis, positive_only=False)
    signed = np.asarray(sampling.sampling_points, dtype=np.int64) // 2
    fit = sampling.fit(np.eye(len(signed), dtype=complex))
    # (1/2beta) sum_n F(i nu_n) = [F(0+) + F(beta-)]/4.
    signed_weights = 0.25 * (basis.u(0.0) + basis.u(beta)) @ fit
    indices = np.unique(np.abs(signed))
    if (indices[0] != 0 or indices[-1] > np.iinfo(np.int32).max
            or not np.array_equal(signed, -signed[::-1])):
        raise ValueError("sparse-ir returned invalid signed bosonic sampling points")
    folded = np.array([signed_weights[np.abs(signed) == index].sum() for index in indices])
    if np.max(np.abs(folded.imag)) > max(1e-10, 100*rpa_tolerance)*max(1.0, np.max(abs(folded))):
        raise ValueError("RPA scalar sum weights have a significant imaginary component")
    weights = folded.real.copy()
    static_relative_error = abs(2*beta*weights[0] - 1)
    if static_relative_error > max(1e-8, 100*rpa_tolerance):
        raise ValueError("IR sum does not resolve the isolated static Matsubara mode")
    # Preserve an arbitrary static anomaly exactly; record the small IR correction.
    weights[0] = 0.5 / beta

    response_basis = sparse_ir.FiniteTempBasis("B", beta=beta, wmax=wmax, eps=tolerance)
    time_sampling = sparse_ir.TauSampling(response_basis)
    tau = np.asarray(time_sampling.sampling_points)
    transform = response_basis.uhat(2*indices).T @ time_sampling.fit(np.eye(len(tau)))
    order = np.argsort(tau)
    tau, transform = tau[order], np.asarray(transform)[:, order]
    if (not np.all(np.isfinite(weights)) or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(tau)) or not np.all(np.diff(tau) > 0)
            or tau[0] <= 0 or tau[-1] >= beta):
        raise ValueError("external library returned invalid RPA weights or transform")
    constant_error = float(np.max(abs(transform.sum(axis=1)/beta - (indices == 0))))
    if constant_error > max(1e-10, 10*tolerance):
        raise ValueError("response transform does not integrate the constant mode correctly")
    return dict(tau=tau, indices=indices, weights=weights, transform=transform,
                rpa_tolerance=rpa_tolerance,
                constant_mode_relative_error=constant_error,
                uncorrected_static_weight_relative_error=float(static_relative_error))


def generate_grid(output, beta, wmax, rpa_wmax, tolerance, rpa_tolerance=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    grid = build_grid(beta, wmax, rpa_wmax, tolerance, rpa_tolerance)
    tau, indices, weights, transform = (grid[key] for key in ("tau", "indices", "weights", "transform"))
    version = importlib.metadata.version("sparse-ir")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        handle.write(f"LIBRPA_THERMAL_RPA_V1\nsparse-ir {version}\nB exp_plus integral\n")
        handle.write(f"{beta:.17e} {wmax:.17e} {rpa_wmax:.17e} {tolerance:.17e}\n")
        handle.write(f"{len(tau)} {len(indices)}\n")
        np.savetxt(handle, tau, fmt="%.17e")
        for index, weight, row in zip(indices, weights, transform):
            handle.write(f"{index} {weight:.17e}\n")
            np.savetxt(handle, np.c_[row.real, row.imag], fmt="%.17e")
    return dict(output=str(output), method="sparse-ir", version=version,
                beta_ha_inverse=beta, wmax_ha=wmax, rpa_wmax_ha=rpa_wmax, tolerance=tolerance,
                rpa_tolerance=grid["rpa_tolerance"],
                ntau=len(tau), nfreq=len(indices), maximum_matsubara_index=int(indices[-1]),
                constant_mode_relative_error=grid["constant_mode_relative_error"],
                uncorrected_static_weight_relative_error=grid["uncorrected_static_weight_relative_error"],
                scope="sparse infinite RPA sum; spectral bounds and observable convergence must be tested")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beta", type=float, required=True, help="inverse thermal energy in Ha^-1")
    parser.add_argument("--wmax", type=float, required=True, help="bare excitation spectral bound in Ha")
    parser.add_argument("--rpa-wmax", type=float, required=True, help="screened trace-log spectral bound in Ha")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--rpa-tolerance", type=float,
                        help="trace-log basis tolerance; default min(response tolerance,1e-10)")
    args = parser.parse_args()
    print(json.dumps(generate_grid(args.output, args.beta, args.wmax, args.rpa_wmax,
                                   args.tolerance, args.rpa_tolerance), indent=2))
