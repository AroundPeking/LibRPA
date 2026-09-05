#!/usr/bin/env python3
"""Export an external finite-temperature bosonic transform for the LibRPA driver."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def generate_grid(output, method, beta, wmax, tolerance, nfreq):
    output = Path(output)
    if (not np.all(np.isfinite([beta, wmax, tolerance])) or beta <= 0 or wmax <= 0
            or not 0 < tolerance < 1 or not isinstance(nfreq, int) or not 0 < nfreq <= 46340):
        raise ValueError("require positive finite beta/wmax, tolerance in (0,1), and valid nfreq")
    if method not in ("sparse-ir", "pydlr"):
        raise ValueError("method must be sparse-ir or pydlr")
    if output.exists():
        raise FileExistsError(output)
    indices = np.arange(nfreq)
    if method == "sparse-ir":
        import sparse_ir

        basis = sparse_ir.FiniteTempBasis("B", beta=beta, wmax=wmax, eps=tolerance)
        sampling = sparse_ir.TauSampling(basis)
        tau = np.asarray(sampling.sampling_points)
        transform = basis.uhat(2 * indices).T @ sampling.fit(np.eye(len(tau)))
    else:
        from pydlr import dlr

        basis = dlr(lamb=beta * wmax, eps=tolerance, xi=1, nmax=20000)
        tau = np.asarray(basis.get_tau(beta))
        fit = basis.dlr_from_tau(np.eye(len(tau)))
        transform = basis.eval_dlr_freq(fit, 1j * 2.0 * np.pi * indices / beta, beta, xi=1)
    order = np.argsort(tau)
    tau, transform = tau[order], np.asarray(transform)[:, order]
    if (transform.shape != (nfreq, len(tau)) or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(tau)) or not np.all(np.diff(tau) > 0)
            or tau[0] <= 0 or tau[-1] >= beta):
        raise ValueError("external library returned an invalid grid or transform")
    constant_error = float(np.max(np.abs(transform.sum(axis=1) / beta - (indices == 0))))
    if constant_error > max(1e-10, 10.0 * tolerance):
        raise ValueError(f"constant-mode normalization error {constant_error:.6e} exceeds API tolerance")
    version = importlib.metadata.version(method)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        handle.write(f"LIBRPA_THERMAL_TAU_V1\n{method} {version}\nB exp_plus integral\n")
        handle.write(f"{beta:.17e} {wmax:.17e} {tolerance:.17e}\n{len(tau)} {nfreq}\n")
        np.savetxt(handle, tau, fmt="%.17e")
        for index, row in enumerate(transform):
            handle.write(f"{index}\n")
            np.savetxt(handle, np.c_[row.real, row.imag], fmt="%.17e")
    return {
        "output": str(output), "method": method, "version": version,
        "beta_ha_inverse": beta, "wmax_ha": wmax, "tolerance": tolerance,
        "ntau": len(tau), "nfreq": nfreq,
        "constant_mode_relative_error": constant_error,
        "scope": "external time transform only; no sparse frequency sum or physical validation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("sparse-ir", "pydlr"), default="sparse-ir")
    parser.add_argument("--beta", type=float, required=True, help="inverse thermal energy in Ha^-1")
    parser.add_argument("--wmax", type=float, required=True, help="real excitation spectral bound in Ha")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--nfreq", type=int, required=True, help="consecutive nonnegative bosonic modes")
    args = parser.parse_args()
    print(json.dumps(generate_grid(args.output, args.method, args.beta, args.wmax,
                                   args.tolerance, args.nfreq), indent=2))
