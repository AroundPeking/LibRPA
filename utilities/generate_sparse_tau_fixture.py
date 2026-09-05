#!/usr/bin/env python3
"""Generate a pinned external-library fixture for the thermal transform tests."""

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import sparse_ir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    beta, wmax, tolerance, nfreq = 8.0, 2.0, 1e-10, 16
    basis = sparse_ir.FiniteTempBasis("B", beta=beta, wmax=wmax, eps=tolerance)
    sampler = sparse_ir.TauSampling(basis)
    tau = np.asarray(sampler.sampling_points)
    transform = basis.uhat(2 * np.arange(nfreq)).T @ sampler.fit(np.eye(len(tau)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("LIBRPA_SPARSE_TAU_TEST_V1\n")
        handle.write(f"sparse-ir {importlib.metadata.version('sparse-ir')}\n")
        handle.write(f"{beta:.17e} {wmax:.17e} {tolerance:.17e}\n")
        handle.write(f"{len(tau)} {nfreq}\n")
        np.savetxt(handle, tau, fmt="%.17e")
        np.savetxt(handle, np.c_[transform.real.ravel(), transform.imag.ravel()], fmt="%.17e")
    print(f"{args.output}: {len(tau)} time nodes, {nfreq} consecutive bosonic frequencies")


if __name__ == "__main__":
    main()
