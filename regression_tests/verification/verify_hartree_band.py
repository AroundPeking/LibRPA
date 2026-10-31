#!/usr/bin/env python3
"""Finite SciPy/ABACUS H2 path control for mesh-interpolated Hartree feedback.

The mesh is native ABACUS data. SciPy constructs three off-mesh path states
from its saved H(R)/S(R); zero path Vxc defines a numerical integration control,
not an ABACUS NSCF result or a material benchmark. GW/QSGW runs on those states.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "utilities"))
from restore_abacus_csr_precision import read_csr

PATH_K = (0.173, -0.219, 0.5)


def prepare(work):
    case = ROOT / "regression_tests/testcases/qsgw_abacus_H2_hartree"
    with tarfile.open(case / "dataset.tar.gz") as archive:
        archive.extractall(work)
    dataset = work / "dataset"
    hr = read_csr(dataset / "hrs1_nao.csr")
    sr = read_csr(dataset / "srs1_nao.csr")
    coefficients = []
    for index, q in enumerate(PATH_K, 1):
        def fourier(blocks):
            matrix = sum(np.exp(2j * np.pi * q * r[2]) * m for r, m in blocks.items())
            return np.triu(matrix) + np.triu(matrix, 1).conj().T
        energy, c = eigh(fourier(hr) / 2, fourier(sr))
        coefficients.append(c)
        (dataset / f"band_KS_eigenvector_k_{index:05d}.txt").write_bytes(
            np.asarray(c.T, dtype="=c16").tobytes())
        (dataset / f"band_KS_eigenvalue_k_{index:05d}.txt").write_text("".join(
            f"{band+1} 0 {2 if band == 0 else 0} {value:.17e} 0\n"
            for band, value in enumerate(energy)))
        (dataset / f"band_vxck{index}_nao.txt").write_text(
            "# rows 10\n# columns 10\n" + "".join(
                f"Row {i+1}\n" + " ".join("(0,0)" for _ in range(i, 10)) + "\n"
                for i in range(10)))
    (dataset / "band_kpath_info").write_text("10 10 1 3\n" + "".join(
        f"0 0 {q:.17e}\n" for q in PATH_K))
    return (case / "librpa/librpa.in").read_text(), coefficients


def run(binary, work, text, name, ranks, iterations, restart=None):
    directory = work / name
    directory.mkdir()
    overrides = {"qsgw_min_iter": "1", "qsgw_max_iter": str(iterations),
                 "option_bvk_remap": "1", "qsgw_restart_file": restart or "none"}
    lines = [line for line in text.splitlines()
             if line.split("=", 1)[0].strip() not in overrides]
    lines.extend(f"{key} = {value}" for key, value in overrides.items())
    (directory / "librpa.in").write_text("\n".join(lines) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_CBWR="2", MKL_CBWR="COMPATIBLE")
    with (directory / "run.log").open("w") as log:
        subprocess.run(["mpirun", "-np", str(ranks), str(binary)], cwd=directory,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    return directory


def hartree(directory):
    rows = list(csv.DictReader((directory / "qsgw_hartree.csv").open()))
    matrices, path = {}, {}
    for row in rows:
        key = (int(row["state_index"]), row["phase"], int(row["kpoint"]) - 1)
        i, j = int(row["row"]) - 1, int(row["column"]) - 1
        z = complex(float(row["re_Ha"]), float(row["im_Ha"]))
        if row["kind"] == "ao" and row["channel"] == "grid":
            matrices.setdefault(key, np.zeros((10, 10), complex))[i, j] = z
        if row["kind"] == "ks_diagonal" and row["channel"] == "band":
            path.setdefault(key, np.zeros(10, complex))[i] = z
    return matrices, path


def check_projection(directory, coefficients):
    matrices, path = hartree(directory)
    error, signal = 0.0, 0.0
    # For this 1x1x3 mesh and these atom coordinates, each nearest atom-pair
    # image is Rz=-1,0,+1. Its cardinal interpolant is analytic.
    for (state, phase, k), actual in path.items():
        expected = sum((1 + 2*np.cos(2*np.pi*(PATH_K[k] - mesh/3))) / 3 *
                       matrices[state, phase, mesh] for mesh in range(3))
        diagonal = np.diag(coefficients[k].conj().T @ expected @ coefficients[k])
        error = max(error, float(np.max(np.abs(diagonal - actual))))
        signal = max(signal, float(np.max(np.abs(actual))))
    if len(path) != 12 or signal < 1e-5 or error > 2e-12:
        raise AssertionError(f"Hartree path projection: count={len(path)}, signal={signal}, error={error}")
    return error, signal


def matrix_trace(directory, selected_channel):
    matrices = {}
    for line in (directory / "qsgw_matrices.dat").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        state, channel, component, spin, k = (int(fields[0]), int(fields[1]),
                                             fields[2], int(fields[3]), int(fields[4]))
        if channel != selected_channel or int(fields[5]) != -1:
            continue
        block = matrices.setdefault((state, component, spin, k), np.zeros((10, 10), complex))
        block[int(fields[7]), int(fields[8])] = complex(float(fields[9]), float(fields[10]))
    return matrices


def check_feedback(directory):
    matrices = matrix_trace(directory, 1)
    assembly, subtraction, signal, spectrum = 0.0, 0.0, 0.0, 0.0
    eigenvalues = np.loadtxt(directory / "qsgw_eigenvalues.dat")
    for iteration in (1, 2):
        for k in range(3):
            h0, vxc = matrices[0, "h0", 0, k], matrices[0, "vxc_dft", 0, k]
            delta = matrices[iteration - 1, "delta_vh_entering", 0, k]
            expected = h0 - vxc + matrices[iteration, "exx", 0, k] + matrices[iteration, "vc", 0, k] + delta
            assembly = max(assembly, float(np.max(np.abs(matrices[iteration, "raw_h", 0, k] - expected))))
            rows = eigenvalues[(eigenvalues[:, 0] == iteration) & (eigenvalues[:, 1] == 1)
                               & (eigenvalues[:, 3] == k)]
            spectrum = max(spectrum, float(np.max(np.abs(np.linalg.eigvalsh(expected)*27.211386245988 - rows[:, -1]))))
            signal = max(signal, float(np.max(np.abs(delta))))
    for row in csv.DictReader((directory / "qsgw_qp.csv").open()):
        if row["channel"] != "band":
            continue
        state, k, b = int(row["state_index"]), int(row["kpoint"])-1, int(row["band"])-1
        h0, vxc = matrices[0, "h0", 0, k], matrices[0, "vxc_dft", 0, k]
        delta = matrices[state, "delta_vh_entering", 0, k]
        h = h0 if state == 0 else matrices[state, "mixed_h", 0, k]
        u = np.eye(10) if state == 0 else matrices[state, "rotation_u", 0, k]
        expected = np.diag(u.conj().T @ (h - h0 + vxc - delta) @ u)[b].real
        subtraction = max(subtraction, abs(expected - float(row["vstatic_eV"])/27.211386245988))
    if signal < 1e-5 or assembly > 2e-12 or subtraction > 2e-12 or spectrum > 1e-8:
        raise AssertionError((signal, assembly, subtraction, spectrum))
    return {"band_assembly_Ha": assembly, "band_QP_subtraction_Ha": subtraction,
            "band_spectrum_eV": spectrum, "nonzero_feedback_Ha": signal}


def check_hr(directory):
    """Check serialized HR against independently assembled grid/path AO operators.

    In this 1x1x3 fixture the HR Fourier images are -1,0,+1. Comparing the complete
    native-baseline interpolant retains nonzero Hartree and EXX terms, rather
    than checking only eigenspectra or the exporter's own roundtrip residual.
    """
    matrices = matrix_trace(directory, 0)
    native = read_csr(directory.parent / "dataset/hrs1_nao.csr")
    overlap = read_csr(directory.parent / "dataset/srs1_nao.csr")

    def fourier(blocks, q, scale=1):
        return sum(np.exp(2j*np.pi*q*r[2]) * m * scale for r, m in blocks.items())

    duals = []
    for k in range(3):
        c = matrices[0, "wfc_spinor0", 0, k].T
        s = fourier(overlap, k/3)
        s = np.triu(s) + np.triu(s, 1).conj().T
        duals.append(s @ c @ np.linalg.inv(c.conj().T @ s @ c))
    error, signal, assembly = 0.0, 0.0, 0.0
    for iteration in (0, 1, 2):
        exported = read_csr(directory / f"hrs1_nao_qsgw_iter_{iteration:04d}.csr")
        corrections, hartree_ao = [], []
        for k, dual in enumerate(duals):
            h0 = matrices[0, "h0", 0, k]
            if iteration == 0:
                expected, delta = h0, np.zeros((10, 10), complex)
            else:
                delta = matrices[iteration-1, "delta_vh_entering", 0, k]
                expected = (h0 - matrices[0, "vxc_dft", 0, k]
                            + matrices[iteration, "exx", 0, k]
                            + matrices[iteration, "vc", 0, k] + delta)
                assembly = max(assembly, float(np.max(abs(
                    expected - matrices[iteration, "mixed_h", 0, k]))))
            corrections.append(dual @ expected @ dual.conj().T - fourier(native, k/3, .5))
            hartree_ao.append(dual @ delta @ dual.conj().T)
        for q in (0, 1/3, 2/3, *PATH_K):
            weights = [(1+2*np.cos(2*np.pi*(q-k/3)))/3 for k in range(3)]
            expected = fourier(native, q, .5) + sum(w*m for w, m in zip(weights, corrections))
            value = fourier(exported, q, .5)
            error = max(error, float(np.max(abs(value-expected))))
            # Omitting Hartree must fail at the second update.
            signal = max(signal, float(np.max(abs(sum(
                w*m for w, m in zip(weights, hartree_ao))))))
    if error > 2e-11 or assembly > 2e-12 or signal < 1e-5:
        raise AssertionError(f"Hartree/EXX HR: error={error}, assembly={assembly}, signal={signal}")
    return {"HR_grid_path_Ha": error, "HR_assembly_Ha": assembly,
            "HR_nonzero_Hartree_Ha": signal}


def numerical_rows(directory, filename, state=None):
    if filename.endswith(".csv"):
        data = list(csv.DictReader((directory / filename).open()))
        if state is not None:
            data = [r for r in data if int(r["state_index"]) == state]
        columns = ("e0_eV", "vstatic_eV", "sigma_x_eV", "re_sigma_c_effective_eV", "eqp_eV")
        if not data or not any(r["channel"] == "band" for r in data):
            raise AssertionError("Missing band QP results")
        if any(int(r["solver_info"]) != 0 for r in data):
            raise AssertionError("QP root failure")
        result = np.array([[float(r[c]) for c in columns] for r in data])
    else:
        result = np.loadtxt(directory / filename)
        if state is not None:
            result = result[result[:, 0] == state]
    if not result.size or not np.isfinite(result).all():
        raise AssertionError("Missing or nonfinite trajectory")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    text, coefficients = prepare(work)
    serial = run(args.binary.resolve(), work, text, "serial", 1, 2)
    parallel = run(args.binary.resolve(), work, text, "parallel", args.ranks, 2)
    first = run(args.binary.resolve(), work, text, "first", args.ranks, 1)
    resumed = run(args.binary.resolve(), work, text, "resumed", args.ranks, 2,
                  str(first / "qsgw_checkpoint.bin"))
    error, signal = check_projection(serial, coefficients)
    errors = {"band_projection_Ha": error, "band_signal_Ha": signal,
              **check_feedback(serial), **check_hr(serial)}
    for name, directory, state in (("MPI", parallel, None), ("restart", resumed, 2)):
        for filename in ("qsgw_eigenvalues.dat", "qsgw_qp.csv"):
            # A resumed update enters state 1 and completes state 2.
            compare_state = 1 if filename.endswith(".csv") and state is not None else state
            a = numerical_rows(serial, filename, compare_state)
            b = numerical_rows(directory, filename, compare_state)
            delta = float(np.max(np.abs(a - b)))
            errors[f"{name}_{filename}_eV"] = delta
            if delta > 1e-7:
                raise AssertionError(errors)
    (work / "comparison.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
