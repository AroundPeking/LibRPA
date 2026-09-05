"""Independent poles and signed convolution checks for sparse GW operators."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import expit


def w_frequency(nu, a, b, residue):
    return residue * (1 / (1j * nu - a) - 1 / (1j * nu + b))


def w_time(tau, beta, a, b, residue):
    return -residue * (np.exp(-a * tau) / -np.expm1(-beta * a)
                       + np.exp(-b * (beta - tau)) / -np.expm1(-beta * b))


def g_time(tau, beta, xi):
    return np.exp(-xi * tau - np.logaddexp(0, -beta * xi))


def sigma_frequency(omega, beta, xi, a, b, residue):
    # Closed form from residues, independent of either IR fit.
    f = expit(-beta * xi)
    na = np.exp(-beta * a) / -np.expm1(-beta * a)
    nb = np.exp(-beta * b) / -np.expm1(-beta * b)
    return residue * ((1 + na - f) / (1j * omega - xi - a)
                      + (nb + f) / (1j * omega - xi + b))


class ThermalGWGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("generate_thermal_gw_grid.py")
        if not path.exists():
            raise AssertionError("external sparse GW operator generator is not implemented")
        spec = importlib.util.spec_from_file_location("thermal_gw_grid", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.small = cls.module.build_grid(8.0, 1.0, 2.0, 3.0, 1e-10)

    def test_rectangular_full_signed_operators(self):
        grid = self.small
        tau, bm, fn = (grid[key] for key in ("tau", "bosonic_indices", "fermionic_indices"))
        self.assertTrue(np.all(np.diff(tau) > 0))
        self.assertGreater(tau[0], 0)
        self.assertLess(tau[-1], 8)
        np.testing.assert_array_equal(bm, -bm[::-1])
        np.testing.assert_array_equal(fn, -fn[::-1] - 1)
        self.assertIn(0, bm)
        self.assertEqual(grid["B"].shape, (len(tau), len(bm)))
        self.assertEqual(grid["F"].shape, (len(fn), len(tau)))
        self.assertNotEqual(len(bm), len(tau))
        self.assertTrue(np.all(np.isfinite(grid["B"])))
        self.assertTrue(np.all(np.isfinite(grid["F"])))

    def test_static_mode_is_not_rpa_half_weight(self):
        grid, beta = self.small, 8.0
        bm, fn = grid["bosonic_indices"], grid["fermionic_indices"]
        amplitude = 0.7 + 0.4j
        wt = grid["B"] @ (amplitude * (bm == 0))
        np.testing.assert_allclose(wt, amplitude / beta, atol=1e-16, rtol=1e-15)
        for xi in (-0.98, -0.0001, 0.0, 0.0001, 0.98):
            actual = grid["F"] @ (g_time(grid["tau"], beta, xi) * wt)
            expected = -amplitude / (beta * (1j * (2 * fn + 1) * np.pi / beta - xi))
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)

    def check_poles(self, grid, beta, energies, pole_pairs, tolerance):
        nu = 2 * np.pi * grid["bosonic_indices"] / beta
        omega = (2 * grid["fermionic_indices"] + 1) * np.pi / beta
        for a, b in pole_pairs:
            residue = 0.4 - 0.3j
            wt = grid["B"] @ w_frequency(nu, a, b, residue)
            np.testing.assert_allclose(wt, w_time(grid["tau"], beta, a, b, residue),
                                       atol=tolerance, rtol=tolerance)
            for xi in energies:
                sigma = grid["F"] @ (g_time(grid["tau"], beta, xi) * wt)
                np.testing.assert_allclose(
                    sigma, sigma_frequency(omega, beta, xi, a, b, residue),
                    atol=tolerance, rtol=tolerance)

    def test_complex_asymmetric_poles_and_spectral_edges(self):
        self.check_poles(self.small, 8.0, (-0.99, -0.01, 0, 0.01, 0.99),
                         ((0.7, 1.1), (0.002, 0.003), (1.98, 1.99)), 2e-8)

    def test_na_temperature_common_sigma_grid(self):
        beta = 315.7750248494972
        grid = self.module.build_grid(beta, 15.0, 30.0, 45.0, 1e-10)
        self.assertLess(len(grid["tau"]), 100)
        self.assertLess(len(grid["bosonic_indices"]), 100)
        self.assertLess(len(grid["fermionic_indices"]), 100)
        self.check_poles(grid, beta, (-14.9, -0.001, 0, 0.001, 14.9),
                         ((0.002, 0.003), (0.7, 1.1), (29.8, 29.9)), 2e-8)

    def test_direct_uniform_matsubara_convolution(self):
        grid, beta, xi, a, b, residue = self.small, 8.0, -0.27, 0.7, 1.1, 0.4 - 0.3j
        nu = 2 * np.pi * grid["bosonic_indices"] / beta
        wt = grid["B"] @ w_frequency(nu, a, b, residue)
        sigma = grid["F"] @ (g_time(grid["tau"], beta, xi) * wt)
        dense_nu = 2 * np.pi * np.arange(-32768, 32769) / beta
        dense_w = w_frequency(dense_nu, a, b, residue)
        for row, n in enumerate(grid["fermionic_indices"]):
            if abs(n) > 5:
                continue
            omega = (2 * n + 1) * np.pi / beta
            reference = -np.sum(dense_w / (1j * (omega - dense_nu) - xi)) / beta
            closed = sigma_frequency(omega, beta, xi, a, b, residue)
            self.assertLess(abs(reference - closed), 2e-11)
            self.assertLess(abs(sigma[row] - reference), 2e-9)

    def test_serialization_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.dat"
            report = self.module.generate_grid(path, 8.0, 1.0, 2.0, 3.0, 1e-10)
            lines = path.read_text(encoding="ascii").splitlines()
            self.assertEqual(lines[0], "LIBRPA_THERMAL_GW_V1")
            self.assertEqual(lines[1].split()[0], "sparse-ir")
            self.assertEqual(lines[2], "Ha Ha^-1")
            self.assertEqual(lines[3], "B exp_minus inverse F exp_plus integral")
            np.testing.assert_array_equal(list(map(float, lines[4].split())),
                                          [8.0, 1.0, 2.0, 3.0, 1e-10])
            nt, nb, nf = map(int, lines[5].split())
            self.assertEqual((nt, nb, nf), (report["ntau"], report["nboson"], report["nfermion"]))
            tokens = np.asarray(list(map(float, " ".join(lines[6:]).split())))
            self.assertEqual(len(tokens), nt + nb + nf + 2 * nt * (nb + nf))
            offset = 0
            for key, size in (("tau", nt), ("bosonic_indices", nb), ("fermionic_indices", nf)):
                np.testing.assert_array_equal(tokens[offset:offset + size], self.small[key])
                offset += size
            for key, rows, cols in (("B", nt, nb), ("F", nf, nt)):
                pairs = tokens[offset:offset + 2 * rows * cols].reshape(rows, cols, 2)
                np.testing.assert_array_equal(pairs[:, :, 0] + 1j * pairs[:, :, 1], self.small[key])
                offset += 2 * rows * cols
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.generate_grid(path, 8.0, 1.0, 2.0, 3.0, 1e-10)
            self.assertEqual(path.read_bytes(), original)

    def test_decimal_support_sum_boundary(self):
        grid = self.module.build_grid(8.0, 0.1, 0.2, 0.3, 1e-10)
        self.assertGreater(len(grid["tau"]), 0)
        # Permit arithmetic roundoff at equality, not a physically narrower window.
        with self.assertRaises(ValueError):
            self.module.build_grid(8.0, 0.1, 0.2, 0.3 * (1 - 1e-10), 1e-10)

    def test_invalid_spectral_support_and_parameters(self):
        valid = [8.0, 1.0, 2.0, 3.0, 1e-10]
        for pos in range(5):
            for value in (0, -1, np.nan, np.inf):
                parameters = valid.copy()
                parameters[pos] = value
                with self.subTest(pos=pos, value=value), self.assertRaises(ValueError):
                    self.module.build_grid(*parameters)
        for smax in (1.0, 2.999):
            with self.assertRaises(ValueError):
                self.module.build_grid(8, 1, 2, smax, 1e-10)
        with self.assertRaises(ValueError):
            self.module.build_grid(8, 1, 2, 3, 1)
        with self.assertRaises(ValueError):
            self.module.build_grid(1e308, 1, 2, 3, 1e-10)


if __name__ == "__main__":
    unittest.main()
