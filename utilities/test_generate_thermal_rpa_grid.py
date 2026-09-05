"""Independent infinite Matsubara-sum and complex-response checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


class ThermalRpaGridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("generate_thermal_rpa_grid.py")
        if not path.exists():
            raise AssertionError("external sparse RPA grid generator is not implemented")
        spec = importlib.util.spec_from_file_location("thermal_rpa_grid", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_infinite_trace_log_sum_and_static_mode(self):
        for beta in (8.0, 315.7750248494972):
            grid = self.module.build_grid(beta, 15.0, 30.0, 1e-10)
            indices, weights = grid["indices"], grid["weights"]
            self.assertLess(len(indices), 64)
            self.assertTrue(np.all(np.diff(indices) > 0))
            self.assertGreater(indices[-1], len(indices))
            self.assertEqual(weights[0], 0.5 / beta)
            nu = 2 * np.pi * indices / beta
            for energy, coupling in ((0.02, 0.01), (0.5, 0.4), (5.0, 2.0)):
                pole = np.sqrt(energy**2 + coupling)
                x = coupling / (nu**2 + energy**2)
                trace_log = np.log1p(x) - x
                logsinh = lambda z: z + np.log1p(-np.exp(-2*z)) - np.log(2)
                exact = ((logsinh(beta*pole/2) - logsinh(beta*energy/2))/beta
                         - coupling/(4*energy*np.tanh(beta*energy/2)))
                self.assertAlmostEqual(weights @ trace_log, exact, delta=2e-9)
                trace_log[0] += 7.25
                self.assertAlmostEqual(weights @ trace_log, exact + 7.25/(2*beta), delta=2e-9)

    def test_complex_non_even_response_and_constant_mode(self):
        beta = 8.0
        grid = self.module.build_grid(beta, 2.0, 4.0, 1e-10)
        tau, transform, indices = grid["tau"], grid["transform"], grid["indices"]
        np.testing.assert_allclose(transform.sum(axis=1), beta*(indices == 0), atol=2e-8)
        for gap in (-1.9, -0.001, 0.001, 1.9):
            amplitude = 0.3 + 0.7j
            value_tau = amplitude * np.exp(-gap*tau) / np.expm1(-beta*gap)
            expected = amplitude / (1j*2*np.pi*indices/beta - gap)
            np.testing.assert_allclose(transform @ value_tau, expected, rtol=2e-8, atol=2e-8)

    def test_na_temperature_production_tolerance(self):
        beta = 315.7750248494972
        grid = self.module.build_grid(beta, 15.0, 30.0, 1e-8)
        nu = 2*np.pi*grid["indices"]/beta
        for energy, coupling in ((0.02, 0.01), (0.5, 0.4), (5.0, 2.0)):
            pole = np.sqrt(energy**2 + coupling)
            x = coupling / (nu**2 + energy**2)
            logsinh = lambda z: z + np.log1p(-np.exp(-2*z)) - np.log(2)
            exact = ((logsinh(beta*pole/2)-logsinh(beta*energy/2))/beta
                     - coupling/(4*energy*np.tanh(beta*energy/2)))
            self.assertAlmostEqual(grid["weights"] @ (np.log1p(x)-x), exact, delta=2e-7)

    def test_production_time_to_response_to_strong_trace_log(self):
        beta = 315.7750248494972
        grid = self.module.build_grid(beta, 15.0, 30.0, 1e-8)
        tau, transform = grid["tau"], grid["transform"]
        for energy, coupling in ((0.002, 0.01), (0.5, 800.0), (14.0, 300.0)):
            pole = np.sqrt(energy**2 + coupling)
            chi_tau = (-coupling/(2*energy) * (np.exp(-energy*tau) + np.exp(-energy*(beta-tau)))
                       / (-np.expm1(-beta*energy)))
            chi_nu = transform @ chi_tau
            integrand = np.log1p(-chi_nu) + chi_nu
            logsinh = lambda z: z + np.log1p(-np.exp(-2*z)) - np.log(2)
            exact = ((logsinh(beta*pole/2)-logsinh(beta*energy/2))/beta
                     - coupling/(4*energy*np.tanh(beta*energy/2)))
            self.assertLess(abs(grid["weights"] @ integrand-exact), 2e-7)
            for static in (0.3, 100.0):
                with_static = transform @ (chi_tau-static/beta)
                result = grid["weights"] @ (np.log1p(-with_static)+with_static)
                x0 = coupling/energy**2
                correction = (np.log1p(x0+static)-np.log1p(x0)-static)/(2*beta)
                self.assertLess(abs(result-exact-correction), 2e-7)

    def test_file_contract_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"grid.dat"
            report = self.module.generate_grid(path, 8.0, 2.0, 4.0, 1e-10)
            tokens = path.read_text(encoding="ascii").split()
            self.assertEqual(tokens[0], "LIBRPA_THERMAL_RPA_V1")
            self.assertEqual(tokens[1], "sparse-ir")
            self.assertEqual(tokens[3:6], ["B", "exp_plus", "integral"])
            ntau, nfreq = map(int, tokens[10:12])
            self.assertEqual(nfreq, report["nfreq"])
            self.assertEqual(ntau, report["ntau"])
            self.assertEqual(len(tokens), 12 + ntau + nfreq*(2 + 2*ntau))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.generate_grid(path, 8.0, 2.0, 4.0, 1e-10)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_parameters(self):
        for parameters in ((0, 2, 4, 1e-8), (8, -1, 4, 1e-8), (8, 4, 2, 1e-8),
                           (8, 2, 4, 0), (8, 2, np.inf, 1e-8), (np.nan, 2, 4, 1e-8)):
            with self.assertRaises(ValueError):
                self.module.build_grid(*parameters)
        for tolerance in (0, -1, 1e-7, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.module.build_grid(8, 2, 4, 1e-8, tolerance)


if __name__ == "__main__":
    unittest.main()
