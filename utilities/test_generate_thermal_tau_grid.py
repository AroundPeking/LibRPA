import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_thermal_tau_grid import generate_grid


class ThermalGridExportTest(unittest.TestCase):
    def test_library_transforms_and_file_layout(self):
        for method in ("sparse-ir", "pydlr"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "grid.dat"
                report = generate_grid(output, method, 8.0, 2.0, 1e-8, 16)
                tokens = output.read_text().split()
                self.assertEqual(tokens[:2], ["LIBRPA_THERMAL_TAU_V1", method])
                self.assertEqual(tokens[3:6], ["B", "exp_plus", "integral"])
                ntau, nfreq = map(int, tokens[9:11])
                self.assertEqual(nfreq, 16)
                self.assertEqual(ntau, report["ntau"])
                tau = np.array(tokens[11:11 + ntau], dtype=float)
                rows = np.array(tokens[11 + ntau:], dtype=float).reshape(nfreq, 1 + 2 * ntau)
                np.testing.assert_array_equal(rows[:, 0], np.arange(nfreq))
                transform = rows[:, 1::2] + 1j * rows[:, 2::2]
                frequency = 2 * np.pi * np.arange(nfreq) / 8.0
                np.testing.assert_allclose(transform.sum(axis=1) / 8.0, frequency == 0, atol=1e-8)
                # A non-even imaginary-time response exposes the complex Fourier sign.
                gap = 0.7
                probability = (1.0 / (1.0 + np.exp(-4.0 * gap))) ** 2
                tau_response = -probability * np.exp(-gap * tau)
                exact = probability * (-np.expm1(-8.0 * gap)) / (-gap + 1j * frequency)
                np.testing.assert_allclose(transform @ tau_response, exact, rtol=2e-8, atol=1e-9)
                self.assertLess(report["constant_mode_relative_error"], 1e-8)

    def test_rejects_parameters_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.dat"
            for beta, wmax, tolerance, nfreq in [
                (0, 2, 1e-8, 16), (8, -2, 1e-8, 16), (8, 2, 0, 16),
                (8, 2, 1, 16), (8, 2, 1e-8, 0), (float("nan"), 2, 1e-8, 16),
            ]:
                with self.subTest(values=(beta, wmax, tolerance, nfreq)), self.assertRaises(ValueError):
                    generate_grid(output, "sparse-ir", beta, wmax, tolerance, nfreq)
                self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                generate_grid(output, "unknown", 8, 2, 1e-8, 16)

    def test_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.dat"
            output.write_text("retain this result")
            with self.assertRaises(FileExistsError):
                generate_grid(output, "sparse-ir", 8, 2, 1e-8, 16)
            self.assertEqual(output.read_text(), "retain this result")


if __name__ == "__main__":
    unittest.main()
