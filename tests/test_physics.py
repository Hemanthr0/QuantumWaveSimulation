import unittest
import numpy as np

from src.wavepacket import gaussian_wavepacket
from src.potentials import free_particle, barrier_potential
from src.solver import crank_nicolson
from src.observables import probability_norm, energy_expectation, uncertainties


class TestQuantumWaveSimulation(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-12.0, 12.0, 601)
        self.dx = self.x[1] - self.x[0]
        self.dt = 0.0005

    def test_initial_wavepacket_is_normalized(self):
        psi = gaussian_wavepacket(self.x, x0=-3, k0=2, sigma=0.7)
        self.assertAlmostEqual(probability_norm(psi, self.dx), 1.0, places=6)

    def test_probability_is_conserved(self):
        psi = gaussian_wavepacket(self.x, x0=-3, k0=2, sigma=0.7)
        V = free_particle(self.x)
        norm0 = probability_norm(psi, self.dx)
        for _ in range(100):
            psi = crank_nicolson(psi, V, self.dx, self.dt)
        self.assertAlmostEqual(probability_norm(psi, self.dx), norm0, places=5)

    def test_energy_is_approximately_conserved(self):
        psi = gaussian_wavepacket(self.x, x0=-3, k0=2, sigma=0.7)
        V = free_particle(self.x)
        e0 = energy_expectation(psi, V, self.dx)
        for _ in range(100):
            psi = crank_nicolson(psi, V, self.dx, self.dt)
        e1 = energy_expectation(psi, V, self.dx)
        self.assertLess(abs(e1-e0), 2e-3)

    def test_uncertainty_product_is_positive(self):
        psi = gaussian_wavepacket(self.x, x0=0, k0=1, sigma=0.7)
        values = uncertainties(self.x, psi)
        self.assertGreater(values["delta_x"], 0)
        self.assertGreater(values["delta_p"], 0)
        self.assertGreaterEqual(values["uncertainty_product"], 0.49)

    def test_barrier_shape(self):
        V = barrier_potential(self.x, V0=3.0, width=1.0)
        self.assertAlmostEqual(float(V.max()), 3.0)
        self.assertEqual(float(V[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
