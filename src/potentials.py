"""One-dimensional potential-energy profiles in dimensionless units."""
import numpy as np


def free_particle(x):
    return np.zeros_like(x, dtype=float)


def harmonic_oscillator(x, k=0.5):
    if k < 0:
        raise ValueError("The spring constant k must be non-negative.")
    return 0.5 * k * x**2


def double_well(x, a=0.04, b=1.0):
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive.")
    return a * (x**2 - b)**2


def barrier_potential(x, V0=1.0, width=0.5):
    if V0 < 0 or width <= 0:
        raise ValueError("V0 must be non-negative and width must be positive.")
    return np.where(np.abs(x) <= width / 2, V0, 0.0)


def finite_well(x, depth=1.0, width=2.0):
    """A finite square well of depth `depth` below zero outside."""
    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be positive.")
    return np.where(np.abs(x) <= width / 2, -depth, 0.0)
