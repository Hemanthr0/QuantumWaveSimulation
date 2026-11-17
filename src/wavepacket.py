"""Initial wavefunctions and normalization helpers."""
import numpy as np


def probability_norm(psi, dx):
    """Return integral |psi|^2 dx using the grid spacing."""
    return float(np.trapezoid(np.abs(psi)**2, dx=dx))


def normalize_wavefunction(psi, dx):
    norm = probability_norm(psi, dx)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Wavefunction must have a finite, positive norm.")
    return psi / np.sqrt(norm)


def gaussian_wavepacket(x, x0=-2.0, k0=5.0, sigma=0.5):
    """Gaussian packet exp[-(x-x0)^2/(4 sigma^2)] exp(i k0 x).

    Here sigma is the initial standard deviation of position probability
    density for a sufficiently large, well-resolved domain.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    psi = np.exp(-((x - x0)**2) / (4 * sigma**2)) * np.exp(1j * k0 * x)
    dx = float(x[1] - x[0])
    return normalize_wavefunction(psi.astype(complex), dx)
