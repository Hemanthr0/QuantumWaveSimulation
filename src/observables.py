"""Physical observables for a 1D wavefunction on a uniform grid."""
import numpy as np


def _integrate(values, dx):
    # Composite trapezoid rule; avoids deprecated np.trapz in newer NumPy.
    return np.trapezoid(values, dx=dx)


def probability_norm(psi, dx):
    return float(np.real(_integrate(np.abs(psi)**2, dx)))


def expectation_x(x, psi):
    dx = float(x[1] - x[0])
    return float(np.real(_integrate(x * np.abs(psi)**2, dx)))


def expectation_x2(x, psi):
    dx = float(x[1] - x[0])
    return float(np.real(_integrate(x**2 * np.abs(psi)**2, dx)))


def expectation_momentum(psi, dx, hbar=1.0):
    dpsi = np.gradient(psi, dx, edge_order=2)
    return float(np.real(_integrate(np.conjugate(psi) * (-1j*hbar*dpsi), dx)))


def expectation_p2(psi, dx, hbar=1.0):
    d2psi = np.gradient(np.gradient(psi, dx, edge_order=2), dx, edge_order=2)
    return float(np.real(_integrate(np.conjugate(psi) * (-hbar**2*d2psi), dx)))


def uncertainties(x, psi, hbar=1.0):
    dx = float(x[1] - x[0])
    norm = probability_norm(psi, dx)
    if norm <= 0:
        raise ValueError("Wavefunction has zero probability norm.")
    x_mean = expectation_x(x, psi) / norm
    x2_mean = expectation_x2(x, psi) / norm
    p_mean = expectation_momentum(psi, dx, hbar) / norm
    p2_mean = expectation_p2(psi, dx, hbar) / norm
    delta_x = np.sqrt(max(0.0, x2_mean - x_mean**2))
    delta_p = np.sqrt(max(0.0, p2_mean - p_mean**2))
    return {
        "x_mean": x_mean, "p_mean": p_mean,
        "delta_x": float(delta_x), "delta_p": float(delta_p),
        "uncertainty_product": float(delta_x * delta_p),
        "heisenberg_limit": hbar / 2.0,
    }


def energy_expectation(psi, V, dx, hbar=1.0, mass=1.0):
    d2psi = np.gradient(np.gradient(psi, dx, edge_order=2), dx, edge_order=2)
    kinetic_density = np.conjugate(psi) * (-hbar**2/(2*mass) * d2psi)
    potential_density = V * np.abs(psi)**2
    return float(np.real(_integrate(kinetic_density + potential_density, dx)))


def probability_current(psi, dx, hbar=1.0, mass=1.0):
    """j = (hbar/m) Im(psi* dpsi/dx)."""
    dpsi = np.gradient(psi, dx, edge_order=2)
    return (hbar / mass) * np.imag(np.conjugate(psi) * dpsi)


def region_probabilities(x, psi, left_edge, right_edge):
    """Return left, barrier-region and right probability fractions."""
    dx = float(x[1] - x[0])
    density = np.abs(psi)**2
    left = _integrate(density[x < left_edge], dx) if np.any(x < left_edge) else 0.0
    barrier = _integrate(density[(x >= left_edge) & (x <= right_edge)], dx)
    right = _integrate(density[x > right_edge], dx) if np.any(x > right_edge) else 0.0
    total = left + barrier + right
    if total > 0:
        left, barrier, right = left/total, barrier/total, right/total
    return {"left": left, "barrier": barrier, "right": right}
