"""Crank–Nicolson time propagation for the 1D TDSE.

Uses dimensionless units hbar = m = 1 and fixed (Dirichlet) boundary
conditions psi=0 at the two ends of the spatial grid.
"""
import numpy as np


def _solve_tridiagonal(lower, diagonal, upper, rhs):
    """Thomas algorithm for a complex tridiagonal linear system."""
    n = len(diagonal)
    c = np.asarray(upper, dtype=complex).copy()
    d = np.asarray(diagonal, dtype=complex).copy()
    b = np.asarray(lower, dtype=complex).copy()
    r = np.asarray(rhs, dtype=complex).copy()

    for i in range(1, n):
        if abs(d[i - 1]) < 1e-14:
            raise np.linalg.LinAlgError("Near-zero pivot in tridiagonal solver.")
        factor = b[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        r[i] -= factor * r[i - 1]

    if abs(d[-1]) < 1e-14:
        raise np.linalg.LinAlgError("Near-zero final pivot in tridiagonal solver.")
    solution = np.empty(n, dtype=complex)
    solution[-1] = r[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = (r[i] - c[i] * solution[i + 1]) / d[i]
    return solution


def crank_nicolson(psi, V, dx, dt, hbar=1.0, mass=1.0):
    """Advance psi by one Crank–Nicolson time step.

    The endpoints are held at zero. The interior points are propagated
    without artificial renormalization, allowing norm conservation to be
    checked as a numerical diagnostic.
    """
    psi = np.asarray(psi, dtype=complex)
    V = np.asarray(V, dtype=float)
    if psi.ndim != 1 or V.shape != psi.shape or len(psi) < 3:
        raise ValueError("psi and V must be matching 1D arrays of length >= 3.")
    if dx <= 0 or dt <= 0 or hbar <= 0 or mass <= 0:
        raise ValueError("dx, dt, hbar and mass must be positive.")

    # H = -hbar^2/(2m) d^2/dx^2 + V, central finite difference.
    c = hbar**2 / (2.0 * mass * dx**2)
    v = V[1:-1]
    h_diag = 2.0 * c + v
    h_off = -c

    alpha = 1j * dt / (2.0 * hbar)
    a_diag = 1.0 + alpha * h_diag
    a_off = np.full(len(psi) - 3, alpha * h_off, dtype=complex)
    b_diag = 1.0 - alpha * h_diag
    b_off = np.full(len(psi) - 3, -alpha * h_off, dtype=complex)

    interior = psi[1:-1]
    rhs = b_diag * interior
    rhs[1:] += b_off * interior[:-1]
    rhs[:-1] += b_off * interior[1:]

    new_psi = np.zeros_like(psi)
    new_psi[1:-1] = _solve_tridiagonal(a_off, a_diag, a_off, rhs)
    return new_psi
