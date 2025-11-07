import numpy as np

def crank_nicolson(psi, V, dx, dt):
    N = len(psi)
    hbar = 1.0
    m = 1.0
    r = 1j * hbar * dt / (4 * m * dx**2)

    diag = (1 + 2*r) + 1j * dt * V / (2*hbar)
    off = -r * np.ones(N - 1)

    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    B = np.diag((1 - 2*r) - 1j * dt * V / (2*hbar)) + np.diag(r * np.ones(N - 1), 1) + np.diag(r * np.ones(N - 1), -1)

    A[0, :] = A[-1, :] = 0
    A[0, 0] = A[-1, -1] = 1
    B[0, :] = B[-1, :] = 0
    B[0, 0] = B[-1, -1] = 1

    rhs = B @ psi
    psi_new = np.linalg.solve(A, rhs)
    return psi_new / np.linalg.norm(psi_new)
