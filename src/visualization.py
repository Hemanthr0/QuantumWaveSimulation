import numpy as np
import matplotlib.pyplot as plt
from solver import crank_nicolson

def expectation_values(x, psi, V, dx):
    prob = np.abs(psi)**2
    exp_x = np.trapz(x * prob, x)
    kinetic = -0.5 * np.trapz(np.conj(psi) * np.gradient(np.gradient(psi, dx), dx), x)
    potential = np.trapz(V * prob, x)
    return exp_x, kinetic.real + potential.real

def animate_wavefunction(x, psi0, V, dt, steps, title):
    dx = x[1] - x[0]
    psi = psi0.copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line1, = ax.plot(x, np.abs(psi)**2, label='|ψ(x)|²')
    line2, = ax.plot(x, V / np.max(V + 1e-9) * np.max(np.abs(psi)**2), 'r--', label='V(x) (scaled)')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{title} — Quantum Wave Simulation')
    ax.legend()

    energies = []
    exp_positions = []

    for i in range(steps):
        psi = crank_nicolson(psi, V, dx, dt)
        line1.set_ydata(np.abs(psi)**2)
        plt.pause(0.01)

        exp_x, E = expectation_values(x, psi, V, dx)
        energies.append(E)
        exp_positions.append(exp_x)

    plt.ioff()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(energies)
    plt.title("Expectation Energy vs Time")
    plt.subplot(2, 1, 2)
    plt.plot(exp_positions)
    plt.title("<x> vs Time")
    plt.tight_layout()
    plt.show()
