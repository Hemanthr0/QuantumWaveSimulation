"""Run the QuantumWaveSimulation project from the repository root."""
import numpy as np
import matplotlib.pyplot as plt

from src.potentials import (free_particle, harmonic_oscillator,
                            double_well, barrier_potential, finite_well)
from src.wavepacket import gaussian_wavepacket
from src.solver import crank_nicolson
from src.observables import (probability_norm, uncertainties,
                             energy_expectation, probability_current,
                             region_probabilities)
from src.visualization import animate_wavefunction


def choose_potential(x):
    print("\nChoose a potential:")
    print("1. Free particle")
    print("2. Harmonic oscillator")
    print("3. Double well")
    print("4. Rectangular barrier (tunnelling/scattering)")
    print("5. Finite square well")
    choice = input("Choice [1]: ").strip() or "1"

    if choice == "2":
        k = float(input("Spring constant k [0.5]: ") or 0.5)
        return harmonic_oscillator(x, k), "Harmonic oscillator"
    if choice == "3":
        a = float(input("Parameter a [0.04]: ") or 0.04)
        b = float(input("Parameter b [1.0]: ") or 1.0)
        return double_well(x, a, b), "Double well"
    if choice == "4":
        height = float(input("Barrier height V0 [2.0]: ") or 2.0)
        width = float(input("Barrier width [0.8]: ") or 0.8)
        return barrier_potential(x, height, width), "Rectangular barrier"
    if choice == "5":
        depth = float(input("Well depth [1.0]: ") or 1.0)
        width = float(input("Well width [2.0]: ") or 2.0)
        return finite_well(x, depth, width), "Finite square well"
    return free_particle(x), "Free particle"


def main():
    print("QuantumWaveSimulation 2.0")
    print("Units: hbar = m = 1 (dimensionless units)")
    x = np.linspace(-10.0, 10.0, 801)
    dx = x[1] - x[0]
    V, name = choose_potential(x)
    x0 = float(input("Initial packet centre x0 [-4]: ") or -4.0)
    k0 = float(input("Initial wave number k0 [3]: ") or 3.0)
    sigma = float(input("Initial position width sigma [0.6]: ") or 0.6)
    dt = float(input("Time step dt [0.0005]: ") or 0.0005)
    steps = int(input("Number of steps [1200]: ") or 1200)

    psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    print(f"\nInitial probability norm: {probability_norm(psi0, dx):.8f}")
    print("Propagating wave packet...")
    psi_final, diagnostics = animate_wavefunction(
        x, psi0, V, dt, steps, f"{name} — Crank–Nicolson"
    )

    final_uncertainty = uncertainties(x, psi_final)
    final_energy = energy_expectation(psi_final, V, dx)
    print("\nFinal-state diagnostics")
    print(f"Probability norm: {probability_norm(psi_final, dx):.8f}")
    print(f"Energy expectation: {final_energy:.8f}")
    print(f"Delta x: {final_uncertainty['delta_x']:.6f}")
    print(f"Delta p: {final_uncertainty['delta_p']:.6f}")
    print(f"Delta x Delta p: {final_uncertainty['uncertainty_product']:.6f}")
    print(f"Heisenberg lower bound (hbar/2): {final_uncertainty['heisenberg_limit']:.6f}")

    current = probability_current(psi_final, dx)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, current)
    ax.set(title="Probability current at final time",
           xlabel="Position x", ylabel="Probability current j(x,t)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    if name == "Rectangular barrier":
        width = float(input("Re-enter barrier width for region analysis [0.8]: ") or 0.8)
        probs = region_probabilities(x, psi_final, -width/2, width/2)
        print("Final probability fractions (packet may still overlap barrier):")
        print(f"Left: {probs['left']:.4f} | Barrier: {probs['barrier']:.4f} | Right: {probs['right']:.4f}")


if __name__ == "__main__":
    main()
