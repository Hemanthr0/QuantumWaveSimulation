import numpy as np
from solver import crank_nicolson
from potentials import *
from wavepacket import gaussian_wavepacket
from visualization import animate_wavefunction

def choose_potential(x):
    print("Choose a potential to simulate:")
    print("1. Free Particle")
    print("2. Harmonic Oscillator")
    print("3. Double Well")
    print("4. Barrier Potential (Quantum Tunneling)")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        return free_particle(x), "Free Particle"
    elif choice == "2":
        k = float(input("Enter spring constant k (e.g. 0.5): ") or 0.5)
        return harmonic_oscillator(x, k), f"Harmonic Oscillator (k={k})"
    elif choice == "3":
        a = float(input("Enter a (e.g. 0.04): ") or 0.04)
        b = float(input("Enter b (e.g. 1): ") or 1)
        return double_well(x, a, b), f"Double Well (a={a}, b={b})"
    elif choice == "4":
        V0 = float(input("Enter barrier height V0 (e.g. 1): ") or 1)
        width = float(input("Enter barrier width (e.g. 0.2): ") or 0.2)
        return barrier_potential(x, V0, width), f"Barrier Potential (V0={V0}, width={width})"
    else:
        print("Invalid choice. Defaulting to Free Particle.")
        return free_particle(x), "Free Particle"

def main():
    x = np.linspace(-5, 5, 500)
    dx = x[1] - x[0]

    V, V_name = choose_potential(x)
    psi0 = gaussian_wavepacket(x, x0=-2, k0=5, sigma=0.5)

    dt = 0.001
    steps = 300

    animate_wavefunction(x, psi0, V, dt, steps, V_name)

if __name__ == "__main__":
    main()
