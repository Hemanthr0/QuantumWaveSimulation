"""Plots and animation for the quantum wave simulation."""
import numpy as np
import matplotlib.pyplot as plt
from .solver import crank_nicolson
from .observables import energy_expectation, expectation_x, probability_norm


def plot_initial_state(x, psi, V, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, np.abs(psi)**2, label=r"$|\\psi(x,0)|^2$")
    ax.plot(x, V, "--", label="Potential V(x)")
    ax.set(xlabel="Position x", ylabel="Probability density / potential",
           title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def animate_wavefunction(x, psi0, V, dt, steps, title, frame_stride=5):
    dx = float(x[1] - x[0])
    psi = psi0.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    density_line, = ax.plot(x, np.abs(psi)**2, label=r"$|\\psi(x,t)|^2$")
    potential_line, = ax.plot(x, V, "r--", alpha=0.7, label="V(x)")
    ax.set(xlabel="Position x", ylabel="Probability density / potential",
           title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.show(block=False)

    times, norms, energies, positions = [], [], [], []
    for step in range(steps):
        psi = crank_nicolson(psi, V, dx, dt)
        if step % frame_stride == 0 or step == steps - 1:
            density_line.set_ydata(np.abs(psi)**2)
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"{title} | t = {(step+1)*dt:.3f}")
            fig.canvas.draw_idle()
            plt.pause(0.001)
        times.append((step+1)*dt)
        norms.append(probability_norm(psi, dx))
        energies.append(energy_expectation(psi, V, dx))
        positions.append(expectation_x(x, psi))

    fig2, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(times, norms)
    axes[0].set_ylabel("Probability norm")
    axes[1].plot(times, energies)
    axes[1].set_ylabel("Energy")
    axes[2].plot(times, positions)
    axes[2].set_ylabel(r"$\\langle x \\rangle$")
    axes[2].set_xlabel("Time")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig2.suptitle("Numerical diagnostics and expectation values")
    fig2.tight_layout()
    plt.show()
    return psi, {"time": np.asarray(times), "norm": np.asarray(norms),
                 "energy": np.asarray(energies), "x_mean": np.asarray(positions)}
