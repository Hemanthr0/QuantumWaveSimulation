import numpy as np

def free_particle(x):
    return np.zeros_like(x)

def harmonic_oscillator(x, k=0.5):
    return 0.5 * k * x**2

def double_well(x, a=0.04, b=1):
    return a * (x**2 - b)**2

def barrier_potential(x, V0=1.0, width=0.2):
    V = np.zeros_like(x)
    mask = np.abs(x) < width/2
    V[mask] = V0
    return V
