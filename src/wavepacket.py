import numpy as np

def gaussian_wavepacket(x, x0, k0, sigma):
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi
