# QuantumWaveSimulation 2.0

A modular Python project for numerical investigation of one-dimensional quantum wave-packet dynamics using the Crank–Nicolson method.

## Features

- Free-particle, harmonic-oscillator, double-well, rectangular-barrier and finite-well potentials.
- Gaussian initial wave packet with integral-based normalization.
- Crank–Nicolson time propagation with a tridiagonal Thomas solver.
- No step-by-step artificial renormalization: probability conservation can be tested.
- Time traces of probability norm, energy expectation and mean position.
- Position/momentum uncertainty and Heisenberg uncertainty-product diagnostics.
- Probability-current plot.
- Automated physics-oriented tests.

## Physics model

The code solves the 1D time-dependent Schrödinger equation

\[
i\hbar\partial_t\psi =
\left[-\frac{\hbar^2}{2m}\partial_x^2+V(x)\right]\psi.
\]

The default units are dimensionless, with \(\hbar=m=1\). Fixed zero-wavefunction boundary conditions are used at both ends of the finite spatial domain.

## Setup

Python 3.9 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python -m src.main
```

Follow the prompts to select a potential and simulation parameters.

## Run tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests check initial normalization, probability conservation, approximate energy conservation, uncertainty positivity and the barrier profile.

## Suggested experiments

1. **Free packet spreading:** change `sigma` and compare the growth of \(\Delta x\).
2. **Barrier scattering:** select the rectangular barrier, then vary its height and width.
3. **Harmonic oscillator:** observe bounded motion in the confining potential.
4. **Numerical convergence:** repeat a run with smaller `dt` and finer `dx`; compare norm and energy diagnostics.

## Important interpretation notes

- This is a computational learning project, not an experimental quantum sensor.
- The finite spatial domain and fixed boundaries can affect long simulations if the packet reaches an edge.
- The reported left/barrier/right probabilities are instantaneous region probabilities, not automatically asymptotic reflection and transmission coefficients.
- Numerical energy expectation uses finite-difference derivatives and should be interpreted with discretization error in mind.

## Repository layout

```text
QuantumWaveSimulation/
├── src/
│   ├── main.py
│   ├── potentials.py
│   ├── solver.py
│   ├── wavepacket.py
│   ├── observables.py
│   └── visualization.py
├── tests/
│   └── test_physics.py
├── docs/
│   └── theory.md
├── results/
│   └── figures/
├── requirements.txt
└── README.md
```

## Next research-oriented extensions

- Compare finite-difference results against analytic free-particle evolution.
- Add grid/time-step convergence studies.
- Implement absorbing boundary layers for scattering simulations.
- Estimate asymptotic reflection and transmission only after validating the scattering protocol.
- Explore time-dependent potentials and driven systems.

## Author

Hemanth R — computational quantum mechanics learning project.
