# Theory and numerical method

## Model

The project solves the one-dimensional time-dependent Schrödinger equation

\[
i\hbar\frac{\partial \psi(x,t)}{\partial t}
=
\left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}+V(x)\right]\psi(x,t).
\]

The default simulations use dimensionless units \(\hbar=m=1\). The spatial second derivative is approximated by a centred finite difference. The resulting Hamiltonian is tridiagonal.

## Crank–Nicolson propagation

For a time-independent Hamiltonian \(H\), the Crank–Nicolson update is

\[
\left(I+\frac{i\Delta t}{2\hbar}H\right)\psi^{n+1}
=
\left(I-\frac{i\Delta t}{2\hbar}H\right)\psi^n.
\]

For a Hermitian discretized Hamiltonian this update is unitary, up to linear-solver and floating-point errors. The code therefore does not renormalize the state after every step; norm conservation remains a meaningful diagnostic.

The two spatial endpoints are fixed to zero (Dirichlet boundary conditions). The wave packet should remain far from these boundaries during a run to avoid boundary effects.

## Observables

The probability density is \(|\psi|^2\), and the probability current in one dimension is

\[
j(x,t)=\frac{\hbar}{m}\operatorname{Im}\left(\psi^*\frac{\partial\psi}{\partial x}\right).
\]

Position and momentum uncertainties are calculated from the corresponding expectation values. Numerical derivatives and finite domains introduce discretization and boundary errors, so the uncertainty relation and energy conservation should be checked under grid/time-step refinement.

## Scope and limitations

This is an educational computational model, not a validated experimental quantum sensor model. The project currently uses a single particle in one spatial dimension, static potentials, finite-difference derivatives and hard-wall boundaries. Scattering-region probability fractions are snapshots; they should not be interpreted as asymptotic reflection/transmission coefficients unless the packet has cleanly separated from the barrier and the domain/boundaries are sufficiently large.
