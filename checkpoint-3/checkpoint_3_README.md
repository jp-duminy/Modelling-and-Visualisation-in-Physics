# Checkpoint 3 MVP

Welcome to CP3 of MVP. Last updated: 2026-04-02

## Introduction

This checkpoint deals with partial differential equations, specifically the Cahn-Hilliard equation and Poisson's equation for E and B fields (including different algorithms for iteration). These can be numerically solved by discretisation; it is convenient to non-dimensionalise (C-H) or choose appropriate units (Poisson). This lets us work in characteristic length and timescales, meaning all quantities in the problem are of order unity. This allows simplicity when switching between problems: for example, if we have a tolerance criterion somewhere, knowing all quantities will be of order unity means this tolerance can be fixed (i.e. 1e-6) instead of having to choose one based on the specifics of the problem at hand. C-H can be nondimensionalised; Poisson cannot, but we can absorb quantities into the charge density rho such that it is dimensionless. This can be done without loss of generality.

## Dependencies
- Defaults: numpy, pandas, scipy, matplotlib
- numba, joblib, scienceplots

## Usage: General

This follows the familiar format from CP2: there exist global arguments and also individual arguments for each scenario.

- N: the size of the 2D lattice (x and y)
- animate: whether to animate or not
- measure: to run a data collection run
- dx: the characteristic length of the system
- n_steps: the number of steps for which a run should last

This checkpoint also uses joblib and numba for some easy optimisation.

## Usage: Cahn-Hilliard

The C-H solver has the following specific arguments:

- phi0: value of phi0 to initialise the lattice with (random uniform noise of 0.1 either side)
- timestep: characteristic time of the system
- measure-interval: the n_steps interval at which lattice measurements are taken

## Usage: Poisson

The Poisson solver has the following specific arguments:

- problem: whether to model either an electric or magnetic field
- method: the method of iteration (Jacobi or Gauss-Seidel)
- tol: convergence criterion tolerance
- omega: the value of omega for successive over-relaxation. This is by default 1.0 such that SOR reduces to default G-S.
- initial_state: the charge configuration to start from. This is a monopole/gaussian for electric and wire for magnetic.
