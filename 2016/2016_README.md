# 2016 Exam README

This is the code for the 2016 MVP exam: the Ising model with an external magnetic field.

## Dependencies

- numpy, pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse, time

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

Arguments:

- N: the size of the NxN lattice
- dynamics: please always input glauber (not yet deprecated)
- n_sweeps: the number of sweeps to run for each simulation (or each iteration in a data collection run)
- measure_interval: the number of sweeps between measurements
- animate: whether or not to animate
- sweep_direction: whether to start from high h or low h
- save_data: data collection mode, for part c onwards
- lattice_config: whether to initialise the lattice with random spins or pointing in one direction
- h: the initial value of the magnetic field across the lattice
- dynamic_h: whether to dynamically update the magnetic field (for part d onwards)

Or run --help for a more thorough explanation.

Example usage: 

python 2016_exam.py --N 50 --dynamics glauber --n_sweeps 100000 --measure_interval 100  --h 0.0 --lattice_config random  --dynamic_h 

## To Note

- Errors are computed with bootstrap
System can record:
- Average magnetisation and average staggered magnetisation with variances, by default
- Average energy, by default
- Staggered magnetisation as a function of time, with dynamic_h mode
- Maximum field strength, with dynamic_h mode

Last updated: 04/05/2026, JP Duminy 