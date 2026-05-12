# 2026 Exam README

Welcome to the code for the 2020 MVP exam: the contact process model.

## Dependencies

- numpy, pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

Arguments:

- N: the grid size 
- animate: toggle animation
- max_steps: number of steps to iterate for
- measure_interval: the number of steps between measurements
- task_*: toggles data collection for task_* where * = b,c,d,f (select argument)
- p: the characteristic infection probability

Note: survival probability trials are not callable from the command terminal outside of data collection for their tasks. Should you desire this mode, it is a simple change using initialise_grid and adding a new argument.

Errors are computed with block bootstrap.

e.g. python 2020.py --animate --p 0.5
e.g. python 2020.py --task_c 

Last updated: 12/05/2026 JP Duminy
