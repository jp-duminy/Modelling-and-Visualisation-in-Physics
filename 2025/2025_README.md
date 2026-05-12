# 2025 Exam README

Welcome to the code for the 2025 MVP exam: the crystal lattice PDE solver.

## Dependencies

- numpy, pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse, time

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

Arguments:

- N: the grid size 
- animate: toggle animation
- save_animation: saves an animation of the run
- max_steps: number of steps to iterate for
- dx: characteristic size of system
- dt: characteristic timescale of system
- task_*: toggles data collection for task_* where * = b,c,d (select argument)
- a: value of parameter a
- q0: value of parameter q0
- M: value of parameter M
- phi0: initial value of phi to initialise on the lattice

e.g. python 2025.py --animate --a 0.1 --q0 0.5 --M 0.1 --max_steps 10000 
e.g. python 2025.py --task_c

Note: the system takes a long time to evolve and equilibrate under its specified evolution equation. For this reason, it is recommended to use an 0.01 timestep and a large number of maximum steps (~100,000) to see results.

Last updated: 12/05/2026 JP Duminy
