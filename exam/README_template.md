# 2026 Exam README

Welcome to the code for the 2026 MVP exam: 

## Dependencies

- numpy, (scipy), pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse, time

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

Arguments:

- N: the grid size 
- animate: toggle animation
- max_steps: number of steps to iterate for
- eq_steps: the number of steps to equilibrate the system for before taking measurements
- measure_interval: the number of steps between measurements
- task_*: toggles data collection for task_* where * = b,c,d (select argument)
- a: value of parameter a
- T0: the initial value of T

e.g. python 2026.py --animate --a 0.1 --max_steps 10000 
e.g. python 2026.py --task_c 

Last updated: 13/05/2026 JP Duminy
