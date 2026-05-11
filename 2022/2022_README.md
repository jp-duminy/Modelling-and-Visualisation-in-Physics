# 2022 Exam README

This is the code for the 2022 MVP exam: the three coupled chemical species PDES.
(NB I found this exam really tough)

## Dependencies

- numpy, pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse, time

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

The code is separated into the engine room (the fast solver functions) and a class which gets the physics out of there.

Global arguments:

- animate: toggles animation display
- N: the grid size
- max_steps: the number of steps for which the simulation should run
- dx: the characteristic size of the system (recommended 1.0)
- dt: the timestep of the system (recommended 0.001)
- task_x: toggle data collection for task_x where x= b,c,d,e,f (separate commands)
- D: value of the diffusion coefficient
- p: value of the constant p
- q: value of the constant q

e.g. python 2022.py --N 50 --animate --dx 1.0 --dt 0.001 -D 1.0 -p 2.5 -q 1.0
e.g. python 2022.py --task_c

Last updated: 11/05/2026 JP Duminy