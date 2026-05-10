# 2023 Exam README

This is the code for the 2023 MVP exam: the Rock-Paper-Scissors (RPS) cellular automata model.

The code functions in either a parallel, deterministic configuration or sequential, stochastic (see Usage).

## Dependencies

- numpy, pandas, matplotlib, numba, joblib, scienceplots
- Defaults: argparse, time

Note: for accurate data collection results, perform one burn-in run of the code; this is because the numba JIT compiling can interfere with the results, so it is best to run the code so numba may cache the functions, then begin collecting data.

## Usage

There are two classes: ParallelRPS and SequentialRPS. They have overlapping subarguments; however, the distinction is important.

Global arguments:

- N: the grid size
- max_steps: the number of steps for which the simulation should run
- animate: toggles animation display
- eq_steps: the number of equilibration steps before measurments are taken

Subarguments - Parallel:

- measure: collects data for task d) of the exam.
- measure_interval: interval between measurement taking

e.g. python 2023.py --N 100  --animate Parallel --measure --measure_interval 1000                      

Subarguments - Sequential:

- p1: the probability of R -> P
- p2: the probability of P -> S
- p3: the probability of S -> R
- measure: collects data for task e) of the exam.
- measure_interval: interval between measurement taking.

e.g. python 2023.py --N 50 --animate Sequential --p1 0.5 --p2 0.5 --p3 0.5

Last updated: 10/05/2026 JP Duminy