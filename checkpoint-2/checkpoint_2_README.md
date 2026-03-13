# Checkpoint 2 MVP

Welcome to CP2 of MVP. Last updated: 2026-03-05

## Introduction

This checkpoint deals with two examples of cellular automata: Arthur Conway's game of life and the SIRS model.

Arthur Conway's game of life is a zero-player game on a 2D lattice where a cell can be either dead or alive. It is deterministic and parallel: whether a cell survives, dies or is reborn in a timestep depends on a set of simple rules that correspond to the status of its neighbouring eight cells. Different patterns can thus emerge. There are absorbing states, which are unchanging in time (such as the beehive); oscillating states, which flick between different patterns (the blinker); and travelling states, which move through the grid as a structure (the glider). Although simple, since it was published in 1970 the Game of Life has attracted a cult following to characterise and understand its various behaviours.

The susceptible-infected-recovered-susceptible (SIRS) model is another cellular automaton, but one which seeks to model the spread of infection. In its simplest form, a cell can be one of the three aforementioned states. It is sequential and stochastic: a cell may only move through the states sequentially, so for example infected cannot become susceptible. 

## Usage: General

As with CP1, the specific conditions of the simulation are specified by command line arguments. It is slightly different this time, however, as there are two models to choose from. Both use the same lattice initialisation, so the global arguments are:

- N: the size of the site (x and y)
- max-steps: the maximum number of steps to run in a simulation
- eq_steps: the number of steps to use when deciding on equilibrium conditions
- animate: whether to display animation or not

This checkpoint also uses joblib and numba for some easy optimisation.

## Usage: GoL

The GoL has the following specific arguments:

- config: use 'random' for random initialisation; 'glider' and 'oscillating' spawn a singular glider and blinker respectively.
- density: the fraction of alive cells on initialisation (higher density -> more alive cells)
- measure: run a data collection run for the histogram

Note: glider will also output a CoM plot and compute the speed of the glider.

## Usage: SIRS

The SIRS model has the following specific arguments:

- p1: the probability a susceptible site becomes infected
- p2: the probability an infected site recovers
- p3: the probability a recovered site becomes susceptible
- measure: begin a data collection run (tasks 3 & 4)
- immunity-frac: add immune (white) cells which do not interact with the disease as a fraction of total cells
- immunity: performs task 5 data collection and plotting


## Example Usage

python checkpoint_2/checkpoint_2.py --N 50 --max_steps 20000 --eq-steps 10 GoL --config glider --density 0.5
python checkpoint_2/checkpoint_2.py --N 50 --max_steps 20000 --eq-steps 10 GoL --config random --density 0.5 --measure
python checkpoint_2/checkpoint_2.py --N 50 --animate SIRS --p1 0.5 --p2 0.5 --p3 0.5 --measure

Specific SIRS values:
Absorbing state: p1 = 0.1, p2 = 0.5, p3 = 0.4
Dynamic equilibrium: p1 = 0.3, p2 = 0.3, p3 = 0.5
Cyclic infected waves: p1 = 0.8, p2 = 0.1, p3 = 0.015