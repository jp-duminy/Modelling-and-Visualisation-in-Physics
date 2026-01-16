# Checkpoint 1 MVP

This is the code for MVP CP1: Monte Carlo simulations, contained within checkpoint_1.py.

## Introduction

CP1 simulates the 2D Ising model on a lattice of user-desired size. The Ising model is a statistical-mechanical model of ferromagnetism. Each site on the lattice exists in either a spin-up (+1) or spin-down (-1) state and its properties are dependent on its interaction with its nearest neighbours. CP1 is specifically a Monte Carlo simulation: equilibrium states of the Ising model can be sampled with either Glauber dynamics in conjunction with the Metropolis algorithm, or Kawasaki dynamics.

The code uses a specific seed such that simulations are reproducible.

## Running the Code

The simulation takes a number of command-line arguments. It can be run with:

python checkpoint_1.py

And associated arguments, which can be accessed with --help but are listed here:

- lenx / leny: the x/y length of the lattice (recommended 50x50)
- temp: the initial value of temperature that the simulation will start with (range 1.0 - 3.0)
- dynamics: the dynamics (Glauber/Kawasaki) to simulate
- lattice-config: whether to initialise with all-up or random spins (recommended all-up for Glauber, random for Kawasaki)
- sweeps: the number of sweeps to take per temperature
- sweep-direction: whether to increase or decrease temperature
- measure: toggles measuring of physical quantities of interest
- save-data: toggles saving of measured data
- animate: toggles live lattice plot
- update-interval: toggles interval at which data is plotted

Only temp and dynamics are required arguments; lattice size, number of sweeps & lattice configuration default to 50x50, 10,000 and all-up respectively.

## Example Usage

python checkpoint_1.py --dynamics glauber --lenx 50 --leny 50 --temp 3.0 --sweep-direction down --lattice-config random --animate --measure --save-data
python checkpoint_1.py --dynamics kawasaki --lenx 50 --leny 50 --temp 1.0 --sweep-direction up --lattice-config random --animate --measure --save-data

## Physical Quantities

If measure is toggled, the simulation will record data of the following physical quantities of interest:

- Average Total Magnetisation
- Susceptibility
- Average Total Energy
- Heat Capacity per Spin

The simulation sweeps through a range of temperatures depending on user input, within the range T = 1.0-3.0.

This data will be plotted for the entire simulation at its conclusion. save-data can be toggled to automatically save the outputs. 

## Errors

Errors are computed using the bootstrap method, which resamples (with replacement) from the data k number of times to create k sets of size n from which the errors can be computed from the standard deviation of the new samples. Errors are automatically computed once --measure is toggled and will be displayed on the plotted data as well as saved to the outputs if requested.

