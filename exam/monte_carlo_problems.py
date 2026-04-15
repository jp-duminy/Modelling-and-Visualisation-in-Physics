"""

Monte Carlo Methods (Ising Model).

Top functions are the numba engine room which handles agnostic computations.
I like to work with classes, so I will make a bootstrap error class and a specific ising model class.

"""

import numpy as np 
from numba import njit, prange
import pandas

import argparse

#
# engine room
#

@njit
def metropolis_acceptance(delta_E: float, beta: float) -> bool:
    """
    Returns Metropolis algorithm acceptance criterion.
    """
    if delta_E <= 0:
        return True
    return np.random.random() < np.exp(-beta * delta_E)

@njit
def total_energy_ising(lattice: np.ndarray, N: int) -> int:
    """
    Total energy for an Ising model lattice, where spins are +/- 1.
    """
    E = 0.0
    for i in prange(N):
        for j in range(N):
            E += -lattice[i,j] * (lattice[i,(j+1) % N] + lattice[(i+1) % N,j]) # right and top neighbour (avoid double counting)
    return E

@njit
def total_magnetisation(lattice: np.ndarray) -> int:
    """
    Total magnetisation of the lattice (sum of all sites)
    """
    return np.sum(lattice)

@njit
def nearest_neighbours_sum(lattice: np.ndarray, pos: tuple, N: int) -> int:
    """
    Sum the four nearest neighbours.
    """
    i, j = pos
    nn_sum = 0
    nn_sum += (lattice[(i - 1) % N,j] + # up
                lattice[(i + 1) % N,j] + # down
                lattice[i,(j - 1) % N] + # left
                lattice[i,(j + 1) % N]) # right
    return nn_sum

@njit
def neighbour_check(pos1: tuple, pos2: tuple, N: int) -> bool:
    """
    Check if two points on the lattice are neighbours (vertically or horizontally).
    Necessary for Kawasaki dynamics.
    """
    i1, j1 = pos1
    i2, j2 = pos2

    # distances in y and x directions
    di = min(abs(i1 - i2), N - abs(i1 - i2)) # min() method handles periodic boundary conditions
    dj = min(abs(j1 - j2), N - abs(j1 - j2))

    return (di == 1 and dj == 0) or (di == 0 and dj == 1) # distance must only be 1 (ignore diagonals)
    
@njit
def glauber_sweep(lattice: np.ndarray, N: int, beta: float) -> None:
    """
    Performs one full Glauber sweep over the lattice.
    """
    n_sites = N**2

    for _ in range(n_sites): # do not use prange, monte carlo is sequential

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        pos = (i, j)

        spin = lattice[i,j]
        sigma = nearest_neighbours_sum(lattice, pos, N)
        delta_E = 2 * sigma * spin

        if metropolis_acceptance(delta_E, beta):
            lattice[i,j] *= -1

@njit
def kawasaki_selection(lattice: np.ndarray, N: int) -> tuple:
    """
    Selects two random sites on the lattice for Kawasaki dynamics; computes their delta_Es.
    """
    # first site
    i1 = np.random.randint(0, N) # rows
    j1 = np.random.randint(0, N) # columns

    # second site: ensure they are not the same site
    i2, j2 = i1, j1 # at first, make the second site equal to the first site
    while (i2, j2) == (i1, j1): # then use a while loop to change them to a different site
        i2 = np.random.randint(0, N)
        j2 = np.random.randint(0, N) # while loop ensures the same site is not randomly picked again

    pos1, pos2 = (i1, j1), (i2, j2)

    spin1 = lattice[pos1]
    spin2 = lattice[pos2]

    # ignore condition if the spins are identical (no effect)
    if spin1 == spin2:
        return 0, pos1, pos2
    
    sigma1 = nearest_neighbours_sum(lattice, pos1, N)
    delta_E1 = 2 * sigma1 * spin1

    sigma2 = nearest_neighbours_sum(lattice, pos2, N)
    delta_E2 = 2 * sigma2 * spin2

    if neighbour_check(pos1, pos2, N): # if they are nearest neighbours
        delta_E2 = 2 * (sigma2 - spin1) * spin2 # simply subtract the flipped spin from the sigma term
        delta_E1 = 2 * (sigma1 - spin2) * spin1

    return delta_E1 + delta_E2, pos1, pos2

@njit
def kawasaki_sweep(lattice: np.ndarray, N: int, beta: float) -> None:
    """
    Performs one full Kawasaki sweep over the lattice.
    """
    n_sites = N**2
    
    for _ in range(n_sites): # do not use prange, monte carlo is sequential

        delta_E, pos1, pos2 = kawasaki_selection(lattice, N)
        if metropolis_acceptance(delta_E, beta):
            lattice[pos1], lattice[pos2] = lattice[pos2], lattice[pos1]

