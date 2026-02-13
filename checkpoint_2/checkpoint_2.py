import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse

class Grid:
    def __init__(self, N, config, density, game):
        self.N = N
        self.config = config
        self.density = density # density of live cells
        self.game = game

        # GoL shifts for np.roll
        self.eight_shifts = [(-1, -1), # top-left
                             (-1, 0), # top
                             (-1, 1), # top-right
                             (0, -1), # left
                            (0, 1), # right
                            (1, -1), # bottom-left
                            (1, 0), # bottom
                            (1, 1) # bottom-right
                            ]
        
        # SIRS shifts for np.roll
        self.four_shifts = [(-1, 0), # top
                            (0, -1), # left
                            (0, 1), # right
                            (1, 0) # bottom
                            ]

        self.lattice = None

    def initialise_lattice(self):
        """
        Agnostic lattice initialisation for either game. 
        """
        if self.game == 'GoL':
            if self.config == 'random':
                self.lattice = (np.random.random((self.N, self.N)) < self.density).astype(int) # density of live cells
            # can add more configurations for glider etc. 
        elif self.game == 'SIRS':
            self.lattice = np.random.choice([0, 1, 2], size=(self.N, self.N))

    def count_neighbours(self):
        """
        Counts the nearest-neighbour sum, pertinent to the GoL.
        """
        # make an empty lattice that we can store the total for
        total = np.zeros_like(self.lattice)

        for dx, dy in self.eight_shifts:
            total += np.roll(np.roll(self.lattice, dx, axis=0), dy, axis=1) # np.roll implementation (lecture 1)

        return total
    
class GameOfLife:
    def __init__(self, grid, max_steps, eq_steps):
        self.grid = grid # pass grid object (same logic as CP1)
        self.grid.initialise_lattice()
        self.history = deque(maxlen=eq_steps)
        self.eq_time = None

        self.max_steps = max_steps

    def step(self):
        """
        Runs one step of the game of life over the entire grid.
        """
        neighbour_sum = self.grid.count_neighbours() # number of alive neighbouring cells
    
        # make use of boolean logic + union operator for quick vectorised operations
        resurrected = (self.grid.lattice == 0) & (neighbour_sum == 3) # cell is resurrected if it is dead and has 3 alive neighbours
        survive = (self.grid.lattice == 1) & ((neighbour_sum == 2) | (neighbour_sum == 3)) # cell survives if it is alive and has 2 or 3 alive neighbours

        # store the boolean values of surviving/reborn cells
        self.grid.lattice = (resurrected | survive).astype(int) # astype(int) means all-else becomes 0, i.e. the death condition is satisfied via an implicit else: condition

    def equilibrium_state(self) -> bool:
        """
        Checks whether the game has reached an equilibrium state & defines which type it is.
        """
        # for first step
        if len(self.history) == 0:
            return False
        
        # absorbing state (check this FIRST)
        if np.array_equal(self.grid.lattice, self.history[-1]):
            print(f"Absorbing state reached.")
            return True
        
        # oscillating state
        if any(np.array_equal(self.grid.lattice, h) for h in self.history): # any() should not terminate travelling states in theory
            print(f"Oscillating state reached.")
            return True
        
        return False

    def run(self):
        """
        Runs the game of life until the user-defined maximum number of steps is reached.
        """
        step_number = 0
        while step_number < self.max_steps:
            current_state = self.grid.lattice.copy() # from lecture 1, .copy() is necessary
            self.history.append(current_state) 
            self.step()
            eq = self.equilibrium_state()
            if eq:
                self.eq_time = step_number
                break
            step_number += 1
        if self.eq_time is None:
            print(f"Did not equilibrate before reaching maximum steps.")
            self.eq_time = self.max_steps

class Visualiser:
    def __init__(self, updater, interval=50, cmap='binary'):
        self.updater = updater
        self.interval = interval
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(updater.grid.lattice, cmap=cmap, interpolation='nearest')

    def _update(self, frame):
        self.updater.step()
        self.im.set_data(self.updater.grid.lattice)
        self.ax.set_title(f'Step: {frame}')
        return [self.im]

    def animate(self):
        ani = FuncAnimation(self.fig, self._update, frames=self.updater.max_steps, interval=self.interval, blit=True)
        plt.show()

