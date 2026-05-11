"""

Cellular Automata (SIRS, GoL).

More intertwined than previous checkpoint.

"""

# default packages
import numpy as np
import pandas as pd

# utils
from collections import deque # for history checking purposes
import argparse # for command line functionality
from time import perf_counter

# plotting
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import scienceplots
plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

# overdrive
from numba import njit
from joblib import Parallel, delayed

#
# lattice shifts
#

four_shifts = np.array([(-1, 0), # top
                        (0, -1), # left
                        (0, 1), # right
                        (1, 0) # bottom
                        ])

eight_shifts = np.array([(-1, -1), # top-left
                            (-1, 0), # top-middle
                            (-1, 1), # top-right
                            (0, -1), # left
                            (0, 1), # right
                            (1, -1), # bottom-left
                            (1, 0), # bottom-middle
                            (1, 1) # bottom-right
                            ])

#
# engine room
#

def four_neighbours(grid: np.ndarray) -> np.ndarray:
    """
    Parallel contribution sum from nearest four neighbours.
    """
    neighbour_sum = np.zeros_like(grid)
    
    for dx, dy in four_shifts:
        neighbour_sum += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)

    return neighbour_sum

def eight_neighbours(grid: np.ndarray) -> np.ndarray:
    """
    Parallel contribution sum from nearest eight neighbours (includes diagonals).
    """
    neighbour_sum = np.zeros_like(grid)
    
    for dx, dy in eight_shifts:
        neighbour_sum += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)

    return neighbour_sum

def eight_state_neighbours(grid: np.ndarray, state: int) -> np.ndarray:
    """
    Parallel contribution sum from nearest eight neighbours in same state (includes diagonals).
    """
    neighbour_sum = np.zeros_like(grid)
    
    for dx, dy in eight_shifts:
        neighbour_sum += np.roll(np.roll((grid==state), dx, axis=0), dy, axis=1)

    return neighbour_sum

@njit
def sirs_sweep(grid: np.ndarray, N: int, p1: float, p2: float, 
               p3: float) -> None:
    """
    Sequential, stochastic SIRS sweep over grid.
    """
    n_sites = N**2
    # pregenerate random sites
    sites_i = np.random.randint(0, N, size=n_sites)
    sites_j = np.random.randint(0, N, size=n_sites)
    randoms = np.random.random(n_sites)

    for k in range(n_sites):
        i = sites_i[k]
        j = sites_j[k]

        # p1: susceptible -> infected
        if grid[i, j] == 0:
            for m in range(4): # check across four neighbours
                ni = (i + four_shifts[m,0]) % N
                nj = (j + four_shifts[m,1]) % N
                if grid[ni,nj] == 1: # 1: neighbour is infected
                    if randoms[k] < p1:
                        grid[i,j] = 1
                    break # no need to keep iterating

        # p2: infected -> recovered
        elif grid[i,j] == 1:
            if randoms[k] < p2:
                grid[i,j] = 2

        # p3: recovered -> susceptible
        elif grid[i,j] == 2:
            if randoms[k] < p3:
                grid[i,j] = 0

class GameOfLife:

    def __init__(self, N: int, density: float, config: str, eq_steps: int):

        self.N = N
        self.density = density # density of live cells
        self.config = config
        self.history = deque(maxlen=eq_steps)
        
        self.equilibrated = False
        self.eq_time = None

        self.initialise_grid()

    def initialise_grid(self) -> np.ndarray:
        """
        Initialises the game of life grid.
        """
        if self.config == 'random':
            self.grid = (np.random.random((self.N, self.N)) < self.density).astype(int) # must cast to integer array
        else:
            centre = [self.N // 2, self.N // 2]
            self.grid = np.zeros((self.N, self.N))
            if self.config == 'oscillating':
                self.place_blinker(centre)
            elif self.config == 'travelling':
                self.place_glider(centre)
                self.com_history = []
                self.offset = np.array([0.0, 0.0])
                self.previous_com = None

    def place_glider(self, loc: tuple) -> np.ndarray:
        """
        Place a glider (translating state) on the input loc.
        """
        i, j = loc

        # glider shape
        glider = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 1, 1]])

        # slice the lattice to place the glider
        self.grid[i-1:i+2, j-1:j+2] = glider     

    def place_blinker(self, loc: tuple) -> np.ndarray:
        """
        Place a blinker (oscillating state) on the input loc.
        """  
        i, j = loc

        self.grid[i-1:i+2, j] = 1

    def compute_glider_com(self):
        """
        Computes the centre-of-mass of the glider, appropriately accounting for PBCs.
        """
        live = np.argwhere(self.grid == 1)
        com = np.zeros(2)
        for dim in range(2):
            coords = live[:, dim].astype(float)
            if coords.max() - coords.min() > self.N / 2:
                # cluster straddles boundary — shift low values up
                coords[coords < self.N / 2] += self.N
            com[dim] = coords.mean()
        return com
    
    def unwrap_centre_of_mass(self, raw_com):
        """
        Handles centre of mass by unwrapping the periodic boundary conditions.
        """
        raw = np.array(raw_com)
        if self.previous_com is not None:
            delta = raw - self.previous_com # change in centre-of-mass
            # detect wrapping in both dimensions
            # the N/5 means that the glider has jumped more than N/5 (given its speed, this means it has crossed the boundary)
            self.offset[delta > self.N/5] -= self.N
            self.offset[delta < -self.N/5] += self.N
        self.previous_com = raw.copy()
        return raw + self.offset # add offset to account for PBCs
    
    def estimate_speed(self):
        """
        Estimates the speed of the glider from the centre-of-mass history.
        """
        com_array = np.array(self.com_history)
        t = np.arange(len(com_array))

        fit_x = np.polyfit(t, com_array[:, 0], 1)
        fit_y = np.polyfit(t, com_array[:, 1], 1)

        pd.DataFrame({
        'step': t,
        'com_x': com_array[:, 0],
        'com_y': com_array[:, 1],
        'fit_x': np.polyval(fit_x, t),
        'fit_y': np.polyval(fit_y, t)
        }).to_csv('glider_com.csv', index=False)

        fig, ax = plt.subplots()
        ax.plot(t, com_array[:, 0], '.', label='COM x', markersize=2)
        ax.plot(t, com_array[:, 1], '.', label='COM y', markersize=2)
        # polyval uses the polynomial fit
        ax.plot(t, np.polyval(fit_x, t), '--', label=f'$v_x$ = {fit_x[0]:.4f}')
        ax.plot(t, np.polyval(fit_y, t), '--', label=f'$v_y$ = {fit_y[0]:.4f}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Centre-of-Mass Position')
        ax.set_title('Glider Centre-of-Mass with Linear Fit')
        ax.legend()
        plt.show()

        # 0.35355

        # velocity is simply the gradient
        vx = fit_x[0]
        vy = fit_y[0]

        return np.sqrt(vx**2 + vy**2)

    def step(self) -> np.ndarray:
        """
        One global application of the GoL rules (parallel, deterministic).
        """
        current_state = self.grid.copy()
        self.history.append(current_state)
        neighbour_sum = eight_neighbours(current_state) # number of alive neighbouring cells

        # make use of boolean logic + union operator for quick vectorised operations
        resurrected = (current_state == 0) & (neighbour_sum == 3) # cell is resurrected if it is dead and has 3 alive neighbours
        survive = (current_state == 1) & ((neighbour_sum == 2) | (neighbour_sum == 3)) # cell survives if it is alive and has 2 or 3 alive neighbours

        if self.config == 'travelling':
            com = self.compute_glider_com()
            offset_com = self.unwrap_centre_of_mass(com)
            self.com_history.append(offset_com)

        # store the boolean values of surviving/reborn cells
        self.grid = (resurrected | survive).astype(int) # note: astype(int) on boolean logic

    def equilibrium_state(self) -> bool:
        """
        Checks whether the game has reached an equilibrium state & defines which type it is.
        """
        # for first step
        if len(self.history) == 0:
            return False
        
        # absorbing state (check this FIRST)
        if np.array_equal(self.grid, self.history[-1]):
            print(f"Absorbing state reached.")
            return True
        
        # oscillating state
        if any(np.array_equal(self.grid, h) for h in self.history): # any() should not terminate travelling states in theory
            print(f"Oscillating state reached.")
            return True
        
        return False
    
    def _animate_step(self, frame):
        """
        Animates a step of the lattice.
        """
        self.step()

        self.im.set_data(self.grid)
        self.ax.set_title(f'Step: {frame}')

        if not self.equilibrated and self.equilibrium_state():
            self.eq_time = frame
            self.equilibrated = True

        return [self.im]
    
    def run(self, animate: bool = False, max_steps: int = 10000):
        """
        Runs the game of life until the user-defined maximum number of steps is reached.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid, cmap='binary', interpolation='nearest')
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()
        else:
            step_number = 0
            while step_number < max_steps:
                self.step()

                if not self.equilibrated and self.equilibrium_state():
                    self.eq_time = step_number
                    print(f"Reached equilibrium after {step_number} steps.")
                    self.equilibrated = True

                step_number += 1

        if self.eq_time is None:
            print("Did not equilibrate.")
            self.eq_time = max_steps

class SIRS:

    def __init__(self, N: int, p1: float, p2: float, p3: float, 
                 frac_immune: float = 0.0):

        # infection probabilities
        self.p1 = p1 # p(S -> I), suscpetible == 0
        self.p2 = p2 # p(I -> R), infected == 1
        self.p3 = p3 # p(R -> S), recovered == 2

        self.N = N
        self.grid = np.random.choice([0, 1, 2], size=(N, N)).astype(int)
        self.cmap = ListedColormap(['blue', 'red', 'green', 'white']) # white: immune
        self.norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

        if frac_immune > 0:
            self.apply_immunity(frac_immune=frac_immune)

    def apply_immunity(self, frac_immune: float) -> np.ndarray:
        """
        Apply immune cells (value 3) to the lattice (for SIRS)
        """
        n_immune = int(self.N**2 * frac_immune) # must cast to int to avoid random choice crashing
        immune_sites = np.random.choice(self.N**2, size=n_immune, replace=False) 
        self.grid.flat[immune_sites] = 3

    def sweep(self) -> np.ndarray:
        """
        Wraps the engine room function for visibility.
        """
        sirs_sweep(self.grid, self.N, self.p1, self.p2, self.p3)

    def _animate_sweep(self, frame) -> list:
        """
        Wraps sweep() for animation.
        """
        self.sweep()

        self.im.set_data(self.grid)
        self.ax.set_title(f'Step: {frame}')

        return [self.im]
    
    def measure_infected(self) -> int:
        """
        Helper function to measure the fraction of infected on the site at a given time.
        """
        return np.sum(self.grid == 1) / self.N**2

    def run(self, animate: bool = False, max_steps: int = 10000, measure_interval: int = 1000, eq_steps: int = 100):
        """
        Runs the SIRS model until the user-defined maximum number of steps is reached.
        """
        self.infected_fraction = []
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            cbar = self.fig.colorbar(self.im, ax=self.ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Susceptible', 'Infected', 'Recovered'])
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for i in range(max_steps):
                self.sweep()
                if i > eq_steps and i % measure_interval == 0:
                    self.infected_fraction.append(self.measure_infected())
    
class BootstrapErrorAnalysis:
    """
    Implementation of bootstrap resampling error analysis.
    Reused some methods from CP1.
    """
    def __init__(self, k=1000, seed=2317434):
        self.k = k # number of times we resample
        self.generator = np.random.default_rng(seed) # my student ID

    def resample_blocks(self, data, block_size=50):
        """
        Instead of resampling from the full dataset, we resample from blocks.
        
        Our data is correlated over time since SIRS is time-dependent. For this reason it is better
        to use block bootstrap to replicate the correlation and account for this in our error bars,
        avoiding underestimation of our errors. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        """
        n_blocks = len(data) // block_size
        blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        chosen = self.generator.integers(0, n_blocks, size=n_blocks)
        return blocks[chosen].flatten()

    def calculate_errors(self, infected_timeseries, block_size=50):
        """
        Compute the errors from the resampled data using block bootstrap.
        """
        data = np.array(infected_timeseries)
        bootstrap_variances = np.zeros(self.k)
        for i in range(self.k):
            resampled = self.resample_blocks(data, block_size) 
            bootstrap_variances[i] = np.var(resampled)
        return np.std(bootstrap_variances)

parser = argparse.ArgumentParser(description='Cellular Automata: GoL and SIRS')

# shared arguments
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--eq_steps', type=int, default=10)

subparsers = parser.add_subparsers(dest='game', required=True)

# game of life arguments
gol_parser = subparsers.add_parser('GoL')
gol_parser.add_argument('--config', type=str, choices=['random', 'travelling', 'oscillating'])
gol_parser.add_argument('--density', type=float, default=0.5)

# SIRS arguments
sirs_parser = subparsers.add_parser('SIRS')
sirs_parser.add_argument('--p1', type=float, required=True, help='p(S->I)')
sirs_parser.add_argument('--p2', type=float, required=True, help='p(I->R)')
sirs_parser.add_argument('--p3', type=float, required=True, help='p(R->S)')
sirs_parser.add_argument('--immunity_frac', type=float, default=0.0)
sirs_parser.add_argument('--measure_interval', type=int, default=100)

def main():
    args = parser.parse_args()

    if args.game == 'SIRS':
        sirs = SIRS(N=args.N, p1=args.p1, p2=args.p2, p3=args.p3, frac_immune=args.immunity_frac)
        sirs.run(animate=args.animate, max_steps=args.max_steps, measure_interval=args.measure_interval, eq_steps=args.eq_steps)

    elif args.game == 'GoL':
        gol = GameOfLife(N=args.N, density=args.density, config=args.config, eq_steps=args.eq_steps)
        gol.run(animate=args.animate, max_steps=args.max_steps)

if __name__ == '__main__':
    main()