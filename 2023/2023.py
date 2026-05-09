"""

Rock paper scissors cellular automata.

Initially parallel and deterministic (GoL), then sequential and stochastic (SIRS).

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
from tqdm import tqdm
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

def eight_neighbours(grid: np.ndarray, state: int) -> np.ndarray:
    """
    Parallel contribution sum from nearest eight neighbours (includes diagonals).
    """
    neighbour_sum = np.zeros_like(grid)
    
    for dx, dy in eight_shifts:
        neighbour_sum += np.roll(np.roll((grid==state), dx, axis=0), dy, axis=1)

    return neighbour_sum

@njit
def sequential_sweep(grid: np.ndarray, N: int, p1: float, p2: float, 
               p3: float) -> None:
    """
    Sequential, stochastic sweep over grid.
    """
    n_sites = N**2
    # pregenerate random sites
    sites_i = np.random.randint(0, N, size=n_sites)
    sites_j = np.random.randint(0, N, size=n_sites)
    randoms = np.random.random(n_sites)

    for k in range(n_sites):
        i = sites_i[k]
        j = sites_j[k]

        # p1: rock -> paper
        if grid[i, j] == 0:
            for m in range(8): # check across eight neighbours
                ni = (i + eight_shifts[m,0]) % N
                nj = (j + eight_shifts[m,1]) % N
                if grid[ni,nj] == 1: # 1: neighbour is paper
                    if randoms[k] < p1:
                        grid[i,j] = 1
                    break # condition is 'at least' so avoid multiple checks 

        # p2: paper -> scissors
        elif grid[i,j] == 1:
            for m in range(8): # check across eight neighbours
                ni = (i + eight_shifts[m,0]) % N
                nj = (j + eight_shifts[m,1]) % N
                if grid[ni,nj] == 2: # 1: neighbour is scissors
                    if randoms[k] < p2:
                        grid[i,j] = 2
                    break 

        # p3: scissors -> rock
        elif grid[i,j] == 2:
            for m in range(8): # check across eight neighbours
                ni = (i + eight_shifts[m,0]) % N
                nj = (j + eight_shifts[m,1]) % N
                if grid[ni,nj] == 0: # 1: neighbour is rock
                    if randoms[k] < p3:
                        grid[i,j] = 0
                    break 

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

class SequentialRPS:

    def __init__(self, N: int, p1: float, p2: float, p3: float):

        # infection probabilities
        self.p1 = p1 # p(R -> P), rock == 0
        self.p2 = p2 # p(P -> S), paper == 1
        self.p3 = p3 # p(S -> R), scissors == 2
        
        self.N = N
        self.initialise_grid(N=N)
        self.cmap = ListedColormap(['blue', 'red', 'green', 'white']) # white: immune

class ParallelRPS:

    def __init__(self, N: int):

        self.N = N
        self.initialise_grid(N=N)
        self.cmap = ListedColormap(['blue', 'red', 'green', 'white']) # white: immune

    def initialise_grid(self, N):
        """
        Generate grid with distinct pie wedges shape.
        """
        x_grid, y_grid = np.meshgrid(np.arange(N), np.arange(N)) 
        x_grid, y_grid = x_grid - N//2, y_grid - N//2 # N//2 centres the grid
        angles = np.atan2(y_grid, x_grid) # atan2 is the vector one
        # then divide the grid into the 'pie wedges' by 3 equal solid angles
        rock = (angles <= -np.pi/3)
        paper = (angles > -np.pi/3) & (angles <= np.pi/3)
        scissors = (angles >= np.pi/3)

        self.grid = 0*rock + 1*paper + 2*scissors # then apply the convention

    def step(self) -> np.ndarray:
        """
        One global application of the RPS rules (parallel, deterministic).
        """
        current_state = self.grid.copy()

        rock_neighbour_sum = eight_neighbours(current_state, 0) 
        paper_neighbour_sum = eight_neighbours(current_state, 1) 
        scissors_neighbour_sum = eight_neighbours(current_state, 2)

        # boolean logic
        rock_to_paper = (current_state == 0) & (paper_neighbour_sum > 2) # rock cell becomes paper if it is dead and has more than two live neighbours
        paper_to_scissors = (current_state == 1) & (scissors_neighbour_sum > 2) # same rule thereafter
        scissors_to_rock = (current_state == 2) & (rock_neighbour_sum > 2)

        no_change = ~rock_to_paper & ~paper_to_scissors & ~scissors_to_rock # where these arrays = false

        # store the boolean values of surviving/reborn cells
        self.grid = rock_to_paper*1 + paper_to_scissors*2 + scissors_to_rock*0 + no_change*current_state

    def _animate_step(self, frame):
        """
        Animates a sweep of the lattice.
        """
        self.step()

        self.im.set_data(self.grid)
        self.ax.set_title(f'Step: {frame}')

        return [self.im]
    
    def measure_rock_evolution(self) -> int:
        """
        Helper function to check whether the centre cell is a rock.
        """
        return int((self.grid[self.N//2, self.N//2] == 0))
    
    def run(self, animate: bool = False, max_steps: int = 10000, measure_interval: int = 1000, eq_steps: int = 100):
        """
        Runs the parallel RPS model till the max number of steps is reached.
        """
        self.rock_vals = []
        self.t_vals = []
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid, cmap=self.cmap, 
                          vmin=0, vmax=3, interpolation='nearest') # remember to increase vmax for new sites
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for i in range(max_steps):
                self.step()
                if i > eq_steps:
                    self.rock_vals.append(self.measure_rock_evolution())
                    self.t_vals.append(i)

parser = argparse.ArgumentParser(description='Cellular Automata: RPS Models')

# shared arguments
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--eq_steps', type=int, default=10)

subparsers = parser.add_subparsers(dest='type', required=True)

# parallel arguments (not technically necessary)
parallel_parser = subparsers.add_parser('Parallel')
parallel_parser.add_argument('--measure', action='store_true')
parallel_parser.add_argument('--measure_interval', type=int, default=10)

# sequential arguments
sequential_parser = subparsers.add_parser('Sequential')
sequential_parser.add_argument('--p1', type=float, required=True, help='p(S->I)')
sequential_parser.add_argument('--p2', type=float, required=True, help='p(I->R)')
sequential_parser.add_argument('--p3', type=float, required=True, help='p(R->S)')
sequential_parser.add_argument('--immunity_frac', type=float, default=0.0)
sequential_parser.add_argument('--measure_interval', type=int, default=100)

def main():
    args = parser.parse_args()

    if args.type == 'Parallel':
        rps = ParallelRPS(N=args.N)
        if args.measure:
            rps.run(animate=False, max_steps=args.max_steps)
            rock_vals = rps.rock_vals
            t_vals = rps.t_vals

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(t_vals, rock_vals)
            ax.set_xlabel(r"$N_{iters}$")
            ax.set_ylabel(r"$R$")
            ax.grid()
            ax.set_title(f"Evolution of Central Point over Time (Parallel)")
            fig.savefig(f"cell_rock_evolution.png", dpi=300)
            plt.show()
            pd.DataFrame({'rock_vals': rock_vals, 't_vals': t_vals}).to_csv('cell_rock_evolution_data.csv')
        else:
            rps.run(animate=args.animate, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
