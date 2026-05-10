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
from matplotlib.colors import BoundaryNorm
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
               p3: float) -> np.ndarray:
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

    return grid

class ParallelRPS:

    def __init__(self, N: int):

        self.N = N
        self.initialise_grid(N=N)
        self.cmap = ListedColormap(['salmon', 'steelblue', 'purple']) 
        self.norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

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
    
    def run(self, animate: bool = False, max_steps: int = 10000, measure_interval: int = 100, eq_steps: int = 100):
        """
        Runs the parallel RPS model till the max number of steps is reached.
        """
        self.rock_vals = []
        self.t_vals = []
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            cbar = self.fig.colorbar(self.im, ax=self.ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Rock', 'Paper', 'Scissors'])
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for i in range(max_steps):
                self.step()
                if i > eq_steps and i % measure_interval == 0:
                    self.rock_vals.append(self.measure_rock_evolution())
                    self.t_vals.append(i)

class SequentialRPS:

    def __init__(self, N: int, p1: float, p2: float, p3: float):

        # infection probabilities
        self.p1 = p1 # p(R -> P), rock == 0
        self.p2 = p2 # p(P -> S), paper == 1
        self.p3 = p3 # p(S -> R), scissors == 2
        
        self.N = N
        self.grid = np.random.choice([0, 1, 2], size=(N, N)).astype(int)
        self.cmap = ListedColormap(['salmon', 'steelblue', 'purple']) 
        self.norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

    def sweep(self) -> np.ndarray:
        """
        Wrapper of numba sequential sweep.
        """
        self.grid = sequential_sweep(grid=self.grid, N=self.N, p1=self.p1, p2=self.p2, p3=self.p3)

    def _animate_sweep(self, frame):
        """
        Animates a sweep of the lattice.
        """
        self.sweep()

        self.im.set_data(self.grid)
        self.ax.set_title(f'Sweep: {frame}')

        return [self.im]
    
    def run(self, animate: bool = False, max_steps: int = 10000, measure_interval: int = 1000, eq_steps: int = 100):
        """
        Runs the parallel RPS model till the max number of steps is reached.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid, cmap=self.cmap, norm=self.norm, interpolation='nearest') 
            cbar = self.fig.colorbar(self.im, ax=self.ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Rock', 'Paper', 'Scissors'])
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for _ in range(max_steps):
                self.sweep()

    def find_minority_fraction(self) -> float:
        """
        Finds which state is the minority.
        """
        least_frequent_state = np.argmin(np.bincount(self.grid.flatten(), minlength=3)) # minlength 3 accounts for an absent state!
        return np.sum((self.grid == least_frequent_state)) / self.N**2

    @staticmethod
    def minority_phase_run_d(p3: float, N: int = 50, p1: float = 0.5, p2: float = 0.5, 
                           eq_steps: int = 1500, measure_steps: int = 5000) -> tuple[np.ndarray, ...]:
        """
        Question d) data collection (joblib friendly)
        """
        rps = SequentialRPS(N=N, p1=p1, p2=p2, p3=p3)
        for _ in range(eq_steps):
            rps.sweep()
        fractions = np.zeros(measure_steps)
        for i in range(measure_steps):
            rps.sweep()
            fractions[i] = rps.find_minority_fraction()
        return np.mean(fractions), np.var(fractions)
    
    @staticmethod
    def minority_phase_run_e(p2: float, p3: float, N: int = 50, p1: float = 0.5, 
                           eq_steps: int = 1500, measure_interval: int = 100, measure_steps: int = 5000) -> tuple[np.ndarray, ...]:
        """
        Question d) data collection (joblib friendly)
        """
        rps = SequentialRPS(N=N, p1=p1, p2=p2, p3=p3)
        for _ in range(eq_steps):
            rps.sweep()
        fractions = np.zeros(measure_steps)
        for i in range(measure_steps):
            rps.sweep()
            if i % measure_interval == 0:
                fractions[i] = rps.find_minority_fraction()
        return np.mean(fractions)

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
sequential_parser.add_argument('--measure_interval', type=int, default=100)
sequential_parser.add_argument('--measure', action='store_true')

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
            fig.savefig(f"task_b_graph.png", dpi=300)
            plt.show()
            pd.DataFrame({'rock_vals': rock_vals, 't_vals': t_vals}).to_csv('task_b_data.csv')
        else:
            rps.run(animate=args.animate, max_steps=args.max_steps)

    elif args.type == 'Sequential':

        if args.measure:

            # task d data collection
            p3_vals_d = np.arange(0, 0.1, 0.005) # 20 steps between 0 and 0.1
            d_results = np.array(Parallel(n_jobs=-1, return_as='list')(
                delayed(SequentialRPS.minority_phase_run_d)(p3) for p3 in p3_vals_d))
            means_d = d_results[:,0]
            vars_d = d_results[:,1]
            fig1, ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(p3_vals_d, means_d)
            ax1.set_xlabel(r"$p_3$")
            ax1.set_ylabel(f"Mean Minority Fraction")
            ax1.grid()
            ax1.set_title(f"Mean Minority Fraction as a Function of p3 (Part d)")
            fig1.savefig(f"task_d_mean_graph.png", dpi=300)
            plt.show()
            pd.DataFrame({'minority_fracs': means_d, 'p3_vals': p3_vals_d}).to_csv('task_d_mean_data.csv')

            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.plot(p3_vals_d, vars_d)
            ax2.set_xlabel(r"$p_3$")
            ax2.set_ylabel(f"Variance of Minority Fraction")
            ax2.grid()
            ax2.set_title(f"Variance of Minority Fraction as a Function of p3 (Part d)")
            fig2.savefig(f"task_d_var_graph.png", dpi=300)
            plt.show()
            pd.DataFrame({'minority_fracs': means_d, 'p3_vals': p3_vals_d}).to_csv('task_d_var_data.csv')

            # task e) data collections
            p_vals_e = np.arange(0, 0.3, 0.02) # 15 steps between 0 and 0.3
            e_results = np.array(Parallel(n_jobs=-1, return_as='list')(
                delayed(SequentialRPS.minority_phase_run_e)(p2, p3) for p2 in p_vals_e for p3 in p_vals_e))
            phase_diagram = e_results.reshape(len(p_vals_e), len(p_vals_e))
            fig3, ax3 = plt.subplots(figsize=(8,6))
            im = ax3.imshow(phase_diagram, origin='lower', extent=[0, 0.3, 0, 0.3],
                    aspect='equal', cmap='inferno')
            fig3.colorbar(im, label=f"Minority Fraction")
            ax3.set_xlabel(r'$p_3$ (P $\to$ S)')
            ax3.set_ylabel(r'$p_2$ (S $\to$ R)')
            ax3.set_title(f"RPS Phase Diagram (p(R -> P) = 0.5)")
            fig3.savefig(f"task_e_heatmap.png", dpi=300)
            plt.show()
            pd.DataFrame(phase_diagram, index=p_vals_e, columns=p_vals_e).to_csv('task_e_data.csv')

        else:
            rps = SequentialRPS(N=args.N, p1=args.p1, p2=args.p2, p3=args.p3)
            rps.run(animate=args.animate, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
