"""

2020 exam: the contact process model.

"""

# default packages
import numpy as np
import pandas as pd

# utils
from collections import deque # for history checking purposes
import argparse # for command line functionality

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

@njit
def choose_neighbour(grid: np.ndarray, pos: tuple, N: int) -> tuple:
    """
    Picks one of the nearest four neighbours
    """
    i, j = pos
    moves = np.array([[-1,0], [1,0], [0,1], [0,-1]])
    move_i, move_j = moves[np.random.randint(0, 4)]
    
    return ((i+move_i)%N, (j+move_j)%N)    

@njit
def contact_sweep(grid: np.ndarray, N: int, p: float) -> None:
    """
    Performs one full sweep of the contact process over the lattice.
    """
    n_sites = N**2

    for _ in range(n_sites): # do not use prange, process is sequential

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        pos = (i,j)

        if grid[pos] == 1:
            if np.random.random() <= (1-p):
                grid[pos] = 0
            else:
                neighbour_pos = choose_neighbour(grid=grid, pos=pos, N=N)
                grid[neighbour_pos] = 1 

class ContactProcess:
    """
    Solver for the contact process.
    """
    def __init__(self, N: int, p: float):

        self.N = N
        self.p = p
        self.grid = np.random.choice([0,1], size=(self.N, self.N)).astype(int)

        self.cmap = ListedColormap(['green', 'red'])
        self.norm = BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)

    def survival_probability_initial_condition(self):
        """
        Survival probability initial condition: one active cell.
        """
        self.grid = np.zeros(shape=(self.N, self.N))
        i = np.random.randint(0,self.N)
        j = np.random.randint(0,self.N)
        
        self.grid[i,j] = 1

    def sweep(self) -> None:
        """
        Wrapper of sweep function.
        """
        contact_sweep(self.grid, self.N, self.p)

    def _animate_sweep(self, frame) -> list:
        """
        Wraps sweep() for animation.
        """
        self.sweep()

        self.im.set_data(self.grid)
        self.ax.set_title(f'Sweep: {frame}')

        return [self.im]
    
    def run(self, animate: bool = False, max_steps: int = 10000):
        """
        Runs the contact process until the user-defined maximum number of steps is reached.
        """
        if animate:
            self.fig, self.ax = plt.subplots(figsize=(8,6))
            self.im = self.ax.imshow(self.grid, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            cbar = self.fig.colorbar(self.im, ax=self.ax, ticks=[0,1])
            cbar.ax.set_yticklabels(['Healthy', 'Infected'])
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for _ in range(max_steps):
                self.sweep()

    def grid_fractions(self) -> tuple[float, ...]:
        """
        Computes the fractions of infected and healthy on the site.
        """
        infected_fraction = np.sum((self.grid == 1)) / self.N**2
        healthy_fraction = np.sum((self.grid == 0)) / self.N**2

        return infected_fraction, healthy_fraction
       
    @staticmethod
    def fractions_over_time(p: float, N: int = 50, max_steps: int = 5000, measure_interval: int = 50) -> np.ndarray:
        """
        Computes the fractions over time for a specified value of p.
        """
        cp = ContactProcess(N=N, p=p)
        grid_fracs = []

        for i in range(max_steps):
            cp.sweep()
            if i % measure_interval == 0:
                grid_fracs.append(cp.grid_fractions())
        
        return np.array(grid_fracs)

    @staticmethod
    def survival_probability(p: float, t: int = 300) -> float:
        """
        Performs one run with survival probability condition.
        """
        cp = ContactProcess(N=50, p=p)
        alive = []

        cp.survival_probability_initial_condition()
        for _ in range(t):
            cp.sweep()
            if np.any(cp.grid == 1):
                alive.append(1)
            else:
                alive.append(0)

        return alive
    
    @staticmethod
    def task_b(max_steps: int = 10000, measure_interval: int = 100):
        """
        Task b data collection.
        """
        p_vals = [0.6, 0.7]
        results = Parallel(n_jobs=-1, return_as='list')(
                    delayed(ContactProcess.fractions_over_time)(p, max_steps=max_steps, measure_interval=measure_interval) for p in p_vals)
        t_vals = np.arange(0, max_steps, measure_interval)

        fig, axes = plt.subplots(1, 2, figsize=(15,6))

        for ax, p, fracs in zip(axes, p_vals, results):
            i_fracs, h_fracs = zip(*fracs)
            ax.plot(t_vals, i_fracs, label=f"Infected Fraction", color='r')
            ax.plot(t_vals, h_fracs, label=f"Healthy Fraction", color='g')
            ax.set_title(f"Fractions of Healthy and Infected over Time (p = {p})")
            ax.legend()
            ax.grid()
            ax.set_xlabel(f"Time [sweeps]")
            ax.set_ylabel(r"$\frac{N}{N_{tot}}$")
            pd.DataFrame({"t_vals": t_vals, "i_fracs": i_fracs, "h_fracs": h_fracs}).to_csv(f"task_b_p_{p}_data.csv", index=False)

        fig.savefig(f"task_b_graph.png", dpi=300)

        plt.show()

    @staticmethod
    def task_c(max_steps: int = 10000, measure_interval: int = 100):
        """
        Runs data collection for task c.
        """
        p_vals = np.arange(0.55, 0.705, 0.005)
        results = Parallel(n_jobs=-1, return_as='list')(
                            delayed(ContactProcess.fractions_over_time)(p, max_steps=max_steps, measure_interval=measure_interval) for p in p_vals)
        active_fracs = []
        for fracs in results:
            active_fracs.append(np.mean(fracs[:,0]))

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(p_vals, active_fracs, color='r')
        ax.set_title(f"Fraction of Active Sites against p")
        ax.set_xlabel(f"p")
        ax.set_ylabel(r"$\frac{N}{N_{tot}}$")
        ax.grid()

        fig.savefig(f"task_c_plot.png", dpi=300)
        pd.DataFrame({"p_vals": p_vals, "a_fracs": active_fracs}).to_csv(f"task_c_data.csv", index=False)
        plt.show()

    @staticmethod
    def task_d(max_steps: int = 50000, measure_interval: int = 100):
        """
        Runs data collection for task d.
        """
        p_vals = np.arange(0.55, 0.705, 0.005)
        results = Parallel(n_jobs=-1, return_as='list')(
                            delayed(ContactProcess.fractions_over_time)(p, max_steps=max_steps, measure_interval=measure_interval) for p in p_vals)
        active_vars = []
        active_var_errors = []
        bootstrap = BootstrapErrorAnalysis()
        for fracs in results:
            active_fracs = fracs[:,0]
            active_vars.append(np.var(active_fracs) * 2500) # taking var of fraction incurs another N so multiply through by 2500
            active_var_errors.append(bootstrap.calculate_errors(data=active_fracs, block_size=50) * 2500)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(p_vals, active_vars, yerr=active_var_errors, color='r')
        ax.set_title(f"Fractional Variance of Active Sites against p")
        ax.set_xlabel(f"p")
        ax.set_ylabel(r"$\frac{var(N)}{N_{tot}}$")
        ax.grid()

        fig.savefig(f"task_d_plot.png", dpi=300)
        pd.DataFrame({"p_vals": p_vals, "a_vars": active_vars, "a_var_errors": active_var_errors}).to_csv(f"task_d_data.csv", index=False)
        plt.show()

    @staticmethod
    def task_f(t: int = 300, nsims: int = 1000):
        """
        Data collection for task e.
        """
        p_vals = [0.6, 0.625, 0.65]
        t_vals = np.arange(0, t)

        p_survival_probabilities = []
        for p in p_vals:
            results = Parallel(n_jobs=-1, return_as='list')(
                        delayed(ContactProcess.survival_probability)(p, t) for _ in range(nsims)) 
            probabilities = np.vstack(results)
            survival_probabilities = np.mean(probabilities, axis=0)
            p_survival_probabilities.append(survival_probabilities)

        fig, ax = plt.subplots(figsize=(8,6))

        for p, series in zip(p_vals, p_survival_probabilities):
            ax.plot(t_vals, series, label=f"p = {p}")
            pd.DataFrame({"survival_probabilities": series, "t": t_vals}).to_csv(f"task_f_p_{p}_data.csv", index=False)

        ax.set_title(f"Survival Probability Against Time")
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.set_xlabel(f"Time [sweeps]")
        ax.set_ylabel(f"Survival Probability")
        fig.savefig(f"task_f_data.png", dpi=300)

        plt.show()

class BootstrapErrorAnalysis:
    """
    Implementation of bootstrap resampling error analysis.
    Reused some methods from CP1.
    """
    def __init__(self, k: int = 1000, seed: int = 2317434):
        self.k = k # number of times we resample
        self.generator = np.random.default_rng(seed) # my student ID

    def resample_blocks(self, data: np.ndarray, block_size: int = 50) -> tuple[np.ndarray, ...]:
        """
        Instead of resampling from the full dataset, we resample from blocks (data is correlated).
        """
        n_blocks = len(data) // block_size
        blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        chosen = self.generator.integers(0, n_blocks, size=n_blocks)
        return blocks[chosen].flatten()

    def calculate_errors(self, data: np.ndarray, block_size: int = 50) -> np.ndarray:
        """
        Compute the errors from the resampled data using block bootstrap.
        """
        data = np.array(data)
        bootstrap_variances = np.zeros(self.k)
        for i in range(self.k):
            resampled = self.resample_blocks(data, block_size) 
            bootstrap_variances[i] = np.var(resampled)
        return np.std(bootstrap_variances)


parser = argparse.ArgumentParser(description='Cellular Automata: Contact Process')

# shared arguments
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=100000)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--p', type=float, default=0.5)
parser.add_argument('--task_b', action='store_true')
parser.add_argument('--task_c', action='store_true')
parser.add_argument('--task_d', action='store_true')
parser.add_argument('--task_f', action='store_true')

def main():

    args = parser.parse_args()

    if args.task_b:
        ContactProcess.task_b()

    elif args.task_c:
        ContactProcess.task_c()

    elif args.task_d:
        ContactProcess.task_d()

    elif args.task_f:
        ContactProcess.task_f()

    else:
        cp = ContactProcess(N=args.N, p=args.p)
        cp.run(animate=args.animate, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
