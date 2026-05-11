"""

2022 exam on 3 coupled diffusive PDEs.

"""
# defaults
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# utils
from time import perf_counter
import argparse # for command line functionality

# overdrive
from numba import njit, prange
from joblib import Parallel, delayed

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import scienceplots

plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

#
# engine room
#

@njit(parallel=True)
def compute_laplacian_2D(f: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 2D Laplacian of physical parameter f on a grid.
    """
    laplacian = np.zeros_like(f)
    for i in prange(N):
        for j in range(N):

            laplacian[i, j] = (f[(i+1)%N,j] + f[(i-1)%N,j] +
                               f[i,(j+1)%N] + f[i,(j-1)%N] -
                               4*f[i,j]) / dx**2

    return laplacian

@njit
def a_step(fa: np.ndarray, fb: np.ndarray, fc: np.ndarray, N: int, 
           dx: float, dt: float, D: float, q: float, 
           p: float) -> np.ndarray:
    """
    Advances the a species one step in time.
    """
    concentration_term = (q*fa) * (1-fa-fb-fc) - (p*fa*fc)
    return fa + dt*(D*compute_laplacian_2D(f=fa, N=N, dx=dx) + concentration_term)

@njit
def b_step(fa: np.ndarray, fb: np.ndarray, fc: np.ndarray, N: int, 
           dx: float, dt: float, D: float, q: float, 
           p: float) -> np.ndarray:
    """
    Advances the b species one step in time.
    """
    concentration_term = (q*fb) * (1-fa-fb-fc) - (p*fa*fb)
    return fb + dt*(D*compute_laplacian_2D(f=fb, N=N, dx=dx) + concentration_term)

@njit
def c_step(fa: np.ndarray, fb: np.ndarray, fc: np.ndarray, N: int, 
           dx: float, dt: float, D: float, q: float, 
           p: float) -> np.ndarray:
    """
    Advances the c species one step in time.
    """
    concentration_term = (q*fc) * (1-fa-fb-fc) - (p*fb*fc)
    return fc + dt*(D*compute_laplacian_2D(f=fc, N=N, dx=dx) + concentration_term)

@njit(parallel=True)
def compute_correlation_probability(tau: np.ndarray, N: int) -> np.ndarray:
    """
    Iterates over the grid to find the correlation probability.
    """
    probabilities = np.zeros(N//2)
    for r in prange(N//2):
        matches = 0
        for i in range(N):
            matches += (np.sum(tau[:,i] == tau[:,(i+r)%N]))
        probabilities[r] = (matches / N**2)
    return probabilities

class ThreeChemicalSpecies:
    """
    Discretised solver for the 3 chemical species problem.
    """
    def __init__(self, N: int, dx: float, dt: float, D: float = 1.0, 
                 q: float = 1.0, p: float = 0.5):

        self.N = N
        self.dx = dx
        self.dt = dt
        self.time = 0.0

        self.D = D
        self.q = q
        self.p = p

        self.initialise_grids(N=N)

        self.cmap = ListedColormap(['gray', 'red', 'green', 'blue']) 
        self.norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    def initialise_grids(self, N: int):
        """
        Initialise the three grids of the chemical species.
        """
        rng = np.random.default_rng()
        
        self.a = rng.random(size=(N,N)) / 3 # random() draws from [0,1] so divide by 3
        self.b = rng.random(size=(N,N)) / 3
        self.c = rng.random(size=(N,N)) / 3

        self.tau = self.type_field(fields=(self.a, self.b, self.c))

    def step(self):
        """
        Advance all chemical species one timestep forward.
        """
        a_old, b_old, c_old = self.a.copy(), self.b.copy(), self.c.copy()
        self.a = a_step(fa=a_old, fb=b_old, fc=c_old, N=self.N,
                        dx=self.dx, dt=self.dt, D=self.D, q=self.q,
                        p=self.p)
        self.b = b_step(fa=a_old, fb=b_old, fc=c_old, N=self.N,
                        dx=self.dx, dt=self.dt, D=self.D, q=self.q,
                        p=self.p)
        self.c = c_step(fa=a_old, fb=b_old, fc=c_old, N=self.N,
                        dx=self.dx, dt=self.dt, D=self.D, q=self.q,
                        p=self.p)
        
        remaining_field = (1 - self.a - self.b - self.c)
        self.tau = self.type_field(fields=(remaining_field, self.a, self.b, self.c)) # must go in according to colourmap binding
        
        self.time += self.dt

    def type_field(self, fields: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        For each lattice site, finds the dominant species across the fields.
        """
        return np.argmax(np.stack(fields), axis=0)

    def _animate_step(self, frames: int) -> list:
        """
        Animates a step.
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(10): # increase to make animation go faster
            self.step()
        self.im.set_data(self.tau)
        self.ax.set_title(f't = {self.time:.2f}')
        return [self.im]
    
    def run(self, animate: bool, max_steps: int, save_animation: bool = False):
        """
        Iterate on the solver (animate branch included).
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.tau, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            cbar = self.fig.colorbar(mappable=self.im, ax=self.ax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(['0', '1', '2', '3'])
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            if save_animation:
                self.anim.save('task_d_animation.gif', writer='pillow', fps=30)
            plt.show()

        else:
            for _ in range(max_steps):
                self.step()

    def compute_concentration_fractions(self) -> tuple[float, ...]:
        """
        Computes the fraction of the grid occupied by each species.
        """
        a_frac = np.sum((self.tau == 1)) / self.N**2
        b_frac = np.sum((self.tau == 2)) / self.N**2
        c_frac = np.sum((self.tau == 3)) / self.N**2

        return a_frac, b_frac, c_frac

    @staticmethod
    def fractions_over_time(dt: float, max_steps: int, N: int = 50, dx: float = 1.0):
        """
        Part b) data collection: fractions over time.
        """
        tcs = ThreeChemicalSpecies(N=N, dx=dx, dt=dt, D=1.0, q=1.0, p=0.5)
        a_fracs = np.zeros(max_steps)
        b_fracs = np.zeros(max_steps)
        c_fracs = np.zeros(max_steps)
        t_vals = np.zeros(max_steps)

        for i in range(max_steps):
            tcs.step()
            a_frac, b_frac, c_frac = tcs.compute_concentration_fractions()
            a_fracs[i] = a_frac
            b_fracs[i] = b_frac
            c_fracs[i] = c_frac
            t_vals[i] = tcs.time
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(t_vals, a_fracs, label=f"a fraction", color=f"r")
        ax.plot(t_vals, b_fracs, label=f"b fraction", color=f"g")
        ax.plot(t_vals, c_fracs, label=f"c fraction", color=f"b")
        ax.legend()
        ax.set_ylabel(r"$\frac{N}{N_{tot}}$")
        ax.set_xlabel(f"Time [s]")
        ax.set_title(F"Chemical Species Fractions over Time")
        ax.grid()

        fig.savefig(f"task_b_graph.png", dpi=300)
        pd.DataFrame({'a_frac': a_fracs, 'b_frac': b_fracs, 'c_frac': c_fracs, 'time': t_vals}).to_csv('task_b_data.csv', index=False)

        plt.show()

    @staticmethod
    def absorption_time(dt: float, max_steps: int, N: int = 50, dx: float = 1.0) -> float:
        """
        Part c) data collection: time to absorption.
        """
        tcs = ThreeChemicalSpecies(N=N, dx=dx, dt=dt, D=1.0, q=1.0, p=0.5)
        for _ in range(max_steps):
            tcs.step()
            a_frac, b_frac, c_frac = tcs.compute_concentration_fractions()
            if (a_frac == 1.0) or (b_frac == 1.0) or (c_frac == 1.0):
                return tcs.time
            elif tcs.time > 1000:
                return 1001

    @staticmethod
    def absorption_time_trials() -> None:
        """
        Parallelised data collection.
        """
        n_sims = range(10)
        results = np.array(Parallel(n_jobs=-1, return_as='list')(
            delayed(ThreeChemicalSpecies.absorption_time)(dt=0.01, max_steps=1005) for n in n_sims))
        trimmed_results = results[results < 1000]

        avg_time = np.mean(trimmed_results)
        err = np.std(trimmed_results) / np.sqrt(len(trimmed_results))

        pd.DataFrame({'avg_time': [avg_time], "err": [err]}).to_csv('task_c_data.csv')

        print(f"Average Time to Absorption is {avg_time:.3f}s +/- {err:.3f}")

    @staticmethod
    def sine_model_fit(t: np.ndarray, A: np.ndarray, c: float, P: float, phi: float):
        return A*np.sin(((2*np.pi*t)/P) + phi) + c
    
    @staticmethod
    def oscillatory_behaviour(max_steps: int, eq_steps: int, measure_interval: int) -> None:
        """
        Task e) data collection.
        """
        point_1, point_2 = (10, 10), (40, 40)

        tcs = ThreeChemicalSpecies(N=50, dx=1.0, dt=0.001, D=0.5, 
                                   q=1.0, p=2.5)
        
        t_vals = []
        cell_1_vals = []
        cell_2_vals = []

        for i in range(max_steps):
            tcs.step()
            if (i > eq_steps) and (i % measure_interval == 0):
                cell_1_vals.append(tcs.a[point_1])
                cell_2_vals.append(tcs.a[point_2])
                t_vals.append(tcs.time)

        t_vals, cell_1_vals, cell_2_vals = np.array(t_vals), np.array(cell_1_vals), np.array(cell_2_vals)

        p0 = [1, 0, 25, 0]
        
        (A_1, c_1, P_1, phi_1), _ = curve_fit(tcs.sine_model_fit, t_vals, cell_1_vals, p0=p0)
        (A_2, c_2, P_2, phi_2), _ = curve_fit(tcs.sine_model_fit, t_vals, cell_2_vals, p0=p0)

        cell_1_data = tcs.sine_model_fit(t_vals, A_1, c_1, P_1, phi_1)
        cell_2_data = tcs.sine_model_fit(t_vals, A_2, c_2, P_2, phi_2)

        peaks_1, _ = find_peaks(cell_1_vals)
        peaks_2, _ = find_peaks(cell_2_vals)
        period_1 = np.mean(np.diff(t_vals[peaks_1]))
        period_2 = np.mean(np.diff(t_vals[peaks_2]))

        print(f"Period 1: {P_1:.3f} & {period_1:.3f}")
        print(f"Period 2: {P_2:.3f} & {period_2:.3f}")

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(t_vals, cell_1_vals, label="Cell 1 Data", color='r')
        ax.plot(t_vals, cell_1_data, label="Cell 1 Fit", color='r', linestyle='--')
        ax.scatter(t_vals, cell_2_vals, label="Cell 2 Data", color='b')
        ax.plot(t_vals, cell_2_data, label="Cell 2 Fit", color='b', linestyle='--')
        ax.legend()
        ax.grid()
        ax.set_ylabel(f"Value")
        ax.set_xlabel(f"Time [s]")
        ax.set_title(f"Values of Two Cells Over Time with Fits")

        fig.savefig(f"task_e_graph.png", dpi=300)

        pd.DataFrame({'period_1': [P_1], "period_2": [P_2]}).to_csv('task_e_data.csv')

        plt.show()

    @staticmethod
    def correlation_probabilities(D: float, eq_steps: int = 100000) -> None:
        """
        Find the probabilities cells have the same type field as neighbours.
        """
        tcs = ThreeChemicalSpecies(N=50, dx=1.0, dt=0.001, D=D, 
                                   q=1, p=2.5)
        for _ in range(eq_steps):
            tcs.step()
        probs = compute_correlation_probability(tau=tcs.tau, N=50)
        rs = range(25)

        return probs, rs
                                                                      
    @staticmethod
    def task_f_data() -> None:
        """
        Runs task f in parallel.
        """
        D_vals = [0.5, 0.4, 0.3]
        results = Parallel(n_jobs=-1, return_as='list')(
            delayed(ThreeChemicalSpecies.correlation_probabilities)(D) for D in D_vals)

        fig, ax = plt.subplots(figsize=(8,6))

        for (probs, rs), D in zip(results, D_vals):
            ax.plot(rs, probs, label=f"D = {D}")
            pd.DataFrame({'probabilities': probs, "distance": rs}).to_csv(f"task_f_D_{D}_data.csv")

        ax.set_ylabel(f"Probability")
        ax.set_xlabel(f"Distance [cells]")
        ax.set_title(f"Probability Cells Share Same Type Field against Distance")
        ax.legend()
        ax.grid()

        fig.savefig(f"task_f_graph.png", dpi=300)

        plt.show()


parser = argparse.ArgumentParser(description='2022 Exam')

parser.add_argument('--N', type=int, default=50)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--dx', type=float, default=1.0)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--task_b', action='store_true')
parser.add_argument('--task_c', action='store_true')
parser.add_argument('--task_d', action='store_true')
parser.add_argument('--task_e', action='store_true')
parser.add_argument('--task_f', action='store_true')
parser.add_argument('--D', type=float, default=1.0)
parser.add_argument('--q', type=float, default=1.0)
parser.add_argument('--p', type=float, default=0.5)

def main():

    args = parser.parse_args()
    
    if args.task_b:
        ThreeChemicalSpecies.fractions_over_time(dt=args.dt, max_steps=args.max_steps)
        tcs = ThreeChemicalSpecies(N=50, dx=1.0, dt=0.01, D=args.D, 
                                q=args.q, p=args.p)
        tcs.run(animate=True, max_steps=500, save_animation=True)
    elif args.task_c:
        ThreeChemicalSpecies.absorption_time_trials()
    elif args.task_d:
        tcs = ThreeChemicalSpecies(N=50, dx=1.0, dt=0.01, D=0.5, 
                                q=1.0, p=2.5)
        tcs.run(animate=True, max_steps=500, save_animation=True)
    elif args.task_e:
        ThreeChemicalSpecies.oscillatory_behaviour(max_steps=200000, eq_steps=1000, measure_interval=100)
    elif args.task_f:
        ThreeChemicalSpecies.task_f_data()
    else:
        tcs = ThreeChemicalSpecies(N=args.N, dx=args.dx, dt=args.dt, D=args.D, 
                                q=args.q, p=args.p)
        tcs.run(animate=args.animate, max_steps=args.max_steps)
    
if __name__ == "__main__":
    main()