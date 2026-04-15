"""

Monte Carlo Methods (Ising Model).

Top functions are the numba engine room which handles agnostic computations.
I like to work with classes, so I will make a bootstrap error class and a specific ising model class.

"""

import numpy as np 
from numba import njit, prange
import pandas

import argparse
from time import perf_counter
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots

plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

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

class BootstrapErrorAnalysis:
    """
    Implementation of bootstrap resampling error analysis.
    """
    def __init__(self, k: int = 1000):

        self.k = k # number of times we resample

    def resample(self, data: np.ndarray):
        """
        Randomly resamples from provided data.
        """
        n = len(data)
        indices = np.random.randint(0, n, size=n) # pick random indices to form resample
        return data[indices]
    
    def error(self, data: np.ndarray, observable: callable, *args: tuple) -> np.ndarray:
        """
        Unused function, but potentially useful for exams. Computes error for any defined observable.
        To use: define separate observable function and pass to this with corresponding data.
        """
        values = np.empty(self.k)
        n = len(data)
        for j in range(self.k):
            indices = np.random.randint(0, n, size=n)
            values[j] = observable(data[indices], *args)
        return np.std(values, ddof=1)
    
    def calculate_errors(self, magnetisation, energy, n_sites, beta):
        """
        Standard Ising Model errors
        """
        results = np.empty((self.k, 4)) # kx4 array at once instead of a loop
        for j in range(self.k):
            M_sample = self.resample(magnetisation)
            E_sample = self.resample(energy)
            results[j] = MonteCarloIsing.calculate_observables(M_sample, E_sample, n_sites, beta)
        
        errors = np.std(results, axis=0, ddof=1) # sample of population: lose a degree of freedom (ddof=1)
        return errors
    
class MonteCarloIsing:
    """
    Ising Model Monte Carlo algorithm with engine room., for reference.
    """
    def __init__(self, N: int, T: float, lattice_config: str = 'random',
                 dynamics: str = 'glauber'):

        self.N = N
        self.beta = 1 / T # easier to input T and work in beta
        self.initialise_grid(lattice_config=lattice_config)
        self.dynamics = dynamics

    def initialise_grid(self, lattice_config: str = 'random') -> np.ndarray:
        """
        Generate a grid according to the user-defined configuration..
        """
        if lattice_config == 'random':
            self.lattice = np.random.choice([-1,1], (self.N, self.N)) # remember dtype=int
        elif lattice_config == 'all-up':
            self.lattice = np.ones((self.N, self.N), dtype=int)

    def update_beta(self, T):
        """
        Quick helper function to externally update beta.
        """
        self.beta = 1 / T

    def sweep(self) -> np.ndarray:
        """
        Wraps the chosen dynamics.
        """
        if self.dynamics == 'glauber':
            glauber_sweep(self.lattice, self.N, self.beta)
        elif self.dynamics == 'kawasaki':
            kawasaki_sweep(self.lattice, self.N, self.beta)

    def run(self, n_sweeps: int, measure_interval: int = 10, eq_sweeps: int = 10,
            animate: bool = False) -> None:
        """
        Run the Monte Carlo simulation with optional animation.
        """
        self.magnetisations = []
        self.energies = []
        self.iters = 0

        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.lattice, cmap='viridis', vmin=-1, vmax=1, 
                        interpolation='nearest')
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=n_sweeps // 50, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for _ in range(n_sweeps):
                self.sweep()
                self.iters += 1
                if self.iters >= eq_sweeps and self.iters % measure_interval == 0:
                    self.magnetisations.append(total_magnetisation(self.lattice))
                    self.energies.append(total_energy_ising(self.lattice, self.N))                

    def _animate_sweep(self, frames: int) -> list:
        """
        Animates one sweep of the lattice.
        """
        self.sweep()
        self.iters += 1
        self.im.set_data(self.lattice)
        self.ax.set_title(r"$N_{sweeps} = $" + f"{self.iters}")
        return [self.im]
    
    @staticmethod
    def calculate_observables(magnetisation: np.ndarray, energy: np.ndarray, n_sites: int, beta: float) -> tuple[np.ndarray, ...]:
        """
        Computes pertinent observables.
        """
        avg_magnetisation = np.mean(np.abs((magnetisation))) # avg(abs): lecture notes 1 pg11
        magnetisation_var = np.var(magnetisation) # variance (for susceptibility calculation)

        avg_energy = np.mean(energy)
        energy_var = np.var(energy)

        susceptibility = (beta / n_sites) * magnetisation_var
        cv_per_spin = (beta**2 / n_sites) * energy_var

        return avg_magnetisation, susceptibility, avg_energy, cv_per_spin

parser = argparse.ArgumentParser(description='Monte Carlo Methods')

# command line arguments
parser.add_argument('--N', type=int, default=50, 
                    help='Lattice NxN size (default: 50)')
parser.add_argument('--temp', type=float, required=True,
                    help='Temperature kT')
parser.add_argument('--dynamics', type=str, choices=['glauber', 'kawasaki'],
                    required=True, help='Dynamics to simulate (Kawaski/Glauber)')
parser.add_argument('--n_sweeps', type=int, default=10000,
                    help='Number of sweeps (default: 10000)')
parser.add_argument('--measure_interval', type=int, default=10,
                    help='Number of steps between measurements (default: 10)')
parser.add_argument('--animate', action='store_true',
                    help='Show live animation')
parser.add_argument('--sweep-direction', type=str, choices=['up', 'down'],
                    default='up', help='Decide in which direction the temperature is swept')
parser.add_argument('--save_data', action='store_true',
                    help='Save plots and datafile')
parser.add_argument('--lattice_config', type=str, choices=['random', 'all-up'],
                    default='all-up', help='Lattice structure to initialise')

def main():
    args = parser.parse_args()

    T0 = args.temp
    T_min = 1.0
    T_max = 3.0
    step = 0.1

    if args.sweep_direction == 'up': # sweep temperature depending on user request
        temperatures = np.arange(T0, T_max + step, step) # sweep upwards
    else:
        temperatures = np.arange(T0, T_min - step, -step) # sweep downwards
    
    mc = MonteCarloIsing(args.N, T0, lattice_config=args.lattice_config, dynamics=args.dynamics)

    if args.save_data:
        
        bootstrap = BootstrapErrorAnalysis(k=1000)
        observables = np.empty((len(temperatures), 4))
        errors = np.empty((len(temperatures), 4))

        for idx, T in enumerate(temperatures):
            t0 = perf_counter()
            mc.update_beta(T)
            if args.lattice_config == 'random':
                if idx == 0:  # initial temperature
                    eq_sweeps = args.n_sweeps // 2
                    print(f"  (Using {eq_sweeps} equilibration sweeps for first temperature)")
                else:  # normal for ensuing sweeps
                    eq_sweeps = args.n_sweeps // 100
            else: # not for case of all-up
                eq_sweeps = args.n_sweeps // 100

            mc.run(n_sweeps=args.n_sweeps, eq_sweeps=eq_sweeps,
                measure_interval=10, animate=args.animate)

            M = np.array(mc.magnetisations)
            E = np.array(mc.energies)
            n_sites = args.N**2 # this is important, n_sites and N are not the same

            observables[idx] = mc.calculate_observables(M, E, n_sites=n_sites, beta=mc.beta)
            errors[idx] = bootstrap.calculate_errors(M, E, n_sites=n_sites, beta=mc.beta)

            print(f"\n T = {T:.2f}, ⟨|M|⟩ = {observables[idx,0]:.4f}, "
                f"χ = {observables[idx,1]:.4f}, Cv = {observables[idx,3]:.4f} \n"
                f"Took {(perf_counter() - t0):.3f} seconds.")
            
        labels = [("⟨|M|⟩", "Magnetisation"), ("χ", "Susceptibility"),
            ("⟨E⟩", "Energy"), ("Cv", "Heat Capacity")]

        for col, (ylabel, title) in enumerate(labels):
            plt.figure()
            plt.errorbar(temperatures, observables[:, col], yerr=errors[:, col], marker='o')
            plt.xlabel(r"$\text{Temperature} T$")
            plt.ylabel(ylabel)
            plt.title(f"{args.dynamics.capitalize()} {title}")
            plt.grid()
            if args.save_data:
                plt.savefig(f"{args.dynamics}_{title.lower().replace(' ','_')}.png", dpi=300)
            plt.show()

        df = pandas.DataFrame(
        np.column_stack([temperatures, observables, errors]),
        columns=['T', '|M|', 'chi', 'E', 'cv', '|M|_err', 'chi_err', 'E_err', 'cv_err']
        )
        df.to_csv(f"{args.dynamics}_results_N{args.N}.csv", index=False)

    else:
        mc.run(n_sweeps=args.n_sweeps, measure_interval=10, eq_sweeps=100,
               animate=args.animate)

if __name__ == "__main__":
    main()