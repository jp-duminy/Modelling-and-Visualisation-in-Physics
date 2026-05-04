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
def metropolis_acceptance(delta_E: float) -> bool:
    """
    Returns Metropolis algorithm acceptance criterion.
    """
    if delta_E <= 0:
        return True
    return np.random.random() < np.exp(-delta_E) # exam stipulates we may set beta = 1

@njit
def total_energy_ising(lattice: np.ndarray, N: int, h: np.ndarray) -> int:
    """
    Total energy for an Ising model lattice, where spins are +/- 1.
    """
    E = 0.0
    for i in prange(N):
        for j in range(N):
            E += (lattice[i,j] * (lattice[i,(j+1) % N] + lattice[(i+1) % N,j])) - (h[i,j] * lattice[i,j]) # right and top neighbour (avoid double counting)
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
def glauber_sweep(lattice: np.ndarray, h: np.ndarray, N: int) -> None:
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
        delta_E = 2 * (-1*(sigma + h[i,j])) * spin

        if metropolis_acceptance(delta_E):
            lattice[i,j] *= -1

@njit
def staggered_magnetisation(lattice: np.ndarray, N: int) -> int:
    """
    Returns staggered magnetisation of input lattice.
    """
    staggered_magnetisation = 0.0
    for i in range(N):
        for j in range(N):
            staggered_magnetisation += lattice[i,j] * (-1)**(i + j)
    return staggered_magnetisation

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
    
    def calculate_errors(self, magnetisation: np.ndarray, energy: np.ndarray, stag_magnetisation: np.ndarray) -> np.ndarray:
        """
        Standard Ising Model errors.
        """
        results = np.empty((self.k, 5)) # kx5 array at once instead of a loop
        for j in range(self.k):
            M_sample = self.resample(magnetisation)
            stag_M_sample = self.resample(stag_magnetisation)
            E_sample = self.resample(energy)
            results[j] = MonteCarloIsing.calculate_observables(M_sample, stag_M_sample, E_sample)
        
        errors = np.std(results, axis=0, ddof=1) # sample of population: lose a degree of freedom (ddof=1)
        return errors

@njit
def dynamic_h(h: np.ndarray, N: int, h0: int, n: int, P: int, tau: int) -> np.ndarray:
    """
    Updates the magnetic field (part d)
    """
    for i in range(N):
        for j in range(N):
            h[i,j] = h0 * np.cos((2*np.pi*i)/P) * np.cos(2*np.pi*j/P) * np.sin(2*np.pi*n/tau)    
    return h    
        
class MonteCarloIsing:
    """
    Ising Model Monte Carlo algorithm with engine room., for reference.
    """
    def __init__(self, N: int, lattice_config: str = 'all-up', initial_h: int = 1, dynamics: str = 'glauber',
                 dynamic_h: bool = False):

        self.N = N
        self.initialise_grid(initial_h=initial_h, lattice_config=lattice_config)
        self.dynamics = dynamics
        self.dynamic_h = dynamic_h

    def initialise_grid(self, initial_h: int, lattice_config: str = 'random') -> np.ndarray:
        """
        Generate a grid according to the user-defined configuration..
        """
        if lattice_config == 'random':
            self.lattice = np.random.choice([-1,1], (self.N, self.N)) # remember dtype=int
        elif lattice_config == 'all-up':
            self.lattice = np.ones((self.N, self.N), dtype=int)
        self.h = np.full(shape=(self.N, self.N), fill_value=initial_h)

    def update_h(self, h: float) -> np.ndarray:
        """
        Quick helper function to externally update the magnetic field.
        """
        self.h = np.full(shape=(self.N, self.N), fill_value=h)

    def sweep(self) -> np.ndarray:
        """
        Wraps the chosen dynamics.
        """
        glauber_sweep(self.lattice, self.h, self.N)

    def run(self, n_sweeps: int, measure_interval: int = 10, P: int = 25, eq_sweeps: int = 10,
            animate: bool = False) -> None:
        """
        Run the Monte Carlo simulation with optional animation.
        """
        self.magnetisations = []
        self.staggered_magnetisations = []
        self.energies = []
        if self.dynamic_h:
            self.max_field_strength = []
        self.iters = 0
        snapshot_sweeps = [2500, 5000, 7500]

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
                if self.dynamic_h:
                    self.h = dynamic_h(self.h, N=self.N, h0=10, n=self.iters, P=P, tau=10000)
                if self.iters >= eq_sweeps and self.iters % measure_interval == 0:
                    self.magnetisations.append(total_magnetisation(self.lattice))
                    self.staggered_magnetisations.append(staggered_magnetisation(self.lattice, self.N))
                    self.energies.append(total_energy_ising(self.lattice, self.N, self.h))
                    if self.dynamic_h:
                        max_field = 10*np.sin((2*np.pi*self.iters)/10000)
                        self.max_field_strength.append(max_field)       
                        if self.iters in snapshot_sweeps:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.imshow(self.lattice, cmap='viridis', vmin=-1, vmax=1)
                            ax.set_title(f"Sweep {self.iters}, sin = {np.sin(2*np.pi*self.iters/10000):.2f}")
                            fig.savefig(f"snapshot_sweep_{self.iters}.png", dpi=300, bbox_inches='tight')
                            plt.close(fig)

    def _animate_sweep(self, frames: int) -> list:
        """
        Animates one sweep of the lattice.
        """
        self.sweep()
        self.iters += 1
        if self.dynamic_h:
            self.h = dynamic_h(self.h, N=self.N, h0=10, n=self.iters, P=25, tau=10000)
        self.im.set_data(self.lattice)
        self.ax.set_title(r"$N_{sweeps} = $" + f"{self.iters}")
        return [self.im]
    
    @staticmethod
    def calculate_observables(magnetisation: np.ndarray, stag_magnetisation: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Computes pertinent observables.
        """
        avg_magnetisation = np.mean(np.abs((magnetisation))) # avg(abs): lecture notes 1 pg11
        avg_stag_magnetisation = np.mean(np.abs((stag_magnetisation)))
        magnetisation_var = np.var(magnetisation) # variance (for susceptibility calculation)
        stag_magnetisation_var = np.var(stag_magnetisation)

        avg_energy = np.mean(energy)

        return avg_magnetisation, avg_stag_magnetisation, magnetisation_var, stag_magnetisation_var, avg_energy

parser = argparse.ArgumentParser(description='Monte Carlo Methods')

# command line arguments
parser.add_argument('--N', type=int, default=50, 
                    help='Lattice NxN size (default: 50)')
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
parser.add_argument('--h', type=float, required=True, default=1, help='Initial magnetic field value')
parser.add_argument('--dynamic_h', default=False, action='store_true',
                    help="Whether to update h dynamically (parts d) and e))")

def main():

    args = parser.parse_args()

    h0 = args.h
    h_min = 0.0
    h_max = 10.0
    step = 0.5

    if args.sweep_direction == 'up': # sweep temperature depending on user request
        h_vals = np.arange(h0, h_max + step, step) # sweep upwards
    else:
        h_vals = np.arange(h0, h_min - step, -step) # sweep downwards
    
    mc = MonteCarloIsing(args.N, lattice_config=args.lattice_config, initial_h=h0, dynamics=args.dynamics,
                         dynamic_h = args.dynamic_h)

    if args.save_data and not args.dynamic_h:
        
        bootstrap = BootstrapErrorAnalysis(k=1000)
        observables = np.empty((len(h_vals), 5))
        errors = np.empty((len(h_vals), 5))

        for idx, h in enumerate(h_vals):
            t0 = perf_counter()
            mc.update_h(h)
            if args.lattice_config == 'random':
                if idx == 0:  # initial h
                    eq_sweeps = args.n_sweeps // 2
                    print(f"  (Using {eq_sweeps} equilibration sweeps for first h)")
                else:  # normal for ensuing sweeps
                    eq_sweeps = args.n_sweeps // 10
            else: # not for case of all-up
                eq_sweeps = args.n_sweeps // 10

            mc.run(n_sweeps=args.n_sweeps, eq_sweeps=eq_sweeps,
                measure_interval=10, animate=args.animate)

            M = np.array(mc.magnetisations)
            stag_M = np.array(mc.staggered_magnetisations)
            E = np.array(mc.energies)
            n_sites = args.N**2 # this is important, n_sites and N are not the same

            observables[idx] = mc.calculate_observables(M, stag_M, E)
            errors[idx] = bootstrap.calculate_errors(M, stag_M, E)

            print(f"\n h = {h:.2f}, ⟨|M|⟩ = {observables[idx,0]:.4f}, "
                f"⟨|Ms|⟩ = {observables[idx,1]:.4f}, var(M) = {observables[idx,2]:.4f} \n"
                f"var(Ms) = {observables[idx,3]:.4f}, E = {observables[idx,4]:.4f} \n"
                f"Took {(perf_counter() - t0):.3f} seconds.")
            
        labels = [("⟨|M|⟩", "Magnetisation"), ("⟨|Ms|⟩", "Staggered Magnetisation"),
                ("var(M)", "var(M)"), ("var(Ms)", "var(Ms)"), ("⟨E⟩", "Energy")]

        for col, (ylabel, title) in enumerate(labels):
            plt.figure(figsize=(8, 6))
            plt.errorbar(h_vals, observables[:, col], yerr=errors[:, col], marker='o')
            plt.xlabel(r"$\text{Magnetic Field} h$")
            plt.ylabel(ylabel)
            plt.title(f"{args.dynamics.capitalize()} {title}")
            plt.grid()
            if args.save_data:
                plt.savefig(f"{args.dynamics}_{title.lower().replace(' ','_')}.png", dpi=300)
            plt.show()

        df = pandas.DataFrame(
        np.column_stack([h_vals, observables, errors]),
        columns=['h', '|M|', '|Ms|', 'var(M)', 'var(Ms)', 'E', 
                 '|M|_err', '|Ms|_err', 'var(M)_err', 'var(Ms)_err', 'E_err',]
        )
        df.to_csv(f"{args.dynamics}_results_N{args.N}.csv", index=False)
    elif args.dynamic_h and args.save_data:
        eq_sweeps = 5000
        P_vals = [25, 10]
        for p in P_vals:
            mc.initialise_grid(initial_h=args.h, lattice_config='random')
            mc.run(n_sweeps=args.n_sweeps, eq_sweeps=eq_sweeps,
                measure_interval=10, animate=False, P=p)
            staggered_mag_vals = mc.staggered_magnetisations
            t_vals = np.arange(len(staggered_mag_vals))
            max_field_vals = mc.max_field_strength

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(t_vals, staggered_mag_vals, label="|Ms|")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Ms")
            ax.grid()
            ax.set_title("Staggered Magnetisation against Time")
            fig.savefig(f"p{p}_ms_time.png", dpi=300)
            plt.show()

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(max_field_vals, staggered_mag_vals, label="Max Field Strength")
            ax.set_xlabel("Max Field Strength")
            ax.set_ylabel("|Ms|")
            ax.set_title("Staggered Magnetisation against Maximum Field Strength")
            ax.grid()
            fig.savefig(f"p{p}_ms_fieldstrength.png", dpi=300)
            plt.show()

            df = pandas.DataFrame(
            np.column_stack([staggered_mag_vals, max_field_vals, t_vals]),
            columns=['Ms', 'maxfield', 't']
            )
            df.to_csv(f"dynamic_h_p{p}_results.csv", index=False)
            
    else:
        mc.run(n_sweeps=args.n_sweeps, measure_interval=10, eq_sweeps=100,
               animate=args.animate)

if __name__ == "__main__":
    main()