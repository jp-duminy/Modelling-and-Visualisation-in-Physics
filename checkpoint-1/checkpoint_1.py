import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import pandas as pd
import scienceplots

plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

class Grid:
    """
    Functions that initialise a grid and provides objects for grid operations.
    """
    def __init__(self, nx, ny, beta, lattice_config='all-up', seed=2317434):
        self.x = nx # width of grid
        self.y = ny # height of grid
        self.beta = beta # thermal energy beta = 1/kT
        self.lattice_config = lattice_config

        self.generator = np.random.default_rng(seed) # seed is my student ID

        self.lattice = None
        self.glauber_prob_cache = None
        self.kawasaki_prob_cache = None

    def cached_probabilities(self):
        """
        Stores the discrete probabilities of possible +ve energy changes.
        This should avoid repeatedly calling np.exp which could be computationally expensive.
        """
        # glauber cache
        self.glauber_prob_cache = {}
        for E in [4, 8]: # energy change takes discrete values
            self.glauber_prob_cache[E] = np.exp(-self.beta * E) # so cache these probabilities

        # kawasaki cache, same as glauber but with different energy values
        self.kawasaki_prob_cache = {}
        for E in range(4, 20, 4):
            self.kawasaki_prob_cache[E] = np.exp(-self.beta * E)

    def prepare(self, reinitialise=False):
        """
        Populates the lattice and caches the Boltzmann factors.
        Must be explicitly called before running either algorithm.
        """
        if self.lattice is None or reinitialise:
            if self.lattice_config == 'random':
                self.lattice = self.generator.choice([-1, 1], (self.y, self.x))
            else:
                self.lattice = np.ones((self.y, self.x))

        if self.glauber_prob_cache is None or self.kawasaki_prob_cache is None:
            self.cached_probabilities()

    def update_beta(self, beta):
        """
        Updates beta so the probability caches are reinitialised.
        """
        self.beta = beta
        self.cached_probabilities()

    def nearest_neighbours(self, pos):
        """
        Find sum of spins of nearest neighbours.
        """
        i, j = pos

        neighbour_sum = (self.lattice[(i - 1) % self.y, j] + # up
                        self.lattice[(i + 1) % self.y, j] + # down
                        self.lattice[i, (j - 1) % self.x] + # left
                        self.lattice[i, (j + 1) % self.x]) # right

        return neighbour_sum
    
    def flip_spin(self, pos):
        """
        Checks the effect of flipping the spin of a given site.
        """
        spin = self.lattice[pos]
        sigma = self.nearest_neighbours(pos)

        delta_E = 2 * sigma * spin # should take a discrete value to work with the cache

        if delta_E <= 0:
            self.lattice[pos] *= -1
        else:
            if self.generator.random() < self.glauber_prob_cache[delta_E]: # see function cached_probabilities for explanation
                self.lattice[pos] *= -1

    def neighbour_check(self, pos1, pos2) -> bool:
        """
        Check if two points on the lattice are neighbours (vertically or horizontally).
        """
        i1, j1 = pos1
        i2, j2 = pos2

        # distances in y and x directions
        di = min(abs(i1 - i2), self.y - abs(i1 - i2)) # min() method handles periodic boundary conditions
        dj = min(abs(j1 - j2), self.x - abs(j1 - j2))

        return (di == 1 and dj == 0) or (di == 0 and dj == 1) # distance must only be 1 (ignore diagonals)

    def swap_spin(self, pos1, pos2):
        """
        Checks the effect of swapping two spin sites (for Kawasaki algorithm).
        """
        spin1 = self.lattice[pos1]
        spin2 = self.lattice[pos2]

        # ignore condition if the spins are identical (no effect)
        if spin1 == spin2:
            return
        
        sigma1 = self.nearest_neighbours(pos1)
        delta_E1 = 2 * sigma1 * spin1

        sigma2 = self.nearest_neighbours(pos2)
        delta_E2 = 2 * sigma2 * spin2

        if self.neighbour_check(pos1, pos2): # if they are nearest neighbours
            delta_E2 = 2 * (sigma2 - spin1) * spin2 # simply subtract the flipped spin from the sigma term
            delta_E1 = 2 * (sigma1 - spin2) * spin1
        
        # metropolis algorithm
        delta_E = delta_E1 + delta_E2
        if delta_E <= 0:
            self.lattice[pos1], self.lattice[pos2] = self.lattice[pos2], self.lattice[pos1] # flip the spins
        else:
            if self.generator.random() < self.kawasaki_prob_cache[delta_E]:
                self.lattice[pos1], self.lattice[pos2] = self.lattice[pos2], self.lattice[pos1]

    def magnetisation(self):
        """
        Returns the magnetisation (sum of the spins of the lattice)
        """
        return np.sum(self.lattice)
    
    def energy(self):
        """
        Returns the energy of the entire lattice.
        """
        E = 0.0

        for i in range(self.y):
            for j in range(self.x):
                spin = self.lattice[i, j]

                E += -spin * (self.lattice[i, (j + 1) % self.x] 
                              + self.lattice[(i + 1) %  self.y, j]) # right and above neighbour (avoid double counting)
        
        return E
    
class MonteCarloAlgorithm:
    """
    Base Monte Carlo class for Glauber & Kawasaki algorithms (since both use the same logic)
    """
    def __init__(self, grid):
        self.generator = grid.generator # generator object contains random number methods

        self.grid = grid
        grid.prepare()
    
    def sweep(self):
        """
        Computes one sweep on the lattice.
        """
        n_sites = self.grid.x * self.grid.y
        for _ in range(n_sites):
            self.step()
    
    def run(self, n_sweeps, animate=False, ax=None, update_interval=10,
            measure=False, eq_sweeps=100, measure_interval=10):
        """
        Runs the algorithm for n sweeps with optional animation.
        """

        magnetisation = []
        energy = []
            
        for sweep_num in range(n_sweeps):
            self.sweep()

            if measure and sweep_num >= eq_sweeps and (sweep_num - eq_sweeps) % measure_interval == 0:
                M = self.grid.magnetisation()
                E = self.grid.energy()
                magnetisation.append(M)
                energy.append(E)
            
            if animate and ax is not None and sweep_num % update_interval == 0:
                ax.clear()
                ax.imshow(self.grid.lattice, cmap='viridis', vmin=-1, vmax=1, 
                         interpolation='nearest')
                ax.set_title(f'Sweep {sweep_num}/{n_sweeps}')
                plt.pause(0.01)
            
        if measure:
            return np.array(magnetisation), np.array(energy)
        else:
            return None, None

class GlauberAlgorithm(MonteCarloAlgorithm):
    """
    Glauber algorithm for the Ising model.
    """
    def step(self):
        """
        Picks a random site on the lattice and attempts to flip it.
        """
        i = self.generator.integers(0, self.grid.y) # rows
        j = self.generator.integers(0, self.grid.x) # columns
        pos = (i, j)

        self.grid.flip_spin(pos)

class KawasakiAlgorithm(MonteCarloAlgorithm):
    """
    Computes the Kawasaki algorithm. 
    This implements case i) from the checkpoint where we do two flips and account for nearest neighbour case.
    """
    def step(self):
        """
        Picks two random sites on the lattice and attempts to swap them.
        """
        # first site
        i1 = self.generator.integers(0, self.grid.y) # rows
        j1 = self.generator.integers(0, self.grid.x) # columns
        
        # second site
        i2, j2 = i1, j1 # at first, make the second site equal to the first site
        while (i2, j2) == (i1, j1): # then use a while loop to change them to a different site
            i2 = self.generator.integers(0, self.grid.y)
            j2 = self.generator.integers(0, self.grid.x) # while loop ensures the same site is not randomly picked again

        pos1, pos2 = (i1, j1), (i2, j2)
        
        self.grid.swap_spin(pos1, pos2)

class BootstrapErrorAnalysis:
    """
    Implementation of bootstrap resampling error analysis.
    """
    def __init__(self, k=1000, seed=2317434):
        self.k = k # number of times we resample
        self.generator = np.random.default_rng(seed) # my student ID

    def resample(self, data):
        """
        Randomly resamples from provided data.
        """
        n = len(data)
        indices = self.generator.integers(0, n, size=n) # pick random indices to form resample
        return data[indices]
    
    def calculate_errors(self, magnetisation, energy, N, beta):
        """
        Uses resampled data to compute errors.
        """
        susceptibilities = []
        energies = []
        magnetisations = []
        cvs = []

        # resample 1000 times and recompute observables 
        for _ in range(self.k):
            M_sample = self.resample(magnetisation)
            E_sample = self.resample(energy)

            resampled_M, resampled_susceptibility, resampled_energy, resampled_cv = calculate_observables(M_sample,
                                                                                                        E_sample, N, beta)
            magnetisations.append(resampled_M)
            susceptibilities.append(resampled_susceptibility)
            energies.append(resampled_energy)
            cvs.append(resampled_cv)
        
        # these are resamples so we lose a degree of freedom, ddof=1 (sample vs population distributions)
        susceptibility_err = np.std(susceptibilities, ddof=1)
        energies_err = np.std(energies, ddof=1)
        magnetisations_err = np.std(magnetisations, ddof=1)
        cvs_err = np.std(cvs, ddof=1)

        return susceptibility_err, energies_err, magnetisations_err, cvs_err

def calculate_observables(magnetisation, energy, N, beta):
    """
    Calculates the value of the susceptibility.
    """
    avg_magnetisation = np.abs(np.mean((magnetisation))) # average of the total magnetisation
    magnetisation_var = np.var(magnetisation) # variance (for susceptibility calculation)

    avg_energy = np.mean(energy)
    energy_var = np.var(energy)

    susceptibility = (beta / N) * magnetisation_var
    cv_per_spin = (beta**2 / N) * energy_var

    return avg_magnetisation, susceptibility, avg_energy, cv_per_spin

if __name__ == "__main__":

    # initialise parser object
    parser = argparse.ArgumentParser(description='Checkpoint 1')
    
    # command line arguments
    parser.add_argument('--lenx', type=int, default=50, 
                       help='Lattice width (default: 50)')
    parser.add_argument('--leny', type=int, default=50, 
                       help='Lattice height (default: 50)')
    parser.add_argument('--temp', type=float, required=True,
                       help='Temperature kT')
    parser.add_argument('--dynamics', type=str, choices=['glauber', 'kawasaki'],
                       required=True, help='Dynamics to simulate (Kawaski/Glauber)')
    parser.add_argument('--sweeps', type=int, default=10000,
                       help='Number of sweeps (default: 10000)')
    parser.add_argument('--measure', action='store_true',
                        help='Measure physical quantities of interest')
    parser.add_argument('--animate', action='store_true',
                       help='Show live animation')
    parser.add_argument('--update-interval', type=int, default=10,
                       help='Plot update interval (default: 10)')
    parser.add_argument('--sweep-direction', type=str, choices=['up', 'down'],
                        default='up', help='Decide in which direction the temperature is swept')
    parser.add_argument('--save-data', action='store_true',
                       help='Save plots and datafile')
    parser.add_argument('--lattice-config', type=str, choices=['random', 'all-up'],
                        default='all-up', help='Lattice structure to initialise')
    
    args = parser.parse_args()

    # initialise temperature sweep range based on initial beta input
    T0 = args.temp
    T_min = 1.0
    T_max = 3.0
    step = 0.1

    if args.sweep_direction == 'up': # sweep temperature depending on user request
        temperatures = np.arange(T0, T_max + step, step) # sweep upwards
    else:
        temperatures = np.arange(T0, T_min - step, -step) # sweep downwards

    # initialise lists of quantities of interest
    magnetisations = []
    magnetisations_err = []
    susceptibilities = []
    susceptibilities_err = []
    energies = []
    energies_err = []
    heat_capacities = []
    heat_capacities_err = []

    # measure intervals
    measure_interval = 10 # from lecture 2

    beta0 = 1.0 / temperatures[0]
    grid = Grid(args.lenx, args.leny, beta0, lattice_config=args.lattice_config)    

    if args.dynamics == 'glauber':
        algorithm = GlauberAlgorithm(grid)
    else:
        algorithm = KawasakiAlgorithm(grid)

    # live imshow animation
    if args.animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = None, None

    # now run simulation
    t1 = time.time() / 60 # time in minutes

    # initialise bootstrap object for error analysis outside loop
    bootstrap = BootstrapErrorAnalysis(k=1000, seed=2317434)

    N = args.lenx * args.leny # lattice size

    for idx, T in enumerate(temperatures):  # idx is only for the first random step
        grid.update_beta(1.0 / T)
        print(f"\nT = {T:.2f}, beta = {grid.beta:.4f}")
        tstart = time.time() / 60

        # special case of random config: 5000 equilibration steps at onset
        if args.lattice_config == 'random':
            if idx == 0:  # initial temperature
                eq_sweeps = args.sweeps // 2
                print(f"  (Using {eq_sweeps} equilibration sweeps for first temperature)")
            else:  # normal for ensuing sweeps
                eq_sweeps = args.sweeps // 100
        else: # not for case of all-up
            eq_sweeps = args.sweeps // 100

        # run algorithm and compute observables + errors
        mags, Es = algorithm.run(
            n_sweeps=args.sweeps,
            animate=args.animate,
            ax=ax,
            measure=args.measure,
            eq_sweeps=eq_sweeps,
            measure_interval=measure_interval
            )

        avg_M, chi, avg_E, cv = calculate_observables(mags, Es, N, grid.beta)
        magnetisations.append(avg_M)
        susceptibilities.append(chi)
        energies.append(avg_E)
        heat_capacities.append(cv)

        M_err, chi_err, E_err, cv_err = bootstrap.calculate_errors(mags, Es, N, grid.beta)
        magnetisations_err.append(M_err)
        susceptibilities_err.append(chi_err)
        energies_err.append(E_err)
        heat_capacities_err.append(cv_err)

        tfin = time.time() / 60
        loop_time = tfin - tstart
        print(f"⟨M⟩ = {avg_M:.5f}, χ = {chi:.5f}, ⟨E⟩ = {avg_E:.5f}, Cv = {cv:.5f}; took {loop_time:.3f} minutes.")

    t2 = time.time() / 60
    print(f"\nFinished. Elapsed time: {t2 - t1:.2f} minutes")

    plt.ioff() # turn off animation plot so the graphs can show

    # magnetisation plot
    plt.figure(figsize=(10,6))
    plt.errorbar(temperatures, magnetisations, yerr=magnetisations_err, marker='o', color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Average Total Magnetisation ⟨M⟩")
    plt.title(f"{args.dynamics.capitalize()} Dynamics Magnetisation")
    plt.grid()
    if args.save_data:
        mag_plot = f"{args.dynamics}_graph_magnetisation_N{args.lenx}x{args.leny}.png" # save figure for marker
        plt.savefig(mag_plot, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # susceptibility plot
    plt.figure(figsize=(10,6))
    plt.errorbar(temperatures, susceptibilities, yerr=susceptibilities_err, marker='o', color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility χ")
    plt.title(f"{args.dynamics.capitalize()} Dynamics susceptibility")
    plt.grid()
    if args.save_data:
        chi_plot = f"{args.dynamics}_graph_susceptibility_N{args.lenx}x{args.leny}.png"
        plt.savefig(chi_plot, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # energy plot
    plt.figure(figsize=(10,6))
    plt.errorbar(temperatures, energies, yerr=energies_err, marker='o', color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Average Total Energy ⟨E⟩")
    plt.title(f"{args.dynamics.capitalize()} Dynamics Energy")
    plt.grid()
    if args.save_data:
        energy_plot = f"{args.dynamics}_graph_energy_N{args.lenx}x{args.leny}.png"
        plt.savefig(energy_plot, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # heat capacities plot
    plt.figure(figsize=(10,6))
    plt.errorbar(temperatures, heat_capacities, yerr=heat_capacities_err, marker='o', color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Heat Capacity Per Spin")
    plt.title(f"{args.dynamics.capitalize()} Dynamics Heat Capacity per Spin")
    plt.grid()
    if args.save_data:
        cv_plot = f"{args.dynamics}_graph_cv_N{args.lenx}x{args.leny}.png"
        plt.savefig(cv_plot, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    if args.save_data: # save results to a dataframe if requested
        results_dataframe = pd.DataFrame({
        'Temperature': temperatures,
        'Total Magnetisation': magnetisations,
        'Total Magnetisation Error': magnetisations_err,
        'Susceptibility': susceptibilities,
        'Susceptibility Error': susceptibilities_err,
        'Total Energy': energies,
        'Total Energy Error': energies_err,
        'Heat Capacity per Spin': heat_capacities,
        'Heat Capacity per Spin Error': heat_capacities_err
        })

        # write the dataframe to a csv for the markers
        datafile = f"{args.dynamics}_results_N{args.lenx}x{args.leny}.csv"
        results_dataframe.to_csv(datafile, index=False, float_format='%.6f')

        print(f"\nResults saved to {datafile}")
