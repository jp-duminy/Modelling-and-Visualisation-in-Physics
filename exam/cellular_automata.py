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
        self.four_shifts = np.array([(-1, 0), # top
                            (0, -1), # left
                            (0, 1), # right
                            (1, 0) # bottom
                            ])

        self.lattice = None

    def place_glider(self, loc):
        """
        Place a glider (translating state) on the input loc.
        """
        i, j = loc

        # glider shape
        glider = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 1, 1]])

        # slice the lattice to place the glider
        self.lattice[i-1:i+2, j-1:j+2] = glider     

    def place_blinker(self, loc):
        """
        Place a blinker (oscillating state) on the input loc.
        """  
        i, j = loc

        self.lattice[i-1:i+2, j] = 1

    def initialise_lattice(self):
        """
        Agnostic lattice initialisation for either game. 
        """
        if self.game == 'GoL':
            if self.config == 'random':
                self.lattice = (np.random.random((self.N, self.N)) < self.density).astype(int) # density of live cells
            if self.config == 'glider':
                self.lattice = (np.zeros((self.N, self.N)))
                centre = [self.N // 2, self.N // 2]
                self.place_glider(centre)
            if self.config == 'oscillating':
                self.lattice = (np.zeros((self.N, self.N)))
                centre = [self.N // 2, self.N // 2]
                self.place_blinker(centre)
        elif self.game == 'SIRS':
            self.lattice = np.random.choice([0, 1, 2], size=(self.N, self.N)).astype(int)

    def apply_immunity(self, frac_immune):
        """
        Apply immune cells (value 3) to the lattice (for SIRS)
        """
        n_immune = int(self.N**2 * frac_immune) # to avoid random choice crashing
        immune_sites = np.random.choice(self.N**2, size=n_immune, replace=False)
        self.lattice.flat[immune_sites] = 3

    def count_neighbours(self):
        """
        Counts the nearest-neighbour sum, pertinent to the GoL.
        """
        # make an empty lattice that we can store the total for
        total = np.zeros_like(self.lattice)

        for dx, dy in self.eight_shifts:
            total += np.roll(np.roll(self.lattice, dx, axis=0), dy, axis=1) # np.roll implementation (lecture 1)

        return total
    
def compute_gol_eq_time(N, density, max_steps, eq_steps):
    """
    Joblib-friendly data collection for the Game of Life (task 2)
    """
    grid = Grid(N=N, config='random', density=density, game='GoL')
    updater = GameOfLife(grid, max_steps=max_steps, eq_steps=eq_steps)
    updater.run(animate=False)
    return updater.eq_time
    
class GameOfLife:
    def __init__(self, grid, max_steps, eq_steps):
        self.grid = grid # pass grid object (same logic as CP1)
        self.grid.initialise_lattice()

        self.history = deque(maxlen=eq_steps)

        self.eq_time = None
        self.equilibrated = False
        self.max_steps = max_steps

        self.fig = None
        self.ax = None

        # glider tracking
        self.com_history = []
        self.offset = np.array([0.0, 0.0])
        self.previous_com = None
        
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
    
    def compute_glider_com(self):
        """
        Computes the centre-of-mass of the glider, appropriately accounting for PBCs.
        """
        live = np.argwhere(self.grid.lattice == 1)
        com = np.zeros(2)
        for dim in range(2):
            coords = live[:, dim].astype(float)
            if coords.max() - coords.min() > self.grid.N / 2:
                # cluster straddles boundary — shift low values up
                coords[coords < self.grid.N / 2] += self.grid.N
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
            self.offset[delta > self.grid.N/5] -= self.grid.N
            self.offset[delta < -self.grid.N/5] += self.grid.N
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

    def _animate_step(self, frame):
        """
        Animates a step of the lattice.
        """
        current_state = self.grid.lattice.copy()
        self.history.append(current_state)
        self.step()
        if self.grid.config == 'glider':
            com = self.compute_glider_com()
            offset_com = self.unwrap_centre_of_mass(com)
            self.com_history.append(offset_com)

        self.im.set_data(self.grid.lattice)
        self.ax.set_title(f'Step: {frame}')

        if not self.equilibrated and self.equilibrium_state():
            self.eq_time = frame
            self.equilibrated = True

        return [self.im]

    def run(self, animate=False):
        """
        Runs the game of life until the user-defined maximum number of steps is reached.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid.lattice, cmap='binary', interpolation='nearest')
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=self.max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            step_number = 0
            while step_number < self.max_steps:
                current_state = self.grid.lattice.copy()
                self.history.append(current_state)
                self.step()
                if self.grid.config == 'glider':
                    com = self.compute_glider_com()
                    offset_com = self.unwrap_centre_of_mass(com)
                    self.com_history.append(offset_com)

                if not self.equilibrated and self.equilibrium_state():
                    self.eq_time = step_number
                    print(f"Reached equilibrium after {step_number} steps.")
                    self.equilibrated = True
                step_number += 1

        if self.eq_time is None:
            print("Did not equilibrate.")
            self.eq_time = self.max_steps

# numba functions cannot live in a class
@njit
def sirs_sweep(lattice, N, p1, p2, p3, four_shifts):
    sites_i = np.random.randint(0, N, size=N * N)
    sites_j = np.random.randint(0, N, size=N * N)
    randoms = np.random.random(N * N)

    for k in range(N * N):
        i = sites_i[k]
        j = sites_j[k]

        if lattice[i, j] == 0:
            for m in range(4):
                ni = (i + four_shifts[m, 0]) % N
                nj = (j + four_shifts[m, 1]) % N
                if lattice[ni, nj] == 1:
                    if randoms[k] < p1:
                        lattice[i, j] = 1
                    break

        elif lattice[i, j] == 1:
            if randoms[k] < p2:
                lattice[i, j] = 2

        elif lattice[i, j] == 2:
            if randoms[k] < p3:
                lattice[i, j] = 0

def compute_mean_point(p1, p3, N, eq_steps, measure_steps):
    """
    Joblib-friendly computation of points for heatmap (Task 3)
    """
    p2 = 0.5 # hard-coded
    grid = Grid(N=N, config='random', density=0.5, game='SIRS')
    sim = SIRS(grid, max_steps=0, eq_steps=eq_steps, p1=p1, p2=p2, p3=p3)
    for _ in range(eq_steps):
        sim.sweep()
    infected = []
    for _ in range(measure_steps):
        sim.sweep()
        infected.append(np.sum(sim.grid.lattice == 1) / N**2)
    return np.mean(infected)

def compute_variance_point(p1, p3, N, eq_steps, measure_steps):
    """
    Joblib-friendly computation of variances (Task 4)
    """
    grid = Grid(N=N, config='random', density=0.5, game='SIRS')
    sim = SIRS(grid, max_steps=0, eq_steps=eq_steps, p1=p1, p2=0.5, p3=p3)
    for _ in range(eq_steps):
        sim.sweep()
    infected = np.zeros(measure_steps)
    for i in range(measure_steps):
        sim.sweep()
        infected[i] = np.sum(sim.grid.lattice == 1)
    return infected

def compute_immunity_point(immunity_frac, N, eq_steps, measure_steps):
    """
    Joblib-friendly computation of immune (Task 5). Reuses code but with apply_immunity.
    """
    p1, p2, p3 = 0.5, 0.5, 0.5 # hard code
    grid = Grid(N=N, config='random', density=0.5, game='SIRS')
    sim = SIRS(grid, max_steps=0, eq_steps=eq_steps, p1=p1, p2=p2, p3=p3)
    grid.apply_immunity(immunity_frac)
    for _ in range(eq_steps):
        sim.sweep()
    infected = []
    for _ in range(measure_steps):
        sim.sweep()
        infected.append(np.sum(sim.grid.lattice == 1) / N**2)
    return np.mean(infected)

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

class SIRS:
    def __init__(self, grid, max_steps, eq_steps, p1, p2, p3):

        self.grid = grid
        self.grid.initialise_lattice()
        self.max_steps = max_steps
        self.eq_steps = eq_steps

        self.fig = None
        self.ax = None
        self.cmap = ListedColormap(['blue', 'red', 'green', 'white']) # white is for task 5 (immune)

        # infection probabilities
        self.p1 = p1 # p(S -> I)
        self.p2 = p2 # p(I -> R)
        self.p3 = p3 # p(R -> S)

    def step(self):
        """
        Picks a random site on the lattice and applies model criteria.
        """
        i = np.random.randint(0, self.grid.N)
        j = np.random.randint(0, self.grid.N)

        # susceptible
        if self.grid.lattice[i, j] == 0:  
            # like in Ising, modulo naturally handles boundary conditions (no np.roll this time)
            neighbours = [((i + di) % self.grid.N, (j + dj) % self.grid.N)
                        for di, dj in self.grid.four_shifts]
            # this only computes probability once: check whether it should scale by number of infected
            if any(self.grid.lattice[ni, nj] == 1 for ni, nj in neighbours):
                if np.random.random() < self.p1:
                    self.grid.lattice[i, j] = 1

        # infected
        elif self.grid.lattice[i, j] == 1:
            if np.random.random() < self.p2:
                self.grid.lattice[i, j] = 2

        # recovered
        elif self.grid.lattice[i, j] == 2:
            if np.random.random() < self.p3:
                self.grid.lattice[i, j] = 0
            
    def sweep(self):
        """
        Passes the whole function into numba to optimise runtime.
        """
        # this optimises everything but for numba the syntax must be different, see step() for something more readable
        sirs_sweep(self.grid.lattice, self.grid.N, self.p1, self.p2, self.p3, self.grid.four_shifts)

    def _animate_sweep(self, frame):
        """
        Animates a step of the SIRS model.
        """
        self.sweep()

        self.im.set_data(self.grid.lattice)
        self.ax.set_title(f'Step: {frame}')

        return [self.im]
    
    def measured_infected(self):
        """
        Measures the fraction of infected on the site at a given time.
        """
        return np.sum(self.grid.lattice == 1) / self.grid.N**2

    def run(self, animate=False):
        """
        Runs the SIRS model until the user-defined maximum number of steps is reached.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.grid.lattice, cmap=self.cmap, 
                          vmin=0, vmax=3, interpolation='nearest')
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=self.max_steps, interval=50,
                                    repeat=False)
            plt.show()  
        else:
            for _ in range(self.max_steps):
                self.sweep()


parser = argparse.ArgumentParser(description='Cellular Automata: GoL and SIRS')

# shared arguments
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--eq-steps', type=int, default=10)

subparsers = parser.add_subparsers(dest='game', required=True)

# game of life arguments
gol_parser = subparsers.add_parser('GoL')
gol_parser.add_argument('--config', type=str, choices=['random', 'glider', 'oscillating'])
gol_parser.add_argument('--density', type=float, default=0.5)
gol_parser.add_argument('--measure', action='store_true')

# SIRS arguments
sirs_parser = subparsers.add_parser('SIRS')
sirs_parser.add_argument('--p1', type=float, required=True, help='p(S->I)')
sirs_parser.add_argument('--p2', type=float, required=True, help='p(I->R)')
sirs_parser.add_argument('--p3', type=float, required=True, help='p(R->S)')
sirs_parser.add_argument('--measure', action='store_true')
sirs_parser.add_argument('--immunity', action='store_true')
sirs_parser.add_argument('--immunity-frac', type=float, default=0.0)

def main():
    """
    main() function; includes both GoL and SIRS.
    """
    args = parser.parse_args()

    if args.game == 'GoL':
        # data collection run
        if args.measure:
            data_collection_runs = 1000 # per CP2 guidelines
            t1 = perf_counter()
            # implemented joblib here too
            eq_times = Parallel(n_jobs=-1)(
                delayed(compute_gol_eq_time)(args.N, args.density, args.max_steps, args.eq_steps)
                for _ in tqdm(range(data_collection_runs), desc='GoL Equilibration')
            )

            t2 = perf_counter()
            results_df = pd.DataFrame({
                'Equilibration Times': eq_times
            })
            t_overall = t2 - t1
            print(f"Time taken: {t_overall:.3f} seconds.")

            datafile = f"equilibration_results_N{args.N}x{args.N}.csv"
            results_df.to_csv(datafile, index=False, float_format='%.6f')
            print(f"Saved results to {datafile}")

            plt.hist(eq_times, bins=100, density=True)
            plt.xlabel('Equilibration time (steps)')
            plt.ylabel('Probability Density')
            plt.title('Game of Life: Equilibrium Steps Histogram')
            plt.show()

        # single run
        else:
            grid = Grid(N=args.N, config=args.config, density=args.density, game='GoL')
            updater = GameOfLife(grid, max_steps=args.max_steps, eq_steps=args.eq_steps)
            updater.run(animate=args.animate)
            if args.config == 'glider':
                velocity = updater.estimate_speed()
                print(f"Velocity of glider: {velocity:.5f}")
                c_velocity = velocity / np.sqrt(2)
                print(f"In units of c: {c_velocity:.3f}c/generation")
            print(f"Equilibrium time: {updater.eq_time}")

    elif args.game == 'SIRS':
        if args.measure:
            # hard-code these to prevent human error
            p_values = np.arange(0, 1.05, 0.025)
            eq_steps = 100
            measure_steps = 1000

            # numba needs a burn-in to compile code to machine code, so do a a warmup
            dummy_grid = Grid(N=args.N, config='random', density=0.5, game='SIRS')
            dummy_sim = SIRS(dummy_grid, max_steps=0, eq_steps=0, p1=0.5, p2=0.5, p3=0.5) # ignore values, just a burn-in
            dummy_sim.sweep()

            # task 3: heatmap
            t1 = perf_counter()
            results = Parallel(n_jobs=-1)(
                delayed(compute_mean_point)(p1, p3, args.N, eq_steps, measure_steps)
                for p1, p3 in tqdm([(p1, p3) for p1 in p_values for p3 in p_values], desc='Phase diagram')
            )
            t2 = perf_counter()
            time_taken = t2 - t1
            phase_diagram = np.array(results).reshape(len(p_values), len(p_values))

            plt.figure()
            plt.imshow(phase_diagram, origin='lower', extent=[0, 1, 0, 1],
                    aspect='equal', cmap='inferno')
            plt.colorbar(label=r'$\langle I \rangle / N^2$')
            plt.xlabel(r'$p_3$ (R $\to$ S)')
            plt.ylabel(r'$p_1$ (S $\to$ I)')
            plt.title(f"SIRS Phase Diagram (p(I -> R) = 0.5)")
            plt.show()
            pd.DataFrame(phase_diagram, index=p_values, columns=p_values).to_csv('task3_phase_diagram.csv')
            print(f"Task 3 data saved.")
            print(f"Time taken for Task 3: {time_taken:.3f}")

            # task 4: waves in heatmap
            t1 = perf_counter()
            p1_cut_values = np.arange(0.2, 0.55, 0.01)
            # initialise bootstrap for this one, block bootstrap this time
            bootstrap = BootstrapErrorAnalysis()
            cut_results = Parallel(n_jobs=-1)(
                delayed(compute_variance_point)(p1, 0.5, args.N, 100, 10000)
                for p1 in tqdm(p1_cut_values, desc='Variance cut')
            )

            var_cut = [np.var(ts) / args.N**2 for ts in cut_results]
            var_errors = [bootstrap.calculate_errors(ts) / args.N**2 for ts in cut_results]
            t2 = perf_counter()
            time_taken = t2 - t1

            plt.errorbar(p1_cut_values, var_cut, yerr=var_errors, capsize=3, marker='o')
            plt.xlabel(r'$p_1$ (S $\to$ I)')
            plt.ylabel(r'Var($I / N$)')
            plt.title(f"Variance of the infected fraction at fixed p(I -> R) = 0.5")
            plt.show()
            pd.DataFrame({'p1': p1_cut_values, 'variance': var_cut, 'variance_err': var_errors}).to_csv('task4_waves.csv')
            print(f"Task 4 data saved.")
            print(f"Time taken for Task 4: {time_taken:.3f}")

        elif args.immunity:
            frac_values = np.arange(0, 1.01, 0.02) # granular & comprehensive enough, observe herd immunity at ~0.3
            t1 = perf_counter()
            vaccine_results = Parallel(n_jobs=-1)(
            delayed(compute_immunity_point)(f, args.N, 100, 1000)
            for f in tqdm(frac_values, desc='Vaccination Run')
            )
            t2 = perf_counter()
            time_taken = t2 -t1
            plt.plot(frac_values, vaccine_results, marker='o', markersize=3)
            plt.xlabel(f"Fraction of immune cells")
            plt.ylabel(r'$\langle I \rangle / N^2$')
            plt.title(f"Average infected fraction vs immunity")
            plt.show()
            pd.DataFrame({'immunity_frac': frac_values, 'mean_infected': vaccine_results}).to_csv('task5_immunity.csv')
            print(f"Task 5 data saved.")
            print(f"Time taken for Task 5: {time_taken:.3f}")

        else:
            grid = Grid(N=args.N, config='random', density=0.5, game='SIRS')
            updater = SIRS(grid, max_steps=args.max_steps, eq_steps=args.eq_steps,
                        p1=args.p1, p2=args.p2, p3=args.p3)
            if args.immunity_frac > 0:
                grid.apply_immunity(args.immunity_frac)
            updater.run(animate=args.animate)

if __name__ == '__main__':
    main()

# parameter sets for SIRS model
# blue: susceptible, p1 = p(S -> I)
# red: infected, p2 = p(I -> R)
# green: recovered, p3 = p(R -> S)
# white: immune, defined by immune_frac (task 5)

# absorbing state: p1 = 0.1, p2 = 0.5, p3 = 0.4
# dynamic equilibrium: p1 = 0.3, p2 = 0.3, p3 = 0.5
# cyclic infected waves: p1 = 0.8, p2 = 0.1, p3 = 0.015