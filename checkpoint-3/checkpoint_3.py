# defaults
import numpy as np
import pandas as pd
import scipy

# utils
from time import perf_counter
import argparse # for command line functionality

# overdrive
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots
plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

class Grid:
    """
    Agnostic grid operations.
    """
    def __init__(self, N, dx):

        self.N = N
        self.dx = dx
        self.lattice = None

    def initialise_lattice(self):
        """
        Create a lattice.
        """
        self.lattice = np.zeros((self.N, self.N))

    def compute_laplacian(self, f):
        """
        Computes the discretised laplacian of quantity f of interest (2D).
        """
        return compute_laplacian(f, self.N, self.dx)
    
@njit
def compute_laplacian(f, N, dx):
    laplacian = np.empty_like(f)
    for i in range(N):
        for j in range(N):

            laplacian[i, j] = (f[(i+1)%N,j] + f[(i-1)%N,j]
                     + f[i,(j+1)%N] + f[i,(j-1)%N]
                     - 4*f[i,j]) / dx**2

    return laplacian

@njit
def compute_mu(phi, N, dx):
    return -phi + phi**3 - compute_laplacian(phi, N, dx)

@njit
def cahn_step(phi, N, dt, dx):
    mu = compute_mu(phi, N, dx)
    return phi + dt * compute_laplacian(mu, N, dx)
    
class CahnHilliard:
    """
    Dimensionless, discretised numerical Cahn-Hilliard eqn solver.
    """
    def __init__(self, grid, phi0, dt=0.01):

        self.grid = grid
        self.phi = np.random.uniform(phi0 - 0.1, phi0 + 0.1, (grid.N, grid.N))
        self.dt = dt
        self.time = 0.0

        self.im = None
        self.phi_vals = []
    
    def step(self):
        """
        Advance one step on the lattice.
        """
        self.phi = cahn_step(self.phi, self.grid.N, self.dt, self.grid.dx)
        self.time += self.dt

    def _animate_step(self, frame):
        """
        step() wrapper for animation.
        """
        self.step()
        self.im.set_data(self.phi)
        self.ax.set_title(f'Step: {frame}, t = {self.time:.2f}')
        return [self.im]

    def run(self, nsteps, measure_interval=10, animate=False):
        """
        Run the Cahn-Hilliard equation.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(self.im)
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=nsteps, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for i in range(nsteps):
                self.step()
                if i % measure_interval == 0:
                    self.phi_vals.append(self.phi.copy())


parser = argparse.ArgumentParser(description='Partial Differential Equations')

# shared arguments
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--measure_interval', type=int, default=10)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--dx', type=float, required=True)

subparsers = parser.add_subparsers(dest='equation', required=True)

# game of life arguments
cahn_parser = subparsers.add_parser('C-H')
cahn_parser.add_argument('--phi0', type=float, required=True)
cahn_parser.add_argument('--timestep', type=float, required=True)
cahn_parser.add_argument('--n_steps', type=int, default=10000, required=True)

def main():

    args = parser.parse_args()

    if args.equation == 'C-H':
        grid = Grid(N=args.N, dx=args.dx)
        cahn_hilliard = CahnHilliard(grid=grid, phi0=args.phi0, dt=args.timestep)
        cahn_hilliard.run(nsteps=10000, measure_interval=args.measure_interval, animate=args.animate)

if __name__ == '__main__':
    main()
