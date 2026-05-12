"""

2025 exam on modelling the growth of a crystal via PDEs.

"""

# defaults
import numpy as np
import pandas as pd

# utils
from time import perf_counter
import argparse # for command line functionality

# overdrive
from numba import njit, prange
from joblib import Parallel, delayed

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots

plt.style.use('science') # more scientific style for matplotlib
plt.rcParams['text.usetex'] = False # this avoids an annoying latex installation

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

@njit(parallel=True)
def compute_grad_2D(f: np.ndarray, N: int, dx: float) -> tuple[np.ndarray, ...]:
    """
    Computes the 2D grad of physical parameter f on a grid.
    """
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)

    for i in prange(N):
        for j in range(N):

            grad_x[i,j] = (f[i, (j+1)%N] - f[i, (j-1)%N]) / (2*dx)       
            grad_y[i,j] = (f[(i+1)%N, j] - f[(i-1)%N, j]) / (2*dx)

    return grad_x, grad_y

@njit
def compute_mu(phi: np.ndarray, N: int, dx: float, a: float, 
               q0: float) -> np.ndarray:
    """
    Computes the chemical potential as described in the exam.
    """
    lap_phi = compute_laplacian_2D(f=phi, N=N, dx=dx)

    return -a*phi + phi**3 + phi*(q0**4) + 2*lap_phi*(q0**2) + compute_laplacian_2D(f=lap_phi, N=N, dx=dx)

@njit
def phi_step(phi: np.ndarray, N: int, dx: float, dt: float,
             a: float, q0: float, M: float) -> np.ndarray:
    """
    Advances one step of the algorithm (forward Euler).
    """
    mu = compute_mu(phi=phi, N=N, dx=dx, a=a, q0=q0)

    return phi + dt*(M*compute_laplacian_2D(f=mu, N=N, dx=dx))

@njit
def velocity_field(phi: np.ndarray, N: int, v0: float) -> np.ndarray:
    """
    Computes the velocity field over the lattice (with PBCs)s.
    """
    vx = np.zeros_like(phi)

    for i in range(N):
        vx[i,:] = -v0*np.sin((2*np.pi*i)/N)
        
    return vx

@njit
def modified_phi_step(phi: np.ndarray, N: int, dx: float, dt: float,
             a: float, q0: float, M: float, vx: np.ndarray) -> np.ndarray:
    """
    Advances one step of the new algorithm (forward Euler).
    """
    mu = compute_mu(phi=phi, N=N, dx=dx, a=a, q0=q0)
    rhs = M*compute_laplacian_2D(f=mu, N=N, dx=dx)

    grad_x, _ = compute_grad_2D(f=phi, N=N, dx=dx)

    return phi + dt*(rhs - vx*grad_x)

class CrystalSolver:
    """
    Class to model the evolution of the crystal lattice.
    """
    def __init__(self, N: int, dx: float, dt: float, phi0: float,
                 a: float, q0: float, M: float):
        
        self.N = N
        self.dx = dx
        self.dt = dt

        self.initialise_grid(phi0)

        self.a = a
        self.q0 = q0
        self.M = M

        self.time = 0.0

    def initialise_grid(self, phi0: float) -> np.ndarray:
        """
        Initialises the lattice according to random configuration.
        """
        self.phi = np.random.uniform(phi0 - 0.1, phi0 + 0.1, (self.N, self.N))

    def step(self) -> None:
        """
        Advance one step on the lattice.
        """
        self.phi = phi_step(phi=self.phi, N=self.N, dt=self.dt, dx=self.dx,
                            a=self.a, q0=self.q0, M=self.M)
        self.time += self.dt

    def velocity_dependent_field(self, v0: float) -> None:
        """
        Generates velocity-dependent field.
        """
        self.v0 = v0
        self.vx = velocity_field(phi=self.phi, N=self.N, v0=self.v0)

    def modified_step(self) -> None:
        """
        Uses modified step from later part of exam.
        """
        self.phi = modified_phi_step(phi=self.phi, N=self.N, dt=self.dt, dx=self.dx,
                            a=self.a, q0=self.q0, M=self.M, vx=self.vx)
        self.time += self.dt

    def _animate_sweep(self, frames: int) -> list:
        """
        Animates a sweep (defined as the number of steps in range()).
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(100):
            self.step()
        self.im.set_data(self.phi)
        self.ax.set_title(f't = {self.time:.2f}')
        return [self.im]
    
    def run(self, animate: bool, max_steps: int, save_animation: bool = False, task: str = 'unspecified'):
        """
        Iterate the crystal lattice forwards in time.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', interpolation='nearest')
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
        elif save_animation:
            for _ in range(50000):
                self.step()
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', interpolation='nearest')
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            self.anim.save(f"task_{task}_animation.gif", writer='pillow', fps=30)

        else:
            for _ in range(max_steps):
                self.step()

    def _animate_modified_sweep(self, frames: int) -> list:
        """
        Animates a sweep (defined as the number of steps in range()).
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(100):
            self.modified_step()
        self.im.set_data(self.phi)
        self.ax.set_title(f't = {self.time:.2f}')
        return [self.im]
    
    def run_modified(self, animate: bool, max_steps: int, save_animation: bool = False, task: str = 'd'):
        """
        Iterate the crystal lattice forwards in time.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', interpolation='nearest')
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_modified_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
        elif save_animation:
            for _ in range(50000):
                self.modified_step()
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', interpolation='nearest')
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_modified_sweep,
                                    frames=max_steps, interval=50,
                                    repeat=False)
            self.anim.save(f"task_{task}_animation.gif", writer='pillow', fps=30)

        else:
            for _ in range(max_steps):
                self.modified_step()

    def compute_spatial_variance(self):
        """
        Compute the spatial variance across the lattice.
        """
        return np.var(self.phi)
    
    @staticmethod
    def find_modified_solution(v0: float, phi0: float = -0.1, N: int = 50, dx: float = 1.0, 
                               dt: float = 0.01, a: float = 0.1, q0: float = 0.5, M: float = 0.1, 
                               max_steps: int = 10000, eq_steps: int = 100000, measure_interval: int = 100):
        """
        Finds solution to advection-modified crystal evolution.
        """
        tcs = CrystalSolver(N=N, dx=dx, dt=dt, phi0=phi0, a=a, q0=q0, M=M)
        tcs.velocity_dependent_field(v0=v0)
        for _ in range(eq_steps):
            tcs.modified_step()
        
        return tcs.phi
    
    @staticmethod
    def steady_state_variance(phi0: float, N: int = 50, dx: float = 1.0, dt: float = 0.01, 
                            a: float = 0.1, q0: float = 0.5, M: float = 0.1, max_steps: int = 100000,
                            eq_steps: int = 50000, measure_interval: int = 100):
        """
        Computes steady state variance as a function of time.
        """
        tcs = CrystalSolver(N=N, dx=dx, dt=dt, phi0=phi0, a=a, q0=q0, M=M)
        vars = []

        for i in range(max_steps):
            tcs.step()
            if (i > eq_steps) and (i % measure_interval == 0):
                vars.append(tcs.compute_spatial_variance())

        steady_state_variance = np.mean(np.array(vars))

        return phi0, steady_state_variance

    @staticmethod
    def task_c():
        """
        Parallel data collection for task c.
        """
        phi_vals = np.arange(0.0, 0.25, 0.0125) # half resolution to start with 
        results = np.array(Parallel(n_jobs=-1, return_as='list')(
            delayed(CrystalSolver.steady_state_variance)(phi0) for phi0 in phi_vals))
        
        phi0s, variances = zip(*results)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(phi0s, variances, color='r')
        ax.grid()
        ax.set_title(r"Variance against $\phi_0$")
        ax.set_xlabel(r"$\phi_0$")
        ax.set_ylabel(f"Variance")
        fig.savefig(f"task_c_graph.png", dpi=300)

        pd.DataFrame({"phi0s": phi0s, "vars": variances}).to_csv(f"task_c_data.csv", index=False)

    @staticmethod
    def task_d():
        """
        Parallel data collection for task d.
        """
        v0_vals = [0.001, 0.01, 0.1]
        results = Parallel(n_jobs=-1, return_as='list')(
            delayed(CrystalSolver.find_modified_solution)(v0) for v0 in v0_vals)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, v0, phi in zip(axes, v0_vals, results):
            im = ax.imshow(phi, cmap='coolwarm', interpolation='nearest')
            ax.set_title(f'$v_0$ = {v0}')
            fig.colorbar(im, ax=ax)
        fig.savefig("task_d_snapshots.png", dpi=300)

parser = argparse.ArgumentParser(description='2025 Exam')

parser.add_argument('--N', type=int, default=50)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--save_animation', action='store_true')
parser.add_argument('--dx', type=float, default=1.0)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--task_b', action='store_true')
parser.add_argument('--task_c', action='store_true')
parser.add_argument('--task_d', action='store_true')
parser.add_argument('--a', type=float, default=0.1)
parser.add_argument('--q0', type=float, default=0.5)
parser.add_argument('--M', type=float, default=0.1)
parser.add_argument('--phi0', type=float, default=0.5)

def main():

    args = parser.parse_args()

    if args.task_b:
        phi0_range = [-0.1, 0, 0.1]
        cs = CrystalSolver(N=50, dx=1.0, dt=0.01, phi0=0.0,
                            a=0.1, q0=0.5, M=0.1)
        for phi0 in phi0_range:
            cs.time = 0.0
            cs.initialise_grid(phi0=phi0)
            cs.run(animate=False, max_steps=1000, save_animation=True, task=f"b_phi0_{phi0}")

    elif args.task_c:
        CrystalSolver.task_c()

    elif args.task_d:
        CrystalSolver.task_d()
        v0_range = [0.001, 0.01, 0.1]
        cs = CrystalSolver(N=50, dx=1.0, dt=0.01, phi0=-0.1, a=0.1, q0=0.5, M=0.1)
        for v0 in v0_range:
            cs.time = 0.0
            cs.initialise_grid(phi0=-0.1)
            cs.velocity_dependent_field(v0=v0)
            cs.run_modified(animate=False, max_steps=1000, save_animation=True, task=f"d_v0_{v0}")
    
    else:
        cs = CrystalSolver(N=args.N, dx=args.dx, dt=args.dt, phi0=args.phi0,
                           a=args.a, q0=args.q0, M=args.M)
        cs.run(animate=args.animate, max_steps=args.max_steps, save_animation=args.save_animation)

if __name__ == "__main__":
    main()