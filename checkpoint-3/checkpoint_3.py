# defaults
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

#
# agnostic kernels
#
    
@njit(parallel=True)
def compute_laplacian_2D(f: np.ndarray, N: int, dx: float) -> np.ndarray:
    laplacian = np.zeros_like(f)

    for i in prange(N):
        for j in range(N):

            laplacian[i, j] = (f[(i+1)%N,j] + f[(i-1)%N,j]
                     + f[i,(j+1)%N] + f[i,(j-1)%N]
                     - 4*f[i,j]) / dx**2

    return laplacian

@njit(parallel=True)
def compute_grad_2D(f: np.ndarray, N: int, dx: float) -> np.ndarray:
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)

    for i in prange(N):
        for j in range(N):
            
            grad_x[i,j] = (f[(i+1)%N, j] - f[(i-1)%N, j]) / (2*dx)
            grad_y[i,j] = (f[i, (j+1)%N] - f[i, (j-1)%N]) / (2*dx)

    return grad_x, grad_y

@njit
def compute_mu(phi: np.ndarray, N: int, dx: float) -> np.ndarray:
    return -phi + phi**3 - compute_laplacian_2D(phi, N, dx)

@njit
def cahn_step(phi: np.ndarray, N: int, dt: float, dx: float) -> np.ndarray:
    mu = compute_mu(phi, N, dx)
    return phi + dt * compute_laplacian_2D(mu, N, dx)
    
@njit
def total_free_energy(phi: np.ndarray, N: int, dx: float) -> float:
    grad_x, grad_y = compute_grad_2D(phi, N, dx)
    f = (-1/2 * phi**2) + (1/4 * phi**4) + (1/2 * (grad_x**2 + grad_y**2))
    return np.sum(f) * dx**2

class CahnHilliard:
    """
    Dimensionless, discretised numerical Cahn-Hilliard eqn solver.
    """
    def __init__(self, phi0, N=100, dx=1.0, dt=0.01):

        self.N = N
        self.dx = dx
        self.dt = dt
        self.phi = np.random.uniform(phi0 - 0.1, phi0 + 0.1, (self.N, self.N))
        self.time = 0.0

        self.im = None
        self.phi_vals = []
        self.free_energy_vals = []
        self.time_vals = []
    
    def step(self) -> None:
        """
        Advance one step on the lattice.
        """
        self.phi = cahn_step(self.phi, self.N, self.dt, self.dx)
        self.time += self.dt

    def _animate_sweep(self, frames: int) -> list:
        """
        Animates a sweep (defined as the number of steps in range()).
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(50):
            self.step()
        self.im.set_data(self.phi)
        self.ax.set_title(f't = {self.time:.2f}')
        return [self.im]

    def run(self, n_steps: int, measure_interval: int = 10, animate: bool = False):
        """
        Run the Cahn-Hilliard equation.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(self.im)
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=n_steps // 50, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for i in range(n_steps):
                self.step()
                if i % measure_interval == 0:
                    self.phi_vals.append(self.phi.copy())
                    free_energy = total_free_energy(self.phi, self.N, self.dx)
                    self.free_energy_vals.append(free_energy)
                    self.time_vals.append(self.time)

    @staticmethod
    def data_collection(phi0: float, N: int, dx: float, dt: float, n_steps: int, measure_interval: int) -> tuple[list, list]:
        """
        Wrapper of run function (for parallelised data collection).
        """
        ch = CahnHilliard(phi0=phi0, N=N, dx=dx, dt=dt)
        ch.run(n_steps=n_steps, measure_interval=measure_interval)

        return ch.time_vals, ch.free_energy_vals

#
# the poisson numba engine room
#

@njit(parallel=True)
def _electric_potential_jacobi(phi0: np.ndarray, rho: np.ndarray, dx: float, L: int) -> np.ndarray:
    """
    Jacobi algorithm for electric field in 3D.
    Code comments are applicable to other numba engine room functions.
    """
    phi_new = np.zeros_like(phi0) # zeros_like instead of empty_like applies dirichlet intrinsically
    # range accomodates dirichlet BCs: edges = 0 always
    for i in prange(1, L-1): # only do prange for outer loop (otherwise strange things occur in nested parallel loops)
        for j in range(1, L-1):
            for k in range(1, L-1):
                phi_new[i,j,k] = ((phi0[i+1,j,k] + phi0[i-1,j,k]
                              + phi0[i,j+1,k] + phi0[i,j-1,k]
                              + phi0[i,j,k+1] + phi0[i,j,k-1])
                              + dx**2 * rho[i,j,k]) / 6 # 6 nearest neighbours
    return phi_new

@njit(parallel=True)
def _magnetic_potential_jacobi(phi0: np.ndarray, rho: np.ndarray, dx: float, L: int) -> np.ndarray:
    """
    Jacobi algorithm for magnetic field in 2D.
    """
    phi_new = np.zeros_like(phi0) 
    for i in prange(1, L-1):
        for j in range(1, L-1):
            phi_new[i,j] = ((phi0[i+1,j] + phi0[i-1,j]
                            + phi0[i,j+1] + phi0[i,j-1])
                            + dx**2 * rho[i,j]) / 4 # 4 nearest neighbours
    return phi_new

# gauss seidel is sequential so do not parallelise
@njit
def _electric_potential_gauss_seidel(phi: np.ndarray, rho: np.ndarray, dx: float, L: int, 
                                     omega: float = 1):
    """
    Gauss-Seidel algorithm for electric field.
    Omega controls successive over-relaxation (default 1, reduces to Gauss-Seidel).
    """
    for i in range(1, L-1):
        for j in range(1, L-1):
            for k in range(1, L-1):
                phi_old = phi[i,j,k]
                phi_gs = ((phi[i+1,j,k] + phi[i-1,j,k]
                              + phi[i,j+1,k] + phi[i,j-1,k]
                              + phi[i,j,k+1] + phi[i,j,k-1])
                              + dx**2 * rho[i,j,k]) / 6
                phi[i,j,k] = omega * phi_gs + (1 - omega)*phi_old

@njit
def _magnetic_potential_gauss_seidel(phi: np.ndarray, rho: np.ndarray, dx: float, L: int, 
                                     omega: float = 1):
    """
    Gauss-Seidel algorithm for magnetic field.
    Omega controls successive over-relaxation (default 1, reduces to Gauss-Seidel).
    """
    for i in range(1, L-1):
        for j in range(1, L-1):
            phi_old = phi[i,j]
            phi_gs = ((phi[i+1,j] + phi[i-1,j]
                        + phi[i,j+1] + phi[i,j-1])
                        + dx**2 * rho[i,j]) / 4
            phi[i,j] = omega * phi_gs + (1 - omega)*phi_old

@njit(parallel=True)
def compute_electric_field(f: np.ndarray, L: int, dx: float) -> tuple[np.ndarray, ...]:
    """
    Computes electric field, where E = -grad(phi)
    """
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)
    grad_z = np.zeros_like(f)

    for i in prange(1, L-1):
        for j in range(1, L-1):
            for k in range(1, L-1):

                grad_x[i,j,k] = (f[(i+1),j,k] - f[(i-1),j,k]) / (2*dx)
                grad_y[i,j,k] = (f[i,(j+1),k] - f[i,(j-1),k]) / (2*dx)
                grad_z[i,j,k] = (f[i,j,(k+1)] - f[i,j,(k-1)]) / (2*dx)

    return -grad_x, -grad_y, -grad_z

@njit(parallel=True)
def compute_magnetic_field(f: np.ndarray, L: int, dx: float) -> tuple[np.ndarray, ...]:
    """
    Computes magnetic field, where B = curl(A)
    """
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)

    for i in prange(1, L-1):
        for j in range(1, L-1):

            grad_x[i,j] = (f[(i+1),j] - f[(i-1),j]) / (2*dx)
            grad_y[i,j] = (f[i,(j+1)] - f[i,(j-1)]) / (2*dx)

    return grad_y, -grad_x # negative in cross product (levi-civita)

class Poisson:

    def __init__(self, problem: str, method: str, tolerance: float,
                omega: float = 1.0, N: int = 100, dx: float = 1.0):

        self.N = N 
        self.problem = problem

        if self.problem == 'electric':
            self.phi = np.zeros((self.N, self.N, self.N))
            self.rho = np.zeros((self.N, self.N, self.N))
        elif self.problem == 'magnetic':
            self.phi = np.zeros((self.N, self.N))
            self.rho = np.zeros((self.N, self.N))
        self.dx = dx

        self.method = method
        self.omega = omega
        self.tolerance = tolerance

        self.iters = 0
        self.converged = False

    def initialise(self, initial_state: str) -> None:
        """
        Initialise with monopole/wire/gaussian.
        """
        if initial_state == 'monopole':
            self.rho[self.N//2, self.N//2, self.N//2] = 1.0
        elif initial_state == 'wire':
            self.rho[self.N//2, self.N//2] = 1.0
        elif initial_state == 'gaussian':
            x = np.arange(self.N) - self.N//2 # from -25 -> 25
            X, Y, Z = np.meshgrid(x, x, x, indexing='ij') # indexing ij adheres to array indices rather than matrix indices
            sigma = 2.0  
            self.rho = np.exp(-(X**2 + Y**2 + Z**2) / (2*sigma**2))
            self.rho = self.rho / (np.sum(self.rho) * self.dx**3) # normalise

    def get_midplane(self) -> np.ndarray:
        if self.problem == 'electric':
            return self.phi[self.N//2, :, :]
        elif self.problem == 'magnetic':
            return self.phi
        
    def compute_field(self) -> tuple[np.ndarray, ...]:
        """
        Compute E/B fields where E = -grad(phi) & B = curl(A)
        """
        if self.problem == 'electric':
            Ex, Ey, Ez = compute_electric_field(self.phi, self.N, self.dx)
            return Ex, Ey, Ez
        elif self.problem == 'magnetic':
            Bx, By = compute_magnetic_field(self.phi, self.N, self.dx)
            return Bx, By  
    
    def plot_potential_contour(self) -> None:
        """
        Potential contour plot.
        """
        fig, ax = plt.subplots()

        cf = ax.contourf(self.get_midplane(), levels=50, cmap='viridis') # contourf handles filled contours
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        label = r"$\phi$" if self.problem == 'electric' else r"$A_z$"
        cbar.set_label(label)

        problem_label = self.problem.capitalize()
        ax.set_title(f"{problem_label} " + r"Potential $\phi$ (Midplane))")
        plt.savefig(f"{self.problem}_potential_contour.png", dpi=300, bbox_inches='tight')
    
    def plot_field(self) -> None:
        """
        Field strength plot.
        """
        field = self.compute_field()

        if self.problem == 'electric':
            mid = self.N // 2
            v = field[1][mid, :, :]
            u = field[2][mid, :, :]
        elif self.problem == 'magnetic':
            u, v = field[1], field[0]

        mag = np.sqrt(u**2 + v**2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if self.problem == 'magnetic':
            scale = 1.5
        else:
            scale = 0.4

        ax1.quiver(u, v, scale=scale)
        ax1.set_title(f"{self.problem.capitalize()} Field (Vector)")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")

        img = ax2.imshow(mag, cmap=f"plasma", origin=f"lower")
        ax2.set_title(f"{self.problem.capitalize()} Field (Magnitude)")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")

        cbar = fig.colorbar(img, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(r'$|E|$' if self.problem == 'electric' else r'$|B|$')

        plt.savefig(f"{self.problem}_field_plot.png", dpi=300, bbox_inches='tight')

    @staticmethod
    def electric_potential_model(r: np.ndarray, m: float, c: float):
        return (m/r) + c
    
    @staticmethod
    def magnetic_potential_model(r: np.ndarray, m: float, c: float):
        return (m*np.log(r)) + c

    @staticmethod
    def electric_field_model(r: np.ndarray, m: float, c: float):
        return (m/r**2) + c
    
    @staticmethod
    def magnetic_field_model(r: np.ndarray, m: float, c: float):
        return (m/r) + c

    def plot_and_fit_radial(self, save: bool = False):
        """
        Fits radial curves to potential and field, then plots the fit.
        """
        mid = self.N // 2

        r = np.arange(1, mid)

        if self.problem == 'electric':
            phi_r = self.phi[mid, mid, mid+1:mid+mid]
            Ex, Ey, Ez, = self.compute_field()
            E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            field = E_mag[mid, mid, mid+1:mid+mid]

            (phi_m, phi_c), _ = curve_fit(self.electric_potential_model, r, phi_r)
            (field_m, field_c), _ = curve_fit(self.electric_field_model, r, field)

            phi_data = self.electric_potential_model(r, phi_m, phi_c)
            field_data = self.electric_field_model(r, field_m, field_c)

        elif self.problem == 'magnetic':
            phi_r = self.phi[mid, mid+1:mid+mid]
            Bx, By = self.compute_field()
            B_mag = np.sqrt(Bx**2 + By**2)
            field = B_mag[mid, mid+1:mid+mid]

            (phi_m, phi_c), _ = curve_fit(self.magnetic_potential_model, r, phi_r)
            (field_m, field_c), _ = curve_fit(self.magnetic_field_model, r, field)

            phi_data = self.magnetic_potential_model(r, phi_m, phi_c)
            field_data = self.magnetic_field_model(r, field_m, field_c)

        problem_label = self.problem.capitalize()
        potential_label = r"$\phi$" if self.problem == 'electric' else r"A_z"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.plot(r, phi_data, color='r', label='Best-Fit')
        ax1.scatter(r, phi_r, s=10, label='Data')
        ax1.set_ylabel(potential_label)
        ax1.set_title(f"{problem_label} Potential")

        ax2.plot(r, field_data, color='b', label='Best-Fit')
        ax2.scatter(r, field, s=10, label='Data')
        ax2.set_ylabel(f"Dimensionless Field Strength")
        ax2.set_title(f"{problem_label} Field Strength")

        for ax in [ax1, ax2]:
            ax.set_xlabel(r"Radial Distance $r$")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
        
        if self.problem == 'magnetic':
            ax1.set_yscale('linear')

        if save:
            plt.savefig(f"{self.problem}_radial_fits.png", dpi=300, bbox_inches='tight')
            df = pd.DataFrame({'r': r, 'phi': phi_r, 'field': field})
            df.to_csv(f'{self.problem}_radial.csv', index=False)

    def step(self) -> None:
        """
        Computes one step of the Jacobi algorithm.
        """
        phi0 = self.phi.copy()
        if self.method == 'jacobi':
            if self.problem == 'electric':
                self.phi = _electric_potential_jacobi(phi0, self.rho, self.dx, self.N)
            elif self.problem == 'magnetic':
                self.phi = _magnetic_potential_jacobi(phi0, self.rho, self.dx, self.N)
        elif self.method == 'gs':
            if self.problem == 'electric':
                _electric_potential_gauss_seidel(self.phi, self.rho, self.dx, self.N, self.omega)
            elif self.problem == 'magnetic':
                _magnetic_potential_gauss_seidel(self.phi, self.rho, self.dx, self.N, self.omega)
        self.iters += 1

        if not self.converged and np.max(np.abs(self.phi - phi0)) < self.tolerance:
            self.converged=True
            print(f"Converged in {self.iters} iterations.")

    def _animate_sweep(self, frames: int) -> list:
        """
        Animates a sweep (defined as the number of steps in range()).
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(20):
            self.step()
        midplane = self.get_midplane()
        self.im.set_data(midplane)
        self.im.set_clim(vmin=midplane.min(), vmax=midplane.max())
        self.ax.set_title(r"$N_{\mathrm{iters}}$ = " + f"{self.iters}")
        return [self.im]
    
    def run(self, n_steps: int, animate: bool = False):
        """
        Run the Poisson solver.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.get_midplane(), cmap='viridis')
            plt.colorbar(self.im)
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=n_steps // 50, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for _ in range(n_steps):
                self.step()
                if self.converged:
                    break

    def measure(self) -> None:
        """
        Run data collection, plotting, and saving.
        """
        self.plot_potential_contour()
        self.plot_field()
        self.plot_and_fit_radial(save=True)
        plt.show()

    @staticmethod
    def _sor_single_run(problem: str, tolerance: float, omega: float, N: int, 
                        dx: float, rho: np.ndarray):
        """
        Performs a single run of the SOR GS.
        Standalone function for parallelised data collection.
        """
        po = Poisson(problem=problem, method='gs', tolerance=tolerance,
                    omega=omega, N=N, dx=dx)
        po.rho = rho.copy()
        for _ in range(100000):
            po.step()
            if po.converged:
                break
        return omega, po.iters

    def measure_sor(self) -> None:
        """
        Sweep over omega values and record iterations to convergence.
        """
        omega_range = np.linspace(1.75, 1.95, 40) # generous omega range, can drop if performance requires
        results = Parallel(n_jobs=-1)(
            delayed(self._sor_single_run)(
                self.problem, self.tolerance, omega, self.N, self.dx, self.rho
            )
            for omega in omega_range
        )
        
        omegas, iters = zip(*results)
        
        fig, ax = plt.subplots()
        min_idx = np.argmin(iters)
        ax.plot(omegas, iters, marker='o', color='r', label=f"ω={omegas[min_idx]:.2f}, n={iters[min_idx]}")
        ax.legend()
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('Iterations to Convergence')
        ax.set_title('SOR Convergence')
        
        df = pd.DataFrame({'omega': omegas, 'iterations': iters})
        df.to_csv('sor_convergence.csv', index=False)
        
        plt.show()

parser = argparse.ArgumentParser(description='Partial Differential Equations')

parser.add_argument('--N', type=int, default=50)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--dx', type=float, required=True)
parser.add_argument('--n_steps', type=int, default=10000, required=True)
parser.add_argument('--measure', action='store_true')

subparsers = parser.add_subparsers(dest='equation', required=True)

cahn_parser = subparsers.add_parser('C-H')
cahn_parser.add_argument('--phi0', type=float, required=True)
cahn_parser.add_argument('--dt', type=float, required=True)
cahn_parser.add_argument('--measure_interval', type=int, default=10)

poisson_parser = subparsers.add_parser('Poisson')
poisson_parser.add_argument('--problem', type=str, choices=['magnetic', 'electric'])
poisson_parser.add_argument('--method', type=str, choices=['gs', 'jacobi'])
poisson_parser.add_argument('--tol', type=float, default=1e-3)
poisson_parser.add_argument('--omega', type=float, default=1.0)
poisson_parser.add_argument('--initial_state', type=str, choices=['monopole', 'wire', 'gaussian'])

def main():

    args = parser.parse_args()

    if args.equation == 'C-H':
        if args.measure:
            results = Parallel(n_jobs=2)(
                delayed(CahnHilliard.data_collection)(
                    p, args.N, args.dx, args.dt,
                    args.n_steps, args.measure_interval
                )
                for p in [0.0, 0.5]
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            for ax, (time_vals, fe_vals), phi0 in zip([ax1, ax2], results, [0.0, 0.5]):
                ax.plot(time_vals, fe_vals)
                ax.set_title(r"Cahn-Hilliard Free Energy against Time for $\phi_0 = $" + f"{phi0}")
                ax.set_xlabel(f"Dimensionless Time")
                ax.set_ylabel("Dimensionless Free Energy")

            fig.tight_layout()
            plt.savefig("cahn_hilliard_free_energy.png", dpi=300, bbox_inches='tight')
            plt.show()

        else:
            ch = CahnHilliard(phi0=args.phi0, N=args.N, dx=args.dx, dt=args.dt)
            ch.run(n_steps=args.n_steps,
                   measure_interval=args.measure_interval,
                   animate=args.animate)
            
    elif args.equation == 'Poisson':

        if args.measure:
            po_mag = Poisson(problem='magnetic', method='gs', tolerance=args.tol, omega=1,
                             N=args.N, dx=args.dx)
            po_mag.initialise(initial_state='wire')
            po_mag.run(n_steps=10000)
            po_mag.measure()
            po_mag.measure_sor() # checkpoint says SOR data can be done either with E or B so I'm going with B (2D, faster)

            po_el = Poisson(problem='electric', method='gs', tolerance=args.tol, omega=1,
                            N=args.N, dx=args.dx)
            po_el.initialise(initial_state='monopole')
            po_el.run(n_steps=10000)
            po_el.measure()

        else:
            po = Poisson(problem=args.problem, method=args.method, tolerance=args.tol, omega=args.omega, 
                        N=args.N, dx=args.dx)
            po.initialise(initial_state=args.initial_state)
            po.run(n_steps=args.n_steps, animate=args.animate)

if __name__ == '__main__':
    main()