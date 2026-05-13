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

#
# engine room
#

@njit(parallel=True)
def compute_laplacian_1D(f: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 1D Laplacian of physical parameter f on a grid.
    """
    laplacian = np.zeros_like(f)
    for i in prange(N):
        laplacian[i] = (f[(i+1)%N] + f[(i-1)%N] - 2*f[i]) / dx**2

    return laplacian

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
def compute_laplacian_3D(f: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 3D Laplacian of physical parameter f on a grid.
    """
    laplacian = np.zeros_like(f)
    for i in prange(N):
        for j in range(N):
            for k in range(N):

                laplacian[i, j, k] = (f[(i+1)%N,j,k] + f[(i-1)%N,j,k] +
                                     f[i,(j+1)%N,k] + f[i,(j-1)%N,k] +
                                     f[i,j,(k+1)%N] + f[i,j,(k-1)%N] - 
                                    6*f[i,j,k]) / dx**2

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
            
            grad_x[i,j] = (f[(i+1)%N, j] - f[(i-1)%N, j]) / (2*dx)
            grad_y[i,j] = (f[i, (j+1)%N] - f[i, (j-1)%N]) / (2*dx)

    return grad_x, grad_y

@njit(parallel=True)
def compute_grad_3D(f: np.ndarray, N: int, dx: float) -> tuple[np.ndarray, ...]:
    """
    Computes the 3D grad of physical parameter f on a grid.
    """
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)
    grad_z = np.zeros_like(f)

    for i in prange(N):
        for j in range(N):
            for k in range(N):

                grad_x[i,j,k] = (f[(i+1)%N,j,k] - f[(i-1)%N,j,k]) / (2*dx)
                grad_y[i,j,k] = (f[i,(j+1)%N,k] - f[i,(j-1)%N,k]) / (2*dx)
                grad_z[i,j,k] = (f[i,j,(k+1)%N] - f[i,j,(k-1)%N]) / (2*dx)

    return grad_x, grad_y, grad_z

@njit(parallel=True)
def compute_div_2D(fx: np.ndarray, fy: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 2D divergence of physical parameter f(x,y) with inputs fx & fy on a grid.
    """
    div = np.zeros_like(fx)

    for i in prange(N):
        for j in range(N):
            
            div[i,j] = (((fx[(i+1)%N,j] - fx[(i-1)%N,j]) / (2*dx)) + 
                        ((fy[i,(j+1)%N] - fy[i,(j-1)%N]) / (2*dx)))
            
    return div

@njit(parallel=True)
def compute_div_3D(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 3D divergence of physical parameter f(x,y,z) with inputs fx, fy, fz on a grid.
    """
    div = np.zeros_like(fx)

    for i in prange(N):
        for j in range(N):
            for k in range(N):

                div[i,j,k] = (((fx[(i+1)%N,j,k] - fx[(i-1)%N,j,k]) / (2*dx)) + 
                              ((fy[i,(j+1)%N,k] - fy[i,(j-1)%N,k]) / (2*dx)) +
                              ((fz[i,j,(k+1)%N] - fz[i,j,(k-1)%N]) / (2*dx)))
                
    return div

@njit(parallel=True)
def compute_curl_2D(fx: np.ndarray, fy: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the 2D curl of a physical parameter f(x,y) with inputs fx, fy on a grid.
    """
    curl = np.zeros_like(fx)

    for i in prange(N):
        for j in range(N):

            curl[i,j] = (((fy[(i+1)%N,j] - fy[(i-1)%N,j])) / (2*dx) - 
                        ((fx[i,(j+1)%N] - fx[i,(j-1)%N])) / (2*dx))
            
    return curl

@njit(parallel=True)
def compute_curl_3D(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, N: int, dx: float) -> tuple[np.ndarray, ...]:
    """
    Computes the 3D curl of a physical parameter f(x,yz) with inputs fx, fy, fz on a grid.
    """
    curl_x = np.zeros_like(fx)
    curl_y = np.zeros_like(fy)
    curl_z = np.zeros_like(fz)

    for i in prange(N):
        for j in range(N):
            for k in range(N):

                curl_x[i,j,k] = (((fz[i,(j+1)%N,k] - fz[i,(j-1)%N,k]) -
                                 (fy[i,j,(k+1)%N] - fy[i,j,(k-1)%N])) / (2*dx))
                curl_y[i,j,k] = (((fx[i,j,(k+1)%N] - fx[i,j,(k-1)%N]) -
                                 (fz[(i+1)%N,j,k] - fz[(i-1)%N,j,k])) / (2*dx))
                curl_z[i,j,k] = (((fy[(i+1)%N,j,k] - fy[(i-1)%N,j,k]) -
                                 (fx[i,(j+1)%N,k] - fx[i,(j-1)%N,k])) / (2*dx))
                
    return curl_x, curl_y, curl_z

@njit(parallel=True)
def jacobi_3D_dirichlet(f0: np.ndarray, rho: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Jacobi algorithm for a quantity in 3D with Dirichlet BCs.
    """
    f_new = np.zeros_like(f0) # zeros_like instead of empty_like applies dirichlet intrinsically
    # range accomodates dirichlet BCs: edges = 0 always
    for i in prange(1, N-1): # only do prange for outer loop 
        for j in range(1, N-1):
            for k in range(1, N-1):
                f_new[i,j,k] = ((f0[i+1,j,k] + f0[i-1,j,k]
                              + f0[i,j+1,k] + f0[i,j-1,k]
                              + f0[i,j,k+1] + f0[i,j,k-1])
                              + dx**2 * rho[i,j,k]) / 6 # 6 nearest neighbours
    return f_new

@njit(parallel=True)
def jacobi_2D_dirichlet(f0: np.ndarray, rho: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Jacobi algorithm for a quantity in 2D with Dirichlet BCs.
    """
    f_new = np.zeros_like(f0) 
    for i in prange(1, N-1):
        for j in range(1, N-1):

            f_new[i,j] = ((f0[i+1,j] + f0[i-1,j]
                            + f0[i,j+1] + f0[i,j-1])
                            + dx**2 * rho[i,j]) / 4 # 4 nearest neighbours
    return f_new

@njit
def gauss_seidel_3D_dirichlet(f: np.ndarray, rho: np.ndarray, dx: float, N: int, 
                              omega: float = 1) -> None:
    """
    Gauss-Seidel algorithm for quantity in 3D with Dirichlet BCs.
    Omega controls successive over-relaxation (default 1, reduces to Gauss-Seidel).
    Sequential, do not parallelise.
    """
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):

                f_old = f[i,j,k]
                f_gs = ((f[i+1,j,k] + f[i-1,j,k]
                              + f[i,j+1,k] + f[i,j-1,k]
                              + f[i,j,k+1] + f[i,j,k-1])
                              + dx**2 * rho[i,j,k]) / 6
                
                f[i,j,k] = omega * f_gs + (1 - omega)*f_old

@njit
def gauss_seidel_2D_dirichlet(f: np.ndarray, rho: np.ndarray, dx: float, N: int, 
                              omega: float = 1) -> None:
    """
    Gauss-Seidel algorithm for quantity in 2D with Dirichlet BCs.
    Omega controls successive over-relaxation (default 1, reduces to Gauss-Seidel).
    Sequential, do not parallelise.
    """
    for i in range(1, N-1):
        for j in range(1, N-1):
            f_old = f[i,j]
            f_gs = ((f[i+1,j] + f[i-1,j]
                        + f[i,j+1] + f[i,j-1])
                        + dx**2 * rho[i,j]) / 4
            f[i,j] = omega * f_gs + (1 - omega)*f_old

def apply_dirichlet_bc_2D(f: np.ndarray, val: float = 0.0) -> None:
    """
    Applies the Dirichlet boundary condition to a 2D array: boundaries = val.
    The default value is 0.
    """
    f[0,:] = val
    f[-1,:] = val
    f[:,0] = val
    f[:,-1] = val

def apply_dirichlet_bc_3D(f: np.ndarray, val: float = 0.0) -> None:
    """
    Applies the Dirichlet boundary condition to a 3D array: boundaries = val.
    The default value is 0.
    """
    f[0,:,:] = val
    f[-1,:,:] = val
    f[:,0,:] = val
    f[:,-1,:] = val
    f[:,:,0] = val
    f[:,:,-1] = val

def apply_neumann_bc_2D(f: np.ndarray, dx: float, val: float = 0.0) -> None:
    """
    Applies the Neumann boundary condition to a 2D array: derivate at boundaries = val.
    The default value is 0.
    """
    f[0,:] = f[1,:] + val*dx 
    f[-1,:] = f[-2,:] + val*dx 
    f[:,0] = f[:,1] + val*dx
    f[:,-1] = f[:,-2] + val*dx 

def apply_neumann_bc_3D(f: np.ndarray, dx: float, val: float = 0.0) -> None:
    """
    Applies the Neumann boundary condition to a 3D array: derivate at boundaries = val.
    The default value is 0.
    """
    f[0,:,:] = f[1,:,:] + val*dx 
    f[-1,:,:] = f[-2,:,:] + val*dx 
    f[:,0,:] = f[:,1,:] + val*dx
    f[:,-1,:] = f[:,-2,:] + val*dx 
    f[:,:,0] = f[:,:,1] + val*dx 
    f[:,:,-1] = f[:,:,-2] + val*dx

def apply_dirichlet_bc_2D(f: np.ndarray, edge: str, val: float = 0.0):
    """
    Dirichlet on a single edge.
    """
    if edge == 'left':    f[0, :] = val
    elif edge == 'right': f[-1, :] = val
    elif edge == 'bottom': f[:, 0] = val
    elif edge == 'top':    f[:, -1] = val

def apply_neumann_bc_2D(f: np.ndarray, edge: str, dx: float, val: float = 0.0):
    """
    Neumann (df/dn = val) on a single edge.
    """
    if edge == 'left':    f[0, :] = f[1, :] + val * dx
    elif edge == 'right': f[-1, :] = f[-2, :] + val * dx
    elif edge == 'bottom': f[:, 0] = f[:, 1] + val * dx
    elif edge == 'top':    f[:, -1] = f[:, -2] + val * dx

def apply_dirichlet_bc_3D(f: np.ndarray, edge: str, val: float = 0.0):
    """
    Dirichlet on a single face.
    """
    if edge == 'left':    f[0, :, :] = val
    elif edge == 'right': f[-1, :, :] = val
    elif edge == 'bottom': f[:, 0, :] = val
    elif edge == 'top':    f[:, -1, :] = val
    elif edge == 'front':  f[:, :, 0] = val
    elif edge == 'back':   f[:, :, -1] = val

def apply_neumann_bc_3D(f: np.ndarray, edge: str, dx: float, val: float = 0.0):
    """
    Neumann (df/dn = val) on a single face.
    """
    if edge == 'left':    f[0, :, :] = f[1, :, :] + val * dx
    elif edge == 'right': f[-1, :, :] = f[-2, :, :] + val * dx
    elif edge == 'bottom': f[:, 0, :] = f[:, 1, :] + val * dx
    elif edge == 'top':    f[:, -1, :] = f[:, -2, :] + val * dx
    elif edge == 'front':  f[:, :, 0] = f[:, :, 1] + val * dx
    elif edge == 'back':   f[:, :, -1] = f[:, :, -2] + val * dx

def spatial_integral(f: np.ndarray, dx: float) -> float:
    """
    Integrates over the grid by approximating each cell's area.
    Automatically handles the number of dimensions.
    """
    return np.sum(f) * dx**f.ndim

def find_stable_timestep(D: float, dx: float, ndim: int = 2, safety_threshold: float = 0.9):
    """
    Uses the Von Neumann analysis to find a (somewhat conservative) stable timestep for desired parameters.
    """
    return safety_threshold * dx**2 / (2 * ndim * D)

#
# C-H specific.
#

@njit
def compute_mu(phi: np.ndarray, N: int, dx: float) -> np.ndarray:
    """
    Computes the discretised chemical potential.
    """
    return -phi + phi**3 - compute_laplacian_2D(f=phi, N=N, dx=dx)

@njit
def cahn_step(phi: np.ndarray, N: int, dt: float, dx: float) -> np.ndarray:
    """
    Performs one forward Euler for C-H.
    """
    mu = compute_mu(phi=phi, N=N, dx=dx)
    return phi + dt * compute_laplacian_2D(f=mu, N=N, dx=dx)
    
@njit
def total_free_energy(phi: np.ndarray, N: int, dx: float) -> float:
    """
    Computes the discretised free energy.
    """
    grad_x, grad_y = compute_grad_2D(f=phi, N=N, dx=dx)
    f = (-1/2 * phi**2) + (1/4 * phi**4) + (1/2 * (grad_x**2 + grad_y**2))
    return np.sum(f) * dx**2

class InitialValueProblem:
    """
    Demonstrated via the Cahn-Hilliard equation.
    """
    def __init__(self, phi0: float, N: int = 100, dx: float = 1.0, dt: float = 0.01):

        self.N = N
        self.dx = dx
        self.dt = dt
        self.time = 0.0
        self.initialise_grid(phi0=phi0)

    def initialise_grid(self, phi0: float) -> np.ndarray:
        """
        Initialises the grid according to specified format.
        """
        self.phi = np.random.uniform(phi0 - 0.1, phi0 + 0.1, (self.N, self.N))
    
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

    def run(self, animate: bool, max_steps: int):
        """
        Run the Cahn-Hilliard equation.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.phi, cmap='coolwarm', vmin=-1, vmax=1)
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_sweep,
                                    frames=max_steps // 50, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for _ in range(max_steps):
                self.step()

    @staticmethod
    def data_collection(phi0: float, N: int, dx: float, dt: float, 
                        n_steps: int, measure_interval: int) -> tuple[np.ndarray, ...]:
        """
        Wrapper run (for parallelised data collection).
        """
        free_energy_vals = []
        time_vals = []
        ch = InitialValueProblem(phi0=phi0, N=N, dx=dx, dt=dt) # this is c-h specific
        for i in range(n_steps):
            ch.step()
            if i % measure_interval == 0:
                free_energy_vals.append(total_free_energy(ch.phi, N, dx))
                time_vals.append(ch.time)

        return np.array(time_vals), np.array(free_energy_vals)
    
class BoundaryValueProblem:
    """
    Demonstrated via the Poisson solver.
    """

    def __init__(self, method: str, tolerance: float, omega: float = 1.0, 
                 ndim: int = 2, N: int = 100, dx: float = 1.0):

        self.N = N 
        self.dx = dx
        self.ndim = ndim

        self.phi = self.initialise_grid(N=N, val=0.0, ndim=ndim)
        self.rho = self.initialise_grid(N=N, val=0.0, ndim=ndim)

        self.method = method
        self.omega = omega
        self.tolerance = tolerance

        self.converged = False
        self.iters = 0

    def initialise_grid(self, N: int, val: float, ndim: int) -> np.ndarray:
        """
        Creates grid ndarray.
        """
        shape = tuple([N] * ndim)
        return np.full(shape=shape, fill_value=val)
    
    def initialise_rho(self, initial_state: str = 'monopole') -> None:
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
        """
        Helper function to return midplane of 3D array.
        """
        return self.phi[self.N//2, :, :]
    
    def compute_electric_field(self) -> tuple[np.ndarray, ...]:
        """
        Computes the electric field (3D).
        """
        grad_x, grad_y, grad_z = compute_grad_3D(f=self.phi, N=self.N, dx=self.dx)
        return -grad_x, -grad_y, -grad_z
    
    def compute_magnetic_field(self) -> tuple[np.ndarray, ...]:
        """
        Computes the magnetic field, which is the curl (2D).
        """
        grad_x, grad_y = compute_grad_2D(f=self.phi, N=self.N, dx=self.dx)
        return grad_y, -grad_x
    
    def plot_potential_contour(self) -> None:
        """
        Potential contour plot.
        """
        fig, ax = plt.subplots(figsize=(8,6))

        cf = ax.contourf(self.get_midplane(), levels=50, cmap='viridis') # contourf handles filled contours
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        label = r"$\phi$" # change label as needed
        cbar.set_label(label)

        ax.set_title(r"Arbitrary Potential $\phi$ (Midplane)")
        fig.savefig(f"potential_contour.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def step(self) -> None:
        """
        One step of the solver.
        """
        phi0 = self.phi.copy()
        if self.method == 'gs':
            if self.ndim == 2:
                gauss_seidel_2D_dirichlet(f=self.phi, rho=self.rho, dx=self.dx, N=self.N,
                                          omega=self.omega)
            elif self.ndim == 3:
                gauss_seidel_3D_dirichlet(f=self.phi, rho=self.rho, dx=self.dx, N=self.N,
                            omega=self.omega)
        elif self.method == 'jacobi':
            if self.ndim == 2:
                self.phi = jacobi_2D_dirichlet(f0=phi0, rho=self.rho, dx=self.dx, N=self.N)
            elif self.ndim == 3:
                self.phi = jacobi_3D_dirichlet(f0=phi0, rho=self.rho, dx=self.dx, N=self.N)

        self.iters += 1

        if not self.converged and np.max(np.abs(self.phi - phi0)) < self.tolerance:
            self.converged=True
            print(f"Converged in {self.iters} iterations.")

    def _animate_step(self, frames: int) -> list:
        """
        Animates a step.
        'frames' argument is necessary for FuncAnimation call.
        """
        for _ in range(20):
            self.step()
        if self.ndim == 2:
            midplane = self.phi
        elif self.ndim == 3:
            midplane = self.get_midplane()

        self.im.set_data(midplane)
        self.im.set_clim(vmin=midplane.min(), vmax=midplane.max())
        self.ax.set_title(r"$N_{\mathrm{iters}}$ = " + f"{self.iters}")

        return [self.im]

    def run(self, animate: bool, max_steps: int):
        """
        Run the Poisson solver.
        """
        if animate:
            self.fig, self.ax = plt.subplots()
            if self.ndim == 2:
                midplane = self.phi
            elif self.ndim == 3:
                midplane = self.get_midplane()
            self.im = self.ax.imshow(midplane, cmap='viridis')
            cbar = self.fig.colorbar(self.im)
            cbar.ax.set_title(r"$\phi$")
            self.anim = FuncAnimation(self.fig, self._animate_step,
                                    frames=max_steps // 50, interval=50,
                                    repeat=False)
            plt.show()

        else:
            for _ in range(max_steps):
                self.step()
                if self.converged:
                    break

    @staticmethod
    def _sor_single_run(tolerance: float, omega: float, N: int, dx: float, rho: np.ndarray):
        """
        Performs a single run of the SOR GS.
        Standalone function for parallelised data collection.
        """
        po = BoundaryValueProblem(method='gs', tolerance=tolerance, omega=omega, N=N, 
                                  dx=dx, ndim=rho.ndim)
        po.rho = rho.copy() # protects against parallel instances overwriting rho
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
        results = Parallel(n_jobs=-1, return_as='list')(
            delayed(self._sor_single_run)(self.tolerance, omega, self.N, self.dx, self.rho)
            for omega in omega_range
            )
        omegas, iters = zip(*results)
        
        fig, ax = plt.subplots(figsize=(8,6))
        min_idx = np.argmin(iters)
        ax.plot(omegas, iters, marker='o', color='r', label=f"ω={omegas[min_idx]:.2f}, n={iters[min_idx]}")
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('Iterations to Convergence')
        ax.set_title('SOR Convergence')
        fig.savefig("sor_convergence_plot.png", dpi=300)
        
        pd.DataFrame({'omega': omegas, 'iterations': iters}).to_csv('sor_convergence_data.csv', index=False)
    
        plt.show()        

parser = argparse.ArgumentParser(description='Partial Differential Equations')

parser.add_argument('--N', type=int, default=50)
parser.add_argument('--animate', action='store_true')
parser.add_argument('--dx', type=float, required=True)
parser.add_argument('--max_steps', type=int, default=10000, required=True)
parser.add_argument('--measure', action='store_true')

subparsers = parser.add_subparsers(dest='type', required=True)

initial_parser = subparsers.add_parser('Initial')
initial_parser.add_argument('--phi0', type=float, required=True)
initial_parser.add_argument('--dt', type=float, required=True)
initial_parser.add_argument('--measure_interval', type=int, default=10)

boundary_parser = subparsers.add_parser('Boundary')
boundary_parser.add_argument('--method', type=str, choices=['gs', 'jacobi'])
boundary_parser.add_argument('--tol', type=float, default=1e-3)
boundary_parser.add_argument('--omega', type=float, default=1.0)
boundary_parser.add_argument('--initial_state', type=str, choices=['monopole', 'wire', 'gaussian'])
boundary_parser.add_argument('--ndim', type=int, default=2)

def main():

    args = parser.parse_args()

    if args.type == 'Initial':
        if args.measure:

            results = Parallel(n_jobs=2)(
                    delayed(InitialValueProblem.data_collection)(
                        p, args.N, args.dx, args.dt,
                        args.max_steps, args.measure_interval
                    ) for p in [0.0, 0.5]
                    )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            for ax, (time_vals, fe_vals), phi0 in zip([ax1, ax2], results, [0.0, 0.5]):
                ax.plot(time_vals, fe_vals)
                ax.set_title(r"Cahn-Hilliard Free Energy against Time for $\phi_0 = $" + f"{phi0}")
                ax.set_xlabel(f"Dimensionless Time")
                ax.set_ylabel("Dimensionless Free Energy")
                pd.DataFrame({'t_vals': time_vals, 'fe_vals': fe_vals}).to_csv(f'{phi0}_free_energy_data.csv', index=False)

            fig.tight_layout()
            plt.savefig("free_energy_plot.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            ivp = InitialValueProblem(phi0=args.phi0, N=args.N, dx=args.dx, dt=args.dt)
            ivp.run(animate=args.animate, max_steps=args.max_steps)
    elif args.type == 'Boundary':
        if args.measure:
            po = BoundaryValueProblem(method='gs', tolerance=1e-6, ndim=args.ndim)
            po.initialise_rho(initial_state='monopole')
            po.measure_sor()
        else:
            po = BoundaryValueProblem(method=args.method, tolerance=args.tol, omega=args.omega, N=args.N, 
                                      dx=args.dx, ndim=args.ndim)
            po.initialise_rho(initial_state=args.initial_state)
            po.run(max_steps=args.max_steps, animate=args.animate)

if __name__ == "__main__":
    main()