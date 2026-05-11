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
                              ((fz[i,j,(k+1)%N] - fz[i,j,(k-1)%N] / (2*dx))))
                
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
