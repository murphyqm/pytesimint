#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define Matrix Script

Created on 23/06/2021
by murphyqm

"""
from numba import jit, njit
import numpy as np
import scipy.sparse.linalg
from functools import wraps
import time


def timefn(fn):
    """Decorator to time function."""
    @wraps(fn)  # to expose function name of decorated function
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time


# this is used
class TridiagMatrixCNDirichlet:
    """Sparse matrix object for Crank-Nicolson scheme w/Dirichlet bcs."""

    def __init__(self, main, upper, lower, N):
        """
        Instantiate Tridiagonal matrix object with Dirichlet bcs

        Parameters
        ----------
        main
        upper
        lower
        N
        """
        self.main = main
        self.upper = upper
        self.lower = lower
        self.N = N
        main_band = np.zeros(N - 2)
        main_band[:] = main[1:-1]
        lower_band = np.zeros(N - 3)
        lower_band[:] = lower[2:-1]
        upper_band = np.zeros(N - 3)
        upper_band[:] = upper[1: -2]
        self.mat = scipy.sparse.diags(
            diagonals=[main_band, lower_band, upper_band],
            offsets=[0, -1, 1], shape=(N - 2, N - 2),
            format='csr')

    def __str__(self):
        """Print description of sparse matrix object."""
        dense_print = self.mat.todense()
        return f"Sparse matrix of size {self.N - 2}x{self.N - 2}: \n {dense_print}"

    def densify(self):
        """Return dense version of sparse matrix."""
        return self.mat.todense()


# @timefn
# this is used
class TridiagMatrixCNNeumann:
    """Sparse matrix object for Crank-Nicolson scheme w/Neumann bcs."""

    def __init__(self, main, upper, lower, N):
        """
        Instantiate Tridiagonal matrix object with Neumann boundary conditions

        Parameters
        ----------
        main
        upper
        lower
        N
        """
        self.main = main
        self.upper = upper
        self.lower = lower
        self.N = N
        main_band = np.zeros(N)
        main_band[:] = main[:]
        lower_band = np.zeros(N - 1)
        lower_band[:] = lower[1:]
        lower_band[-1] = lower_band[-1] * 2.0
        upper_band = np.zeros(N - 1)
        upper_band[:] = upper[0:-1]
        upper_band[0] = upper_band[0] * 2.0
        self.mat = scipy.sparse.diags(
            diagonals=[main_band, lower_band, upper_band],
            offsets=[0, -1, 1], shape=(N, N),
            format='csr')

    def __str__(self):
        """Print description of sparse matrix object."""
        dense_print = self.mat.todense()
        return f"Sparse matrix of size {self.N}x{self.N}: \n {dense_print}"

    def densify(self):
        """Return dense version of sparse matrix."""
        return self.mat.todense()


# @njit(nopython=True)
# @timefn
# this is used
@jit(nopython=True)
def initialise_r_vector(dt, dx, diffusivities):
    """Returns an array of r values given spatially varying diffusivity."""
    return (diffusivities * dt) / (dx ** 2)


@jit(nopython=True)
def spvr_1d_initialise_r_vector(dt, dx, diffusivities):
    """Returns an array of r values given spatially varying diffusivity."""
    r_values = (diffusivities * dt) / (dx ** 2)
    # r_values = np.insert(r_values,
    #                      [0, len(r_values)],
    #                      [r_values[0], r_values[-1]])
    r_values = np.concatenate(
        (np.asarray([r_values[0]]),
         r_values,
         np.asarray([r_values[-1]])))
    return r_values


@jit(nopython=True)
def spvr_3d_initialise_r_vector(dt, dx, diffusivities):
    """Returns an array of r values given spatially varying diffusivity."""
    val = diffusivities[0, 0, 0]
    dim0 = diffusivities.shape[0] + 2
    dim1 = diffusivities.shape[1] + 2
    dim2 = diffusivities.shape[2] + 2
    new_array = np.full((dim0, dim1, dim2), val)
    new_array[1:-1, 1:-1, 1:-1] = diffusivities
    r_values = (new_array * dt) / (dx ** 2)
    return r_values

# @timefn
# this is used
@jit(nopython=True)
def initialise_diagonals(r_values, N):
    """Returns arrays of diagonal values for matrices given r arrays."""
    # A matrix: 2 + 2r on the diagonal, -r on the upper and lower
    # B matrix: 2 - 2r on the diagonal, +r on the upper and lower
    if len(r_values) != N:
        print("Warning, diffusivity array is wrong length, please check.")
    # main_diag_A = np.asarray([2.0+(2.0*r) for r in r_values])
    # upper_diag_A = np.asarray([-1.0 * r for r in r_values])
    # lower_diag_A = np.asarray([-1.0 * r for r in r_values])
    # main_diag_B = np.asarray([2.0 - (2.0 * r) for r in r_values])
    # upper_diag_B = np.asarray([r for r in r_values])
    # lower_diag_B = np.asarray([r for r in r_values])

    # main_diag_A = np.asarray([2.0+(2.0*r) for r in r_values])
    main_diag_A = 2.0 + (2.0 * r_values)
    # upper_diag_A = np.asarray([-1.0 * r for r in r_values])
    upper_diag_A = -1 * r_values
    # lower_diag_A = np.asarray([-1.0 * r for r in r_values])
    lower_diag_A = -1 * r_values

    # main_diag_B = np.asarray([2.0 - (2.0 * r) for r in r_values])
    main_diag_B = 2.0 - (2.0 * r_values)
    # upper_diag_B = np.asarray([r for r in r_values])
    upper_diag_B = r_values
    # lower_diag_B = np.asarray([r for r in r_values])
    lower_diag_B = r_values

    return (main_diag_A,
            upper_diag_A,
            lower_diag_A,
            main_diag_B,
            upper_diag_B,
            lower_diag_B)


@jit(nopython=True)
def spvr_initialise_diagonals(r_values, N):
    """Returns arrays of diagonal values for matrices given r arrays."""
    # A matrix: 2 + 2r on the diagonal, -r on the upper and lower
    # B matrix: 2 - 2r on the diagonal, +r on the upper and lower
    if len(r_values) != N+2:
        print("Warning, diffusivity array is wrong length, please check.")
        print("Diffusivity array should include 2 points outside domain.")

    ru_array = np.empty(N)
    rl_array = np.empty(N)
    rd_array = np.empty(N)

    for i in range(1, len(r_values)-1):
        j = i - 1
        ru_array[j] = r_values[i] + r_values[i + 1]
        rl_array[j] = r_values[i] + r_values[i - 1]
        rd_array[j] = (2 * r_values[i]) + r_values[i + 1] + r_values[i - 1]

    main_diag_A = 4.0 + rd_array
    upper_diag_A = -1.0 * ru_array
    lower_diag_A = -1.0 * rl_array

    main_diag_B = 4.0 - rd_array
    upper_diag_B = ru_array
    lower_diag_B = rl_array

    return (main_diag_A,
            upper_diag_A,
            lower_diag_A,
            main_diag_B,
            upper_diag_B,
            lower_diag_B)


# @jit
# @timefn
# this is used
def initialise_matrices_var_dif(main_diag_A,
            upper_diag_A,
            lower_diag_A,
            main_diag_B,
            upper_diag_B,
            lower_diag_B, N, boundary_cond='d'):
    """
    Returns A and B matrices given upper, lower and main diagonals.

    Parameters
    ----------
    main_diag_A
    upper_diag_A
    lower_diag_A
    main_diag_B
    upper_diag_B
    lower_diag_B
    N
    boundary_cond

    Returns
    -------

    """
    if boundary_cond == 'd':
        A = TridiagMatrixCNDirichlet(main_diag_A, upper_diag_A, lower_diag_A, N)
        B = TridiagMatrixCNDirichlet(main_diag_B, upper_diag_B, lower_diag_B, N)

    elif boundary_cond == 'n':
        A = TridiagMatrixCNNeumann(main_diag_A, upper_diag_A, lower_diag_A, N)
        B = TridiagMatrixCNNeumann(main_diag_B, upper_diag_B, lower_diag_B, N)

    return A, B


# @jit(nopython=True)
# @timefn
# this is used
def IBCs_Dirichlet(initial_values, r_values, N):
    """Returns boundary arrays for fixed temp boundary conditions."""
    u = np.asarray(initial_values)
    b1 = np.zeros((N - 2))
    b1[0] = r_values[0] * initial_values[0]
    b1[-1] = r_values[-1] * initial_values[-1]
    b2 = b1.copy()
    return u, b1, b2


# this is used
@jit(nopython=True)
def IBCs_Neumann_zero_flux(initial_values, r_values, N):
    """Returns boundary arrays for zero flux boundary conditions."""
    u = np.asarray(initial_values)
    b1 = np.zeros(N) # zero flux
    b2 = np.zeros(N) # zero flux
    return u, b1, b2


# this is used
class LinearSystemCN:
    """Simple Linear System object for Crank-Nicolson scheme."""

    def __init__(self, A, B, b, b2, x, u, bc='d'):
        """
        Instantiate Linear System for Crank-Nicolson method.
        Parameters
        ----------
        A
        B
        b
        b2
        x
        u
        """
        self.A = A
        self.B = B
        self.b = b
        self.b2 = b2
        self.x = x
        self.initial = u
        self.u = u
        self.bc = bc
        # self.RHS = B.mat.dot(u[1:-1]) + b + b2  # at t = 0
        if bc == 'n':
            self.RHS = B.mat.dot(u) + b + b2  # at t = 0
        elif bc == 'd':
            self.RHS = B.mat.dot(u[1:-1]) + b + b2  # at t = 0
        else:
            print("Enter valid boundary conditions.")

    def __str__(self):
        return f"""Linear system of matrices for Crank Nicholson method.
    Matrix A:
    {self.A.mat.todense()};
    Matrix B:
    {self.B.mat.todense()};
    Column vector b at t=j:
    {self.b};
    Column vector b2 at t=j+1:
    {self.b2};
    Initial conditions:
    {self.initial}"""

    def solve(self, j):
        if self.bc == 'n':
            self.u = scipy.sparse.linalg.spsolve(self.A.mat, self.RHS)
            self.RHS = self.B.mat.dot(self.u) + self.b + self.b2
        else:
            self.u[1:-1] = scipy.sparse.linalg.spsolve(self.A.mat, self.RHS)
            self.RHS = self.B.mat.dot(self.u[1:-1]) + self.b + self.b2


# this is used
def cn_solver_zero(system: object, x, nsteps, u):
    solution = np.zeros((x.size, nsteps+1))
    solution[:, 0] = u

    c = 0
    for j in range(1, nsteps+1):
        system.solve(j)
        solution[:, j] = system.u
    return solution[:, j]


# new functions from jupyter notebook examples

# this is used
# @timefn
@jit(nopython=True)
def define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    blank_vol = np.ones((Nx, Ny, Nz))
    return x, y, z, blank_vol


# @timefn
@jit(nopython=True)
def spacing(n_grid_points, grid_length):
    d = grid_length/(n_grid_points - 1)
    return d


# this is used
# @timefn
@jit(nopython=True)
def set_grid_values_3d(x, y, z, blank_vol, x1, x2, y1, y2, z1, z2, int_value, mant_value):
    new_vol = np.full_like(blank_vol, mant_value)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if x[i] >= x1 and x[i] <= x2 and y[j] >= y1 and y[j] <= y2 and z[k] >= z1 and z[k] <= z2:
                    new_vol[i, j, k] = int_value
    return new_vol


# this is used
# @timefn
@jit(nopython=True)
def set_grid_values_3d_rounded(x, y, z,
                               blank_vol,
                               d, e, f,
                               a, b, c,
                               int_value,
                               mant_value):
    new_vol = np.full_like(blank_vol, mant_value)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if (((x[i] - d)**2)/(a**2)) + \
                 (((y[j] - e)**2)/(b**2)) + \
                 (((z[k] - f)**2)/(c**2)) <= 1:
                    new_vol[i, j, k] = int_value
    return new_vol
