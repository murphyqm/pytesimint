#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/10/2021
by murphyqm

"""
import numpy as np
from . import define_matrix as dm


def v7_iter_func(initial_temps,
                 initial_diffs: object,
                 location_of_intrusion,
                 app_diff,
                 boundary_cond,
                 x, y, z,
                 dx, dy, dz,
                 Nx, Ny, Nz,
                 dt,
                 folder,
                 iterations=4,
                 save_iter=1,
                 fileID="results"):
    full_soln = initial_temps
    assert dx == dy == dz

    filename = f"{folder}{fileID}_0"
    print("saving initial conditions")
    np.save(filename, full_soln)

    for h in range(1, iterations + 1):
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    if location_of_intrusion[i, j, k] == 1:
                        temperature = full_soln[i, j, k]
                        app_diff.eval_at_T(temperature)
                        initial_diffs[i, j, k] = app_diff.kappa_eff
        r_values = dm.spvr_3d_initialise_r_vector(dt, dx, initial_diffs)
        print(f"{h}/{iterations}")
        for j in range(len(y)):
            for k in range(len(z)):
                temps = full_soln[:, j, k]
                # diffs = initial_diffs[:, j, k]
                r_vals = r_values[:, j, k]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Nx)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Nx,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Nx)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Nx)
                system = dm.LinearSystemCN(A, B, b1, b2, x, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, x, 1, u)
                full_soln[:, j, k] = solution.flatten()
        # print(solution)

        for i in range(len(x)):
            for k in range(len(z)):
                # print(f"diffs[{i},:,{k}] = {initial_diffs[i, :, k]}")
                temps = full_soln[i, :, k]
                # diffs = initial_diffs[i, :, k]
                r_vals = r_values[i, :, k]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Ny)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Ny,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Ny)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Ny)
                system = dm.LinearSystemCN(A, B, b1, b2, y, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, y, 1, u)
                full_soln[i, :, k] = solution.flatten()
        # print(solution)

        for i in range(len(x)):
            for j in range(len(y)):
                temps = full_soln[i, j, :]
                # diffs = initial_diffs[i, j, :]
                r_vals = r_values[i, j, :]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Nz)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Nz,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Nz)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Nz)
                system = dm.LinearSystemCN(A, B, b1, b2, z, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, z, 1, u)
                full_soln[i, j, :] = solution.flatten()
        # print(solution)
        filename = f"{folder}{fileID}_{h}"
        if h % save_iter == 0:
            print(f"saving test_{h}")
            np.save(filename, full_soln)


def v8_iter_func(initial_temps,
                 initial_diffs: object,
                 location_of_intrusion,
                 app_diff,
                 boundary_cond,
                 x, y, z,
                 dx, dy, dz,
                 Nx, Ny, Nz,
                 dt,
                 folder,
                 iterations=4,
                 save_iter=1,
                 fileID="results",
                 iter_list=None,):
    if iter_list is None:
        iter_list = [3, 48, 96, 120]
    full_soln = initial_temps
    assert dx == dy == dz

    filename = f"{folder}{fileID}_0"
    print("saving initial conditions")
    np.save(filename, full_soln)
    print(iter_list)

    for h in range(1, iterations + 1):
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    if location_of_intrusion[i, j, k] == 1:
                        temperature = full_soln[i, j, k]
                        app_diff.eval_at_T(temperature)
                        initial_diffs[i, j, k] = app_diff.kappa_eff
        r_values = dm.spvr_3d_initialise_r_vector(dt, dx, initial_diffs)
        print(f"{h}/{iterations}")
        for j in range(len(y)):
            for k in range(len(z)):
                temps = full_soln[:, j, k]
                # diffs = initial_diffs[:, j, k]
                r_vals = r_values[:, j, k]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Nx)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Nx,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Nx)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Nx)
                system = dm.LinearSystemCN(A, B, b1, b2, x, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, x, 1, u)
                full_soln[:, j, k] = solution.flatten()
        # print(solution)

        for i in range(len(x)):
            for k in range(len(z)):
                # print(f"diffs[{i},:,{k}] = {initial_diffs[i, :, k]}")
                temps = full_soln[i, :, k]
                # diffs = initial_diffs[i, :, k]
                r_vals = r_values[i, :, k]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Ny)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Ny,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Ny)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Ny)
                system = dm.LinearSystemCN(A, B, b1, b2, y, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, y, 1, u)
                full_soln[i, :, k] = solution.flatten()
        # print(solution)

        for i in range(len(x)):
            for j in range(len(y)):
                temps = full_soln[i, j, :]
                # diffs = initial_diffs[i, j, :]
                r_vals = r_values[i, j, :]
                (main_diag_A,
                 upper_diag_A,
                 lower_diag_A,
                 main_diag_B,
                 upper_diag_B,
                 lower_diag_B) = dm.spvr_initialise_diagonals(r_vals, Nz)
                A, B = dm.initialise_matrices_var_dif(main_diag_A,
                                                      upper_diag_A,
                                                      lower_diag_A,
                                                      main_diag_B,
                                                      upper_diag_B,
                                                      lower_diag_B, Nz,
                                                      boundary_cond=boundary_cond)
                if boundary_cond == 'd':
                    u, b1, b2 = dm.IBCs_Dirichlet(temps, r_vals, Nz)
                else:
                    u, b1, b2 = dm.IBCs_Neumann_zero_flux(temps, r_vals, Nz)
                system = dm.LinearSystemCN(A, B, b1, b2, z, u, bc=boundary_cond)
                solution = dm.cn_solver_zero(system, z, 1, u)
                full_soln[i, j, :] = solution.flatten()
        # print(solution)
        filename = f"{folder}{fileID}_{h}"
        if h in [iter_list[0], iter_list[1], iter_list[2], iter_list[3]]:
            print(f"saving test_{h}")
            np.save(filename, full_soln)
