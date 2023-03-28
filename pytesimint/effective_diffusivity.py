#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/09/2021
by murphyqm

Module for the effective diffusivity, see notes "crystallisation
in a linear scheme".

Diffusivity takes into account apparent heat capacity, which tracks
latent heat during crystallisation

"""

import numpy as np
from functools import wraps
import time
from numba.experimental import jitclass
from numba import int32, float64, typeof    # import the types
from numba import types
# from numba import pyobject # , pyfunc_type

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


# def freezing_function(T):
#     """Placeholder function for freezing"""
#     y = 2 * T
#     return y
#
#
# class testing_callables:
#     """Testing callables as arguments"""
#
#     def __init__(self, option, function):
#         self.value = option
#         self.function = function
#
#     def eval_func(self, inp):
#         return self.function(inp)
#
#     def save_func_value(self, inp):
#         self.new_val = self.function(inp)

spec = [
    ('T_L', float64),
    ('T_S', float64),
    ('w', float64),
    ('theta', float64),
    ('dtheta', float64),
]

# so adding spec and jitclass changes how the floating points are handled.
# Setting up a long run on arc to see if this impacts it over the timescale
# we are interested in. For a few T-steps, the effect is v small...
# this is used!
# @timefn
@jitclass(spec)  # defining dtype is causing slight difference here in results
class FreezingFunction:
    def __init__(self, T_L, T_S):
        self.T_L = T_L
        self.T_S = T_S
        self.w = np.abs((T_L - T_S) / 2.0)
        self.theta = 1.0
        self.dtheta = 0.0

    def calc_theta(self, T):
        if T < self.T_L:
            self.theta = np.exp(-((T - self.T_L) / self.w) ** 2)
        else:
            self.theta = 1.0
        return self.theta

    def deriv_theta(self, T):
        if T < self.T_L:
            self.dtheta = -2 * (T - self.T_L) / (self.w ** 2) * \
                          np.exp(-((T - self.T_L) / self.w) ** 2)
        else:
            self.dtheta = 0.0
        return self.dtheta

    def fraction(self, T):
        if T < self.T_L:
            self.theta = np.exp(-((T - self.T_L) / self.w) ** 2)
            self.dtheta = -2 * (T - self.T_L) / (self.w ** 2) * \
                np.exp(-((T - self.T_L) / self.w) ** 2)
        else:
            self.theta = 1.0
            self.dtheta = 0.0


# class OldApparentDiffusivity:
#     """Apparent diffusivity to track freezing."""

#     def __init__(self, metal_fraction, liquid_metal_fraction,
#                  solid_metal_fraction, cond_metal, cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, freezing_function):
#         self.phi_m = metal_fraction
#         self.phi_l = liquid_metal_fraction  # so these need to change
#         self.phi_s = solid_metal_fraction  # dependent on temperature
#         self.k_m = cond_metal
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.dTheta_dT = freezing_function

#         self.k_eff = (self.phi_m * self.k_m) + (1 - self.phi_m) * self.k_ol  # this is constant

#     def eval_at_T(self, T):
#         self.dTheta_dT_value = self.dTheta_dT(T)
#         self.rho_c_app = (self.phi_m((self.phi_l * self.rho_l * self.c_l)
#                                      + (self.phi_s * self.rho_s * self.c_s)
#                                      + (self.rho_s * self.L *
#                                         self.dTheta_dT_value))
#                           + ((1 - self.phi_m) * (self.rho_ol * self.c_ol)))
#         self.kappa_eff = self.k_eff / self.rho_c_app

#         return self.kappa_eff


# class ApparentDiffusivityRough:
#     """Apparent diffusivity to track freezing."""

#     def __init__(self, metal_fraction, cond_metal, cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, freezing_function):
#         self.phi_m = metal_fraction

#         self.k_m = cond_metal
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.freezing_func = freezing_function

#         self.phi_l = 1 # liquid_metal_fraction  # so these need to change
#         self.phi_s = 2 # solid_metal_fraction  # dependent on temperature

#         self.k_eff = (self.phi_m * self.k_m) + (1 - self.phi_m) * self.k_ol  # this is constant

#     def eval_at_T(self, T):
#         self.dTheta_dT_value = self.dTheta_dT(T)
#         self.rho_c_app = (self.phi_m((self.phi_l * self.rho_l * self.c_l)
#                                      + (self.phi_s * self.rho_s * self.c_s)
#                                      + (self.rho_s * self.L *
#                                         self.dTheta_dT_value))
#                           + ((1 - self.phi_m) * (self.rho_ol * self.c_ol)))
#         self.kappa_eff = self.k_eff / self.rho_c_app

#         return self.kappa_eff


# class ApparentDiffusivity:
#     """Class to track freezing through apparent diffusivity."""

#     def __init__(self, metal_fraction, cond_metal, cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, function):
#         self.phi = metal_fraction
#         self.k_m = cond_metal
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.func = function
#         self.phi_ol = 1 - self.phi  # does not change
#         self.phi_liq = self.phi * self.func.theta  # variable
#         self.phi_sol = self.phi - self.phi_liq  # variable

#         # constant effective conductivity:
#         # self.k_eff = (self.phi * self.k_m) + (1 - self.phi) * self.k_ol
#         # instead of arithmetic mean or geometric, need the square-root mean
#         self.k_eff = ((self.phi * np.sqrt(self.k_m)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                              (self.phi_sol * self.rho_s * self.c_s) +
#                              self.rho_s * self.L * self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap

#     def eval_at_T(self, T):
#         self.func.fraction(T)
#         self.phi_liq = self.phi * self.func.theta
#         self.phi_sol = self.phi - self.phi_liq

#         self.k_eff = ((self.phi * np.sqrt(self.k_m)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                               (self.phi_sol * self.rho_s * self.c_s) +
#                               self.rho_s * self.L * self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap


# class ApparentDiffusivityWTDepk:
#     """Class to track freezing through apparent diffusivity."""

#     def __init__(self, metal_fraction, cond_metal_s, cond_metal_l,
#                  cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, function):
#         self.phi = metal_fraction
#         self.k_ms = cond_metal_s
#         self.k_ml = cond_metal_l
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.func = function
#         self.phi_ol = 1 - self.phi  # does not change
#         self.phi_liq = self.phi * self.func.theta  # variable
#         self.phi_sol = self.phi - self.phi_liq  # variable

#         # constant effective conductivity:
#         # self.k_eff = (self.phi * self.k_m) + (1 - self.phi) * self.k_ol
#         # instead of arithmetic mean or geometric, need the square-root mean
#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                              (self.phi_sol * self.rho_s * self.c_s) +
#                              self.rho_s * self.L * self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap

#     def eval_at_T(self, T):
#         self.func.fraction(T)
#         self.phi_liq = self.phi * self.func.theta
#         self.phi_sol = self.phi - self.phi_liq

#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                               (self.phi_sol * self.rho_s * self.c_s) +
#                               self.rho_s * self.L * self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap


# class newApparentDiffusivityWTDepk:
#     """Class to track freezing through apparent diffusivity."""

#     def __init__(self, metal_fraction, cond_metal_s, cond_metal_l,
#                  cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, function):
#         self.phi = metal_fraction
#         self.k_ms = cond_metal_s
#         self.k_ml = cond_metal_l
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.func = function
#         self.phi_ol = 1 - self.phi  # does not change
#         self.phi_liq = self.phi * self.func.theta  # variable
#         self.phi_sol = self.phi - self.phi_liq  # variable

#         # constant effective conductivity:
#         # self.k_eff = (self.phi * self.k_m) + (1 - self.phi) * self.k_ol
#         # instead of arithmetic mean or geometric, need the square-root mean
#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                              (self.phi_sol * self.rho_s * self.c_s) +
#                              self.rho_s * self.L * (self.phi) *
#                              self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap

#     def eval_at_T(self, T):
#         self.func.fraction(T)
#         self.phi_liq = self.phi * self.func.theta
#         self.phi_sol = self.phi - self.phi_liq

#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                               (self.phi_sol * self.rho_s * self.c_s) -
#                               self.rho_s * self.L * -self.phi *
#                               self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap


# class newnewApparentDiffusivityWTDepk:
#     """Class to track freezing through apparent diffusivity."""

#     def __init__(self, metal_fraction, cond_metal_s, cond_metal_l,
#                  cond_olivine,
#                  dens_liq_metal, dens_solid_metal, dens_olivine,
#                  heat_cap_liq_metal, heat_cap_solid_metal,
#                  heat_cap_ol, latent_heat, function):
#         self.phi = metal_fraction
#         self.k_ms = cond_metal_s
#         self.k_ml = cond_metal_l
#         self.k_ol = cond_olivine
#         self.rho_l = dens_liq_metal
#         self.rho_s = dens_solid_metal
#         self.rho_ol = dens_olivine
#         self.c_l = heat_cap_liq_metal
#         self.c_s = heat_cap_solid_metal
#         self.c_ol = heat_cap_ol
#         self.L = latent_heat
#         self.func = function
#         self.phi_ol = 1 - self.phi  # does not change
#         self.phi_liq = self.phi * self.func.theta  # variable
#         self.phi_sol = self.phi - self.phi_liq  # variable

#         # constant effective conductivity:
#         # self.k_eff = (self.phi * self.k_m) + (1 - self.phi) * self.k_ol
#         # instead of arithmetic mean or geometric, need the square-root mean
#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                              (self.phi_sol * self.rho_s * self.c_s) +
#                              self.rho_s * self.L * (-self.phi) *
#                              self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap

#     def eval_at_T(self, T):
#         self.func.fraction(T)
#         self.phi_liq = self.phi * self.func.theta
#         self.phi_sol = self.phi - self.phi_liq

#         self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
#                       (self.phi_liq * np.sqrt(self.k_ml)) +
#                       (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

#         self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
#                               (self.phi_sol * self.rho_s * self.c_s) +
#                               self.rho_s * self.L * (-self.phi) *
#                               self.func.dtheta) +
#                              ((1 - self.phi) * self.rho_ol * self.c_ol))

#         self.kappa_eff = self.k_eff / self.app_heat_cap


spec_app = [
    ('phi', float64),
    ('k_ms', float64),
    ('k_ml', float64),
    ('k_ol', float64),
    ('rho_l', float64),
    ('rho_s', float64),
    ('rho_ol', float64),
    ('c_l', float64),
    ('c_s', float64),
    ('c_ol', float64),
    ('L', float64),
    ('func', types.pyobject),
    ('phi_ol', float64),
    ('phi_liq', types.pyobject),
    ('phi_sol', types.pyobject),
    ('k_eff', float64),
    ('app_heat_cap', float64),
    ('kappa_eff', float64),
]
# @timefn
# @jitclass(spec_app)
class nnnAppDiff:
    """Class to track freezing through apparent diffusivity."""

    def __init__(self, metal_fraction, cond_metal_s, cond_metal_l,
                 cond_olivine,
                 dens_liq_metal, dens_solid_metal, dens_olivine,
                 heat_cap_liq_metal, heat_cap_solid_metal,
                 heat_cap_ol, latent_heat, function):
        self.phi = metal_fraction
        self.k_ms = cond_metal_s
        self.k_ml = cond_metal_l
        self.k_ol = cond_olivine
        self.rho_l = dens_liq_metal
        self.rho_s = dens_solid_metal
        self.rho_ol = dens_olivine
        self.c_l = heat_cap_liq_metal
        self.c_s = heat_cap_solid_metal
        self.c_ol = heat_cap_ol
        self.L = latent_heat
        self.func = function
        self.phi_ol = 1.0 - self.phi  # does not change
        self.phi_liq = self.phi * self.func.theta  # variable
        self.phi_sol = self.phi - self.phi_liq  # variable

        # instead of arithmetic mean or geometric, need the square-root mean
        self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
                      (self.phi_liq * np.sqrt(self.k_ml)) +
                      (1.0 - self.phi) * np.sqrt(self.k_ol)) ** 2

        self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
                             (self.phi_sol * self.rho_s * self.c_s) +
                             self.rho_l * self.L * (self.phi) *
                             self.func.dtheta) +
                             ((1.0 - self.phi) * self.rho_ol * self.c_ol))

        self.kappa_eff = self.k_eff / self.app_heat_cap

    def eval_at_T(self, T):
        self.func.fraction(T)
        self.phi_liq = self.phi * self.func.theta
        self.phi_sol = self.phi - self.phi_liq

        self.k_eff = ((self.phi_sol * np.sqrt(self.k_ms)) +
                      (self.phi_liq * np.sqrt(self.k_ml)) +
                      (1 - self.phi) * np.sqrt(self.k_ol)) ** 2

        self.app_heat_cap = (((self.phi_liq * self.rho_l * self.c_l) +
                              (self.phi_sol * self.rho_s * self.c_s) +
                              self.rho_l * self.L * (self.phi) *
                              self.func.dtheta) +
                             ((1 - self.phi) * self.rho_ol * self.c_ol))

        self.kappa_eff = self.k_eff / self.app_heat_cap