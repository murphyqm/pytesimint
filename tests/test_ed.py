#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/10/2021
by murphyqm

"""
# import numpy as np

# from context import effective_diffusivity as ed

import setup_path
import metintrusion.effective_diffusivity as ed
import pytest


def test_freezing_func():
    # Instantiate theta function object
    T_L = 1600  # Wasson and Choi, 2003
    T_S = 1000

    freezing_func = ed.FreezingFunction(T_L, T_S)

    theta_at_1000 = freezing_func.calc_theta(1000)
    dtheta_at_1000 = freezing_func.deriv_theta(1000)

    theta_at_1200 = freezing_func.calc_theta(1200)
    dtheta_at_1200 = freezing_func.deriv_theta(1200)

    freezing_func.fraction(1000)

    assert theta_at_1000 == pytest.approx(freezing_func.theta)
    assert dtheta_at_1000 == pytest.approx(freezing_func.dtheta)

    assert theta_at_1200 == pytest.approx(0.1690133154060661)
    assert dtheta_at_1200 == pytest.approx(0.001502340581387254)

    assert freezing_func.theta == pytest.approx(0.01831563888873418)
    assert freezing_func.dtheta == pytest.approx(0.00024420851851645573)


def test_appdiff():
    metal_fraction = 0.3
    cond_metal_s = 30  # changed this from 30
    cond_metal_l = 40
    cond_olivine = 3.0
    dens_liq_metal = 7020
    dens_solid_metal = 7500
    dens_olivine = 3341
    heat_cap_liq_metal = 835
    heat_cap_solid_metal = 835
    heat_cap_ol = 819
    latent_heat = 2.56e5

    # Instantiate theta function object
    T_L = 1600  # Wasson and Choi, 2003
    T_S = 1000

    freezing_func = ed.FreezingFunction(T_L, T_S)

    app_diff = ed.nnnAppDiff(
        metal_fraction,
        cond_metal_s,
        cond_metal_l,
        cond_olivine,
        dens_liq_metal,
        dens_solid_metal,
        dens_olivine,
        heat_cap_liq_metal,
        heat_cap_solid_metal,
        heat_cap_ol,
        latent_heat,
        freezing_func,
    )

    assert app_diff.phi_liq == pytest.approx(0.3)
    app_diff.eval_at_T(1900)
    assert app_diff.app_heat_cap == pytest.approx(3673905.3)
    assert app_diff.k_eff == pytest.approx(9.670869483043395)
    assert app_diff.phi_liq == pytest.approx(0.3)
    assert app_diff.phi_sol == pytest.approx(0.0)

    app_diff.eval_at_T(1000)
    assert app_diff.app_heat_cap == pytest.approx(3923604.631418906)
    assert app_diff.k_eff == pytest.approx(8.181081853151651)
    assert app_diff.phi_liq == pytest.approx(0.005494691666620253)
    assert app_diff.phi_sol == pytest.approx(0.29450530833337973)
