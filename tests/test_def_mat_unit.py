#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/10/2021
by murphyqm

"""
import numpy as np

# from context import define_matrix as dm
import setup_path
import metintrusion.define_matrix as dm

import pytest


def test_spacing():
    Nx = 200
    Ny = 200
    Nz = 200

    Lx = 400
    Ly = 400
    Lz = 400

    dx = dm.spacing(Nx, Lx)
    dy = dm.spacing(Ny, Ly)
    dz = dm.spacing(Nz, Lz)

    assert dx == pytest.approx(2.0100502512562812)
    assert dy == pytest.approx(2.0100502512562812)
    assert dz == pytest.approx(2.0100502512562812)


def test_spacing2():
    Nx = 201
    Ny = 201
    Nz = 201

    Lx = 400
    Ly = 400
    Lz = 400

    dx = dm.spacing(Nx, Lx)
    dy = dm.spacing(Ny, Ly)
    dz = dm.spacing(Nz, Lz)

    assert dx == pytest.approx(2.0)
    assert dy == pytest.approx(2.0)
    assert dz == pytest.approx(2.0)


def test_grid():
    Nx = 201
    Ny = 201
    Nz = 201

    Lx = 400
    Ly = 400
    Lz = 400

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)
    assert x[-1] == pytest.approx(400.0)
    assert y[-1] == pytest.approx(400.0)
    assert z[-1] == pytest.approx(400.0)
    assert y[-2] == pytest.approx(398.0)
    assert z[-3] == pytest.approx(396.0)
    assert x[2] == pytest.approx(4.0)
    assert x.shape == (201,)
    assert blank_vol.shape == (201, 201, 201)


def test_setting_grid_vals():
    Nx = 5
    Ny = 5
    Nz = 5

    Lx = 400
    Ly = 400
    Lz = 400

    dx = dm.spacing(Nx, Lx)
    dy = dm.spacing(Ny, Ly)
    dz = dm.spacing(Nz, Lz)
    print(dx, dy, dz)

    x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

    x1 = 133
    x2 = 266
    y1 = 133
    y2 = 266
    z1 = 133
    z2 = 266

    interior = 1
    exterior = 0

    location_of_intrusion = dm.set_grid_values_3d(x, y, z, blank_vol,
                                                  x1, x2, y1, y2, z1, z2,
                                                  interior, exterior)
    test_array = np.array([[[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],
                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],
                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],
                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]],
                           [[0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0.]]])

    np.testing.assert_equal(location_of_intrusion, test_array)

