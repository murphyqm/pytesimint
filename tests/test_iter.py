#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/10/2021
by murphyqm

"""
import numpy as np

# from context import iteration as it

import setup_path
import metintrusion.iteration as it
import pytest


def test_test():
    space = it.example_func()
    assert space == pytest.approx(1.25)
