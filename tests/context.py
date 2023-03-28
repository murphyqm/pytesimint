#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context file to allow import of main module.

Created on Tue Feb  2 13:56:32 2021. (pytesimal)

@author: maeve

Modified Tue Mar 28 2023 for use with pytesimint
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../pytesimint/")),
)

import analyse_results
import define_matrix
import effective_diffusivity
import iteration
import PRAM