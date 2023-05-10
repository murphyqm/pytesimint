#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/04/2021
Set pythonpath so package import works for uninstalled package
Copied from `pytesimal`
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../")),
)