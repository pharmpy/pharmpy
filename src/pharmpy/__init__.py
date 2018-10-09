#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
=================================
PharmPy -- Python Pharmacometrics
=================================

PharmPy is an experiment in substituting (some of) the PsN functionality in Python.

Definitions
===========
"""

__version__ = '0.1.0'

import logging

from .model import Model

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['Model']
