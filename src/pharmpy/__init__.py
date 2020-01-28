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

from .model_factory import Model

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['Model']
