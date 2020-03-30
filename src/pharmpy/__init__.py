"""
=================================
Pharmpy -- Python Pharmacometrics
=================================

Pharmpy is an experiment in substituting (some of) the PsN functionality in Python.

Definitions
===========
"""

__version__ = '0.2.0'

import logging

from .model_factory import Model
from .parameter import Parameter, ParameterSet

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['Model', 'Parameter', 'ParameterSet']
