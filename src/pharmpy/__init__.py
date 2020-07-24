"""
=======
Pharmpy
=======

Pharmpy is a python package for pharmacometrics modelling.

Definitions
===========
"""

__version__ = '0.4.0'

import logging

from .model_factory import Model
from .parameter import Parameter, ParameterSet

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['Model', 'Parameter', 'ParameterSet']
