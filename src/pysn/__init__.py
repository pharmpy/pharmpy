#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

__version__ = '0.1.0'

import logging

from .model import Model

logging.getLogger(__name__).addHandler(logging.NullHandler())

__authors__ = [
    'Rikard Nordgren <rikard.nordgren@farmbio.uu.se>',
    'Gunnar Yngman <gunnar.yngman@farmbio.uu.se>',
]

__all__ = ['Model']
